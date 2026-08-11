/* Copyright 2026 Man Group Operations Limited
 *
 * Use of this software is governed by the Business Source License 1.1 included in the file licenses/BSL.txt.
 *
 * As of the Change Date specified in that file, in accordance with the Business Source License, use of this software
 * will be governed by the Apache License, version 2.0.
 */

#include <arcticdb/processing/schema_combine.hpp>

#include <arcticdb/entity/type_utils.hpp>
#include <arcticdb/entity/types_proto.hpp>
#include <arcticdb/entity/timeseries_descriptor.hpp>
#include <arcticdb/log/log.hpp>
#include <arcticdb/pipeline/frame_utils.hpp>
#include <arcticdb/pipeline/input_frame.hpp>
#include <arcticdb/util/preconditions.hpp>
#include <arcticdb/util/type_traits.hpp>

#include <iterator>
#include <optional>
#include <ranges>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace arcticdb {

using namespace proto::descriptors;
using entity::DataType;
using entity::Field;
using entity::IndexDescriptorImpl;
using entity::OutputSchema;
using entity::StreamDescriptor;
using entity::TypeDescriptor;

namespace {
using PandasCommon = NormalizationMetadata_Pandas;

// DataFrames, Series and TimeFrames all describe their index through the same Pandas submessage; every other
// input type - an ndarray, a pickled object, an Arrow table - has none, and returns nullptr.
const PandasCommon* pandas_common(const NormalizationMetadata& norm) {
    switch (norm.input_type_case()) {
    case NormalizationMetadata::kDf:
        return &norm.df().common();
    case NormalizationMetadata::kSeries:
        return &norm.series().common();
    case NormalizationMetadata::kTs:
        return &norm.ts().common();
    default:
        return nullptr;
    }
}

PandasCommon* mutable_pandas_common(NormalizationMetadata& norm) {
    switch (norm.input_type_case()) {
    case NormalizationMetadata::kDf:
        return norm.mutable_df()->mutable_common();
    case NormalizationMetadata::kSeries:
        return norm.mutable_series()->mutable_common();
    case NormalizationMetadata::kTs:
        return norm.mutable_ts()->mutable_common();
    default:
        return nullptr;
    }
}

// Whether the metadata describes the empty index a zero-row write produces. Such an index has no name, no
// timezone and no step, and occupies no descriptor field, so there is nothing in it to reconcile with anything.
// Told apart from a range index, which is also not physically stored, by the step: pandas never gives a RangeIndex
// a step of zero. A multi-index leaves the single-index submessage at its defaults, so it has to be excluded.
bool has_empty_index(const NormalizationMetadata& norm) {
    const auto* common = pandas_common(norm);
    return common != nullptr && !common->has_multi_index() && !common->index().is_physically_stored() &&
           common->index().step() == 0;
}

// The shape of a schema's required fields - its index columns, plus its value column if it is a Series. Derived
// once from the combined normalization metadata rather than per schema, because it is the metadata merge that
// decides which shapes may combine at all, and its result describes the output.
struct RequiredFieldInfo {
    bool has_multi_index{false};
    bool has_series_value_column{false};
    // Index levels occupying a leading descriptor field. Zero for a range index, whose name is held only in the
    // normalization metadata. Cannot be derived from the count alone, as a one-level multi-index is legal.
    size_t num_physical_indices{0};

    [[nodiscard]] size_t num_physical_required_columns() const {
        return num_physical_indices + (has_series_value_column ? 1 : 0);
    }

    // How many of a single schema's leading fields are required ones. An empty index occupies no descriptor
    // field, so a schema carrying one contributes none of the combined index levels and every one of its fields
    // is a data column.
    [[nodiscard]] size_t num_required_columns_for(IndexDescriptorImpl::Type index_type) const {
        const size_t indices = index_type == IndexDescriptorImpl::Type::EMPTY ? 0 : num_physical_indices;
        return indices + (has_series_value_column ? 1 : 0);
    }
};

// What disagreed about the names of the required fields, accumulated across every schema being combined.
// Deliberately not keyed by position alone: a range index has no descriptor field and so occupies no position,
// and recording it at position 0 would collide with the Series value column, which is itself at position 0 when
// there are no index levels at all. Keeping the Series separate makes position 0 mean the index unambiguously.
class RequiredNameMismatches {
  public:
    explicit RequiredNameMismatches(const SchemaCombineOptions& options) : options_(options) {}

    // A multi-index's level names are reported as an index incompatibility: they are what keeps the normalization
    // metadata in step with the data, which is why they must match even under dynamic schema. A single index's
    // name is a descriptor field, so a disagreement about it is a descriptor mismatch.
    void add_index(size_t position, bool multi_index) {
        if (raises()) {
            if (multi_index) {
                normalization::raise<ErrorCode::E_INCOMPATIBLE_INDEX>(
                        "Cannot {}: multi-index level names must match", options_.name()
                );
            }
            schema::raise<ErrorCode::E_DESCRIPTOR_MISMATCH>("Cannot {}: Index names must match", options_.name());
        }
        index_positions_.emplace(position);
    }

    void add_series_name() {
        if (raises()) {
            schema::raise<ErrorCode::E_DESCRIPTOR_MISMATCH>("Cannot {}: Series names must match", options_.name());
        }
        series_name_ = true;
    }

    [[nodiscard]] bool index_at(size_t position) const { return index_positions_.contains(position); }

    [[nodiscard]] const std::unordered_set<size_t>& index_positions() const { return index_positions_; }

    [[nodiscard]] bool series_name() const { return series_name_; }

    [[nodiscard]] bool any() const { return series_name_ || !index_positions_.empty(); }

  private:
    // Raising as each disagreement is found, rather than collecting them and raising afterwards, is what keeps a
    // name mismatch from being reported as whatever type clash it implies: two index levels with different names
    // are not the same level, so there is nothing to be gained from reconciling their types.
    [[nodiscard]] bool raises() const { return options_.name_mismatch == RequiredNameMismatchPolicy::Raise; }

    const SchemaCombineOptions& options_;
    std::unordered_set<size_t> index_positions_{};
    bool series_name_{false};
};

// TODO (monday ref 11325694339): arrow is looser than this once it may be combined with pandas - a zero or one
// level arrow index should be compatible with a multi-index, and an arrow table with no value column with a
// Series - which needs the info to distinguish arrow throughout from arrow merged with pandas.
RequiredFieldInfo required_fields_info(const NormalizationMetadata& norm) {
    RequiredFieldInfo info;
    info.has_series_value_column = norm.has_series();
    if (const auto* common = pandas_common(norm); common != nullptr) {
        info.has_multi_index = common->has_multi_index();
        // The field count in the norm metadata is one less than the actual number of levels in the multi-index.
        info.num_physical_indices = info.has_multi_index                     ? common->multi_index().field_count() + 1
                                    : common->index().is_physically_stored() ? 1
                                                                             : 0;
    } else if (norm.has_experimental_arrow()) {
        info.num_physical_indices = norm.experimental_arrow().has_index() ? 1 : 0;
    }
    return info;
}

// ---------------------------------------------------------------------------------------------------------------------
// Field types
// ---------------------------------------------------------------------------------------------------------------------

// Reconcile the type of a field present in more than one schema, returning nullopt when there is no acceptable
// common type. Callers that must have a type turn that into an error; the inner join defers, because a column
// with no common type does not matter if it does not reach the output.
std::optional<TypeDescriptor> try_combine_field_type(
        const TypeDescriptor& base, const TypeDescriptor& other, TypePromotionPolicy policy
) {
    if (base == other) {
        return base;
    }
    switch (policy) {
    case TypePromotionPolicy::Dynamic:
        return has_valid_common_type(base, other);
    case TypePromotionPolicy::MostPermissive:
        return promotable_type(base, other);
    case TypePromotionPolicy::Static:
        // Only empty->concrete and fixed->dynamic string promotions are allowed; anything else has to be
        // byte-compatible already.
        if (is_empty_type(base.data_type())) {
            return other;
        }
        if (is_empty_type(other.data_type())) {
            return base;
        }
        if (is_sequence_type(base.data_type()) && is_sequence_type(other.data_type()) &&
            is_dynamic_string_type(base.data_type()) != is_dynamic_string_type(other.data_type())) {
            return is_dynamic_string_type(base.data_type()) ? base : other;
        }
        return trivially_compatible_types(base, other) ? std::optional{base} : std::nullopt;
    }
    return std::nullopt;
}

TypeDescriptor combine_field_type(
        const TypeDescriptor& base, const TypeDescriptor& other, const SchemaCombineOptions& options,
        std::string_view name
) {
    const auto combined = try_combine_field_type(base, other, options.type_promotion);
    schema::check<ErrorCode::E_DESCRIPTOR_MISMATCH>(
            combined.has_value(),
            "Cannot {} column {}: no common type between {} and {}",
            options.name(),
            name,
            base,
            other
    );
    return *combined;
}

// ---------------------------------------------------------------------------------------------------------------------
// Stream descriptor
// ---------------------------------------------------------------------------------------------------------------------

// The index type and field count must match across every schema, save that an empty index - which only ever
// describes zero rows, and so says nothing about what the index is - takes on whichever concrete index it meets.
IndexDescriptorImpl combine_index_descriptors(
        std::span<const OutputSchema> schemas, const SchemaCombineOptions& options
) {
    const auto is_empty = [](const IndexDescriptorImpl& index) {
        return index.type() == IndexDescriptorImpl::Type::EMPTY;
    };
    auto result = schemas.front().stream_descriptor().index();
    for (const auto& schema : schemas.subspan(1)) {
        const auto& other = schema.stream_descriptor().index();
        if (is_empty(result)) {
            result = other;
            continue;
        }
        if (is_empty(other)) {
            continue;
        }
        normalization::check<ErrorCode::E_INCOMPATIBLE_INDEX>(
                result.type() == other.type(),
                "Cannot {} {} index to {} index",
                options.name(),
                index_type_to_str(other.type()),
                index_type_to_str(result.type())
        );
        normalization::check<ErrorCode::E_INCOMPATIBLE_INDEX>(
                result.field_count() == other.field_count(),
                "Cannot {}: mismatching index field count, {} and {}",
                options.name(),
                result.field_count(),
                other.field_count()
        );
    }
    return result;
}

// Merge the required fields - the index levels, plus the value column for a Series - which are always the
// leading fields of the descriptor. A schema with an empty index has fewer of them than the output does, so its
// fields line up with the trailing ones: an empty index contributes no levels, but a Series value column still
// does. Name disagreements are recorded rather than applied, so that apply_required_name_mismatches can keep the
// descriptor field names and the normalization metadata in step.
void add_required_fields(
        StreamDescriptor& out, std::span<const OutputSchema> schemas, const RequiredFieldInfo& info,
        RequiredNameMismatches& mismatches, const SchemaCombineOptions& options
) {
    const auto required_fields = info.num_physical_required_columns();
    // Required fields never take the float64 fallback: index levels and the Series value column have to
    // represent their values exactly, even where the data columns are allowed to lose precision.
    auto required_field_options = options;
    if (required_field_options.type_promotion == TypePromotionPolicy::MostPermissive) {
        required_field_options.type_promotion = TypePromotionPolicy::Dynamic;
    }
    // A position stays unset until a schema that has that field is reached, which for an index level is any
    // schema whose index is not the empty one.
    std::vector<std::optional<FieldRef>> fields(required_fields);
    for (const auto& schema : schemas) {
        const auto& desc = schema.stream_descriptor();
        const auto required_for_schema = info.num_required_columns_for(desc.index().type());
        const auto skipped_levels = required_fields - required_for_schema;
        normalization::check<ErrorCode::E_INCOMPATIBLE_INDEX>(
                desc.field_count() >= required_for_schema,
                "Cannot {}: expected at least {} required fields, but received {}",
                options.name(),
                required_for_schema,
                desc.field_count()
        );
        for (size_t idx = 0; idx < required_for_schema; ++idx) {
            const auto& field = desc.field(idx);
            auto& combined = fields[idx + skipped_levels];
            if (!combined.has_value()) {
                combined = field.ref();
                continue;
            }
            if (combined->name() != field.name()) {
                if (idx + skipped_levels < info.num_physical_indices) {
                    mismatches.add_index(idx + skipped_levels, info.has_multi_index);
                } else {
                    mismatches.add_series_name();
                }
            }
            combined->type_ = combine_field_type(combined->type(), field.type(), required_field_options, field.name());
        }
    }
    for (size_t idx = 0; idx < required_fields; ++idx) {
        internal::check<ErrorCode::E_ASSERTION_FAILURE>(
                fields[idx].has_value(), "No schema described required field {} of {}", idx, required_fields
        );
        const auto& field = *fields[idx];
        const bool unnamed = idx < info.num_physical_indices ? mismatches.index_at(idx) : mismatches.series_name();
        if (unnamed) {
            // Use the same naming scheme as _normalization.py does for unnamed multiindex levels, so that
            // later processing which looks for columns of that form keeps working.
            out.fields().add_field(field.type(), idx == 0 ? "index" : fmt::format("__fkidx__{}", idx));
        } else {
            out.add_scalar_field(field.type().data_type(), field.name());
        }
    }
}

// The non-index columns of a schema, in descriptor order. A view rather than a copy, so the names below are
// string_views into descriptors owned by the schemas, which outlive every use here.
auto data_columns(const StreamDescriptor& desc, const RequiredFieldInfo& info) {
    return desc.fields() | std::views::drop(info.num_required_columns_for(desc.index().type()));
}

size_t data_column_count(const StreamDescriptor& desc, const RequiredFieldInfo& info) {
    return desc.field_count() - info.num_required_columns_for(desc.index().type());
}

// Every schema must carry the same non-index columns, in the same order.
void add_data_columns_static(
        StreamDescriptor& out, std::span<const OutputSchema> schemas, const RequiredFieldInfo& info,
        const SchemaCombineOptions& options
) {
    const auto& base = schemas.front().stream_descriptor();
    const auto column_count = data_column_count(base, info);
    const auto num_required_for_base = info.num_required_columns_for(base.index().type());
    std::vector<TypeDescriptor> types;
    types.reserve(column_count);
    for (const auto& field : data_columns(base, info)) {
        types.emplace_back(field.type());
    }
    for (const auto& schema : schemas.subspan(1)) {
        const auto& desc = schema.stream_descriptor();
        schema::check<ErrorCode::E_DESCRIPTOR_MISMATCH>(
                data_column_count(desc, info) == column_count,
                "Cannot {}: mismatching column count, {} and {}",
                options.name(),
                column_count,
                data_column_count(desc, info)
        );
        const auto num_required_for_schema = info.num_required_columns_for(desc.index().type());
        for (size_t idx = 0; idx < column_count; ++idx) {
            const auto name = base.field(idx + num_required_for_base).name();
            const auto& field = desc.field(idx + num_required_for_schema);
            schema::check<ErrorCode::E_DESCRIPTOR_MISMATCH>(
                    name == field.name(),
                    "Cannot {}: mismatching column name, {} and {}",
                    options.name(),
                    name,
                    field.name()
            );
            types[idx] = combine_field_type(types[idx], field.type(), options, name);
        }
    }
    for (size_t idx = 0; idx < column_count; ++idx) {
        out.add_scalar_field(types[idx].data_type(), base.field(idx + num_required_for_base).name());
    }
}

// Keep only the columns present in every schema.
void add_data_columns_intersection(
        StreamDescriptor& out, std::span<const OutputSchema> schemas, const RequiredFieldInfo& info,
        const SchemaCombineOptions& options
) {
    const auto& base = schemas.front().stream_descriptor();
    // The combined type is optional because two schemas may disagree irreconcilably about a column that a
    // third schema does not have at all, in which case the column is dropped and the clash is irrelevant.
    // Cannot use ankerl::unordered_dense as its iterators are not stable across erase.
    std::unordered_map<std::string_view, std::optional<TypeDescriptor>> columns_to_keep;
    for (const auto& field : data_columns(base, info)) {
        columns_to_keep.emplace(field.name(), field.type());
    }
    for (const auto& schema : schemas.subspan(1)) {
        ankerl::unordered_dense::map<std::string_view, TypeDescriptor> other_columns;
        for (const auto& field : data_columns(schema.stream_descriptor(), info)) {
            other_columns.emplace(field.name(), field.type());
        }
        for (auto it = columns_to_keep.begin(); it != columns_to_keep.end();) {
            const auto other_it = other_columns.find(it->first);
            if (other_it == other_columns.end()) {
                it = columns_to_keep.erase(it);
                continue;
            }
            if (it->second.has_value()) {
                it->second = try_combine_field_type(*it->second, other_it->second, options.type_promotion);
            }
            ++it;
        }
    }
    // Everything retained was present in every schema, so emit it in the base schema's order.
    for (const auto& field : data_columns(base, info)) {
        if (const auto it = columns_to_keep.find(field.name()); it != columns_to_keep.end()) {
            schema::check<ErrorCode::E_DESCRIPTOR_MISMATCH>(
                    it->second.has_value(), "Cannot {} column {}: no common type", options.name(), field.name()
            );
            out.add_scalar_field(it->second->data_type(), field.name());
        }
    }
}

// Keep the union of the columns, in the order they are first seen.
void add_data_columns_union(
        StreamDescriptor& out, std::span<const OutputSchema> schemas, const RequiredFieldInfo& info,
        const SchemaCombineOptions& options
) {
    ankerl::unordered_dense::map<std::string_view, TypeDescriptor> columns_to_keep;
    // Maintain the order in which the columns first appeared across the schemas.
    std::vector<std::string_view> column_names_to_keep;
    for (const auto& schema : schemas) {
        for (const auto& field : data_columns(schema.stream_descriptor(), info)) {
            if (const auto [it, inserted] = columns_to_keep.try_emplace(field.name(), field.type()); inserted) {
                column_names_to_keep.emplace_back(field.name());
            } else {
                it->second = combine_field_type(it->second, field.type(), options, field.name());
            }
        }
    }
    for (const auto name : column_names_to_keep) {
        out.add_scalar_field(columns_to_keep.at(name).data_type(), name);
    }
}

void add_data_columns(
        StreamDescriptor& out, std::span<const OutputSchema> schemas, const RequiredFieldInfo& info,
        const SchemaCombineOptions& options
) {
    // The three share only the per-schema column extraction, already factored into data_columns. Their control
    // flow differs enough - positional equality, erase-on-absent, insert-on-new - that merging them behind
    // flags reads worse than leaving them apart.
    switch (options.missing_column) {
    case MissingColumnPolicy::Strict:
        add_data_columns_static(out, schemas, info, options);
        return;
    case MissingColumnPolicy::Drop:
        add_data_columns_intersection(out, schemas, info, options);
        return;
    case MissingColumnPolicy::Keep:
        add_data_columns_union(out, schemas, info, options);
        return;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Normalization metadata
// ---------------------------------------------------------------------------------------------------------------------

bool operator==(
        const NormalizationMetadata_Pandas_ColumnName& lhs, const NormalizationMetadata_Pandas_ColumnName& rhs
) {
    return lhs.is_empty() == rhs.is_empty() && lhs.is_int() == rhs.is_int() && lhs.is_none() == rhs.is_none() &&
           lhs.original_name() == rhs.original_name();
}

template<typename ColumnNameMapParent>
requires util::any_of<
        ColumnNameMapParent, NormalizationMetadata_NormalisedTimeSeries, NormalizationMetadata_PandasDataFrame>
void accumulate_norm_metadata_column_names(ColumnNameMapParent& accumulated, const ColumnNameMapParent& new_entry) {
    accumulated.set_has_synthetic_columns(accumulated.has_synthetic_columns() && new_entry.has_synthetic_columns());
    auto* accumulated_col_names = accumulated.mutable_common()->mutable_col_names();
    for (auto& [col_name, col_name_info] : new_entry.common().col_names()) {
        if (const auto it = accumulated_col_names->find(col_name); it != accumulated_col_names->end()) {
            normalization::check<ErrorCode::E_INCOMPATIBLE_OBJECTS>(
                    it->second == col_name_info,
                    "Merging column name normalization for column: \"{}\" does not allow different ColumnName "
                    "settings for columns named the same way.",
                    col_name
            );
        }
    }
    accumulated_col_names->insert(new_entry.common().col_names().begin(), new_entry.common().col_names().end());
}

void accumulate_norm_metadata_column_names(NormalizationMetadata& accumulated, const NormalizationMetadata& new_entry) {
    if (accumulated.has_df()) {
        ARCTICDB_DEBUG_CHECK(
                ErrorCode::E_ASSERTION_FAILURE,
                new_entry.has_df(),
                "Mismatching normalization metadata types in accumulation"
        );
        accumulate_norm_metadata_column_names(*accumulated.mutable_df(), new_entry.df());
    } else if (accumulated.has_series()) {
        ARCTICDB_DEBUG_CHECK(
                ErrorCode::E_ASSERTION_FAILURE,
                new_entry.has_series(),
                "Mismatching normalization metadata types in accumulation"
        );
        accumulate_norm_metadata_column_names(*accumulated.mutable_series(), new_entry.series());
    }
}

// Whether the metadata describes something whose index and columns can be reconciled with another schema's.
// Everything else - an ndarray, a pickled object - is opaque, and there is nothing to merge but the row count.
bool has_arrow_or_pandas(const NormalizationMetadata& norm) {
    return norm.has_experimental_arrow() || pandas_common(norm) != nullptr;
}

// The kind of object as the normalization metadata names it - "df", "series", "np", "msg_pack_frame" - which is what
// a user comparing two error messages needs, rather than the oneof's tag number.
std::string input_type_name(const NormalizationMetadata& norm) {
    const auto* field = NormalizationMetadata::descriptor()->FindFieldByNumber(norm.input_type_case());
    return field != nullptr ? field->name() : "unset";
}

void check_same_input_type(
        const NormalizationMetadata& lhs, const NormalizationMetadata& rhs, const SchemaCombineOptions& options
) {
    if (lhs.input_type_case() == rhs.input_type_case()) {
        return;
    }
    // A Series beside a DataFrame is much the commonest of these, and worth naming as the user would.
    if ((lhs.has_series() && rhs.has_df()) || (lhs.has_df() && rhs.has_series())) {
        normalization::raise<ErrorCode::E_INCOMPATIBLE_OBJECTS>("Cannot {} a Series with a DataFrame", options.name());
    }
    normalization::raise<ErrorCode::E_INCOMPATIBLE_OBJECTS>(
            "Cannot {} data of differing normalization input types: {} and {}",
            options.name(),
            input_type_name(lhs),
            input_type_name(rhs)
    );
}

// The leading dimension of an ndarray is the one row count recorded in normalization metadata rather than
// alongside it, so unlike every other row count it can be, and has to be, combined here.
NormalizationMetadata combine_ndarray_metadata(
        const NormalizationMetadata& accumulated, const NormalizationMetadata& other,
        const SchemaCombineOptions& options
) {
    normalization::check<ErrorCode::E_UPDATE_NOT_SUPPORTED>(
            options.operation != UPDATE, "current normalization scheme doesn't allow update of ndarray"
    );
    const auto& accumulated_shape = accumulated.np().shape();
    const auto& other_shape = other.np().shape();
    normalization::check<ErrorCode::E_WRONG_SHAPE>(
            !accumulated_shape.empty() && !other_shape.empty(),
            "Cannot {}: numpy array normalization metadata has an empty shape",
            options.name()
    );
    normalization::check<ErrorCode::E_WRONG_SHAPE>(
            std::equal(
                    accumulated_shape.begin() + 1, accumulated_shape.end(), other_shape.begin() + 1, other_shape.end()
            ),
            "The appending NDArray must have the same shape as the existing (excl. the first dimension)"
    );
    auto res = accumulated;
    (*res.mutable_np()->mutable_shape())[0] = accumulated_shape[0] + other_shape[0];
    return res;
}

// Pairwise merge of two normalization metadata objects: timezones, RangeIndex start/step, multi-index fields
// and per-column Arrow metadata. Name disagreements are NOT resolved here - they are recorded as required-field
// positions in non_matching_required_names and applied once by apply_required_name_mismatches, so the
// descriptor field names and the normalization metadata cannot drift apart.
NormalizationMetadata accumulate_norm_metadata(
        const NormalizationMetadata& accumulated, const NormalizationMetadata& other,
        RequiredNameMismatches& mismatches, const SchemaCombineOptions& options
) {
    const auto operation = options.name();
    normalization::check<ErrorCode::E_INCOMPATIBLE_OBJECTS>(
            !accumulated.has_msg_pack_frame() && !other.has_msg_pack_frame(), "Cannot {} pickled data", operation
    );
    if (!has_arrow_or_pandas(accumulated) || !has_arrow_or_pandas(other)) {
        check_same_input_type(accumulated, other, options);
        if (accumulated.has_np()) {
            return combine_ndarray_metadata(accumulated, other, options);
        }
        // Preserve append behavior to overwrite with the new schema.
        return other;
    }

    // An empty index says nothing about what the index would have been, so the other schema decides - which for
    // two empty ones is either, as neither carries anything. Done pairwise so that every schema still takes part
    // in the fold, rather than the empty ones having to be filtered out of it first.
    if (has_empty_index(accumulated)) {
        return other;
    }
    if (has_empty_index(other)) {
        return accumulated;
    }

    // Arrow + arrow
    if (accumulated.has_experimental_arrow() && other.has_experimental_arrow()) {
        normalization::check<ErrorCode::E_INCOMPATIBLE_INDEX>(
                accumulated.experimental_arrow().has_index() == other.experimental_arrow().has_index(),
                "Cannot {} indexed arrow data to unindexed arrow data",
                operation
        );
        // Per-column metadata is merged rather than taking only the base schema's, so that a column only a later
        // schema has keeps what that schema says about it.
        auto res = accumulated;
        auto& res_columns = *res.mutable_experimental_arrow()->mutable_columns();
        // A static-schema append or update is the one case where a per-column disagreement - in practice a
        // timestamp column's timezone - is rejected rather than reconciled: the stored column would otherwise
        // silently change meaning, which is exactly what static schema promises not to do.
        const bool reconcile = options.type_promotion != TypePromotionPolicy::Static;
        schema::check<ErrorCode::E_DESCRIPTOR_MISMATCH>(
                reconcile || res_columns.size() == other.experimental_arrow().columns().size(),
                "Cannot {}: arrow column metadata does not match, {} columns carry it against {}",
                options.name(),
                res_columns.size(),
                other.experimental_arrow().columns().size()
        );
        for (const auto& [column_name, other_column] : other.experimental_arrow().columns()) {
            const auto it = res_columns.find(column_name);
            const bool matches = it != res_columns.end() && other_column.timezone() == it->second.timezone();
            schema::check<ErrorCode::E_DESCRIPTOR_MISMATCH>(
                    reconcile || matches,
                    "Cannot {} column {}: arrow column metadata does not match",
                    options.name(),
                    column_name
            );
            if (it == res_columns.end()) {
                res_columns[column_name] = other_column;
            } else if (!matches) {
                // Present in both, so drop anything they disagree on, such as the timezone.
                it->second.clear_timezone();
            }
        }
        return res;
    }

    // TODO (monday ref 11325694339): To be changed when working on arrow with pandas interop
    // One arrow, one pandas: pandas is preferred as it carries more detail. Compatible when
    // arrow.has_index() == pandas.index().is_physically_stored().
    if (accumulated.has_experimental_arrow() || other.has_experimental_arrow()) {
        const auto& arrow_meta = accumulated.has_experimental_arrow() ? accumulated : other;
        const auto& pandas_meta = accumulated.has_experimental_arrow() ? other : accumulated;
        const auto& common = *pandas_common(pandas_meta);
        normalization::check<ErrorCode::E_INCOMPATIBLE_INDEX>(
                common.has_index(), "Cannot {} arrow-written data to multi-indexed pandas data", operation
        );
        normalization::check<ErrorCode::E_INCOMPATIBLE_INDEX>(
                arrow_meta.experimental_arrow().has_index() == common.index().is_physically_stored(),
                "Cannot {} unindexed data to indexed data",
                operation
        );
        return pandas_meta;
    }

    // Pandas + Pandas. A TimeFrame and a DataFrame describe their index alike but denormalize differently, and a
    // Series carries a value column a DataFrame does not, so the kind of object has to agree.
    check_same_input_type(accumulated, other, options);
    auto res = accumulated;
    auto* res_common = mutable_pandas_common(res);
    const auto& other_common = *pandas_common(other);
    // The shape of the index is what decides how many leading descriptor fields are required ones, so it is
    // checked rather than merged: a single index cannot meaningfully become a level of a multi-index, nor a
    // physically stored index a range one. Checked before any name is looked at, because two indices of different
    // shapes are not the same index and a disagreement about their names says nothing useful about why.
    normalization::check<ErrorCode::E_INCOMPATIBLE_INDEX>(
            res_common->has_multi_index() == other_common.has_multi_index(),
            "Cannot {} multi-indexed data with non-multi-indexed data",
            operation
    );
    if (res_common->has_multi_index()) {
        normalization::check<ErrorCode::E_INCOMPATIBLE_INDEX>(
                res_common->multi_index().field_count() == other_common.multi_index().field_count(),
                "Cannot {}: schemas have different index level counts, {} and {}",
                operation,
                res_common->multi_index().field_count() + 1,
                other_common.multi_index().field_count() + 1
        );
    } else {
        normalization::check<ErrorCode::E_INCOMPATIBLE_INDEX>(
                res_common->index().is_physically_stored() == other_common.index().is_physically_stored(),
                "Cannot {}: one index is physically stored and the other is not",
                operation
        );
    }

    if (res_common->has_multi_index()) {
        auto* res_index = res_common->mutable_multi_index();
        const auto& other_index = other_common.multi_index();
        if (other_index.name() != res_index->name() || other_index.is_int() != res_index->is_int()) {
            mismatches.add_index(0, true);
        }
        if (other_index.tz() != res_index->tz()) {
            res_index->clear_tz();
        }
        for (const auto& [idx, idx_timezone] : other_index.timezone()) {
            if ((*res_index->mutable_timezone())[idx] != idx_timezone) {
                (*res_index->mutable_timezone())[idx] = "";
            }
        }
        // A level both sides already record as unnamed agrees, and stays unnamed by virtue of already being in
        // the accumulated positions. A level only one side records as unnamed is a disagreement. So it is the
        // symmetric difference that matters, computed through sets rather than by walking the two repeated
        // fields in step, as nothing guarantees they are sorted.
        const std::set<uint32_t> res_unnamed{res_index->fake_field_pos().begin(), res_index->fake_field_pos().end()};
        const std::set<uint32_t> other_unnamed{
                other_index.fake_field_pos().begin(), other_index.fake_field_pos().end()
        };
        std::vector<uint32_t> disagreed;
        std::ranges::set_symmetric_difference(res_unnamed, other_unnamed, std::back_inserter(disagreed));
        for (const auto position : disagreed) {
            mismatches.add_index(position, true);
        }
    } else {
        auto* res_index = res_common->mutable_index();
        const auto& other_index = other_common.index();
        if (other_index.name() != res_index->name() || other_index.is_int() != res_index->is_int() ||
            other_index.fake_name() != res_index->fake_name()) {
            mismatches.add_index(0, false);
        }
        if (other_index.tz() != res_index->tz()) {
            res_index->clear_tz();
        }
        if (other_index.step() != res_index->step()) {
            log::version().warn("Mismatching RangeIndexes being combined, setting to start=0, step=1");
            res_index->set_start(0);
            res_index->set_step(1);
        }
    }
    // Last of the required fields, so that a disagreement about the index - which every schema has - is reported
    // ahead of one about the value column, which only a Series has.
    if (res.has_series() &&
        (res_common->has_name() != other_common.has_name() || res_common->name() != other_common.name())) {
        mismatches.add_series_name();
    }
    accumulate_norm_metadata_column_names(res, other);
    return res;
}

// Apply the recorded name disagreements to the normalization metadata. Doing it in one place, from the positions
// gathered while merging the descriptor fields and the metadata, is what keeps the two from drifting apart.
void apply_required_name_mismatches(
        NormalizationMetadata& norm, const RequiredFieldInfo& info, const RequiredNameMismatches& mismatches
) {
    if (!mismatches.any()) {
        return;
    }
    auto* common = mutable_pandas_common(norm);
    if (common == nullptr) {
        return;
    }
    if (info.has_multi_index && !mismatches.index_positions().empty()) {
        auto* multi_index = common->mutable_multi_index();
        // The accumulated positions are the levels every schema agrees are unnamed; the recorded ones are the
        // levels they disagree about. Both end up unnamed, so the output is the union.
        std::set<uint32_t> unnamed{multi_index->fake_field_pos().begin(), multi_index->fake_field_pos().end()};
        for (const auto position : mismatches.index_positions()) {
            unnamed.insert(static_cast<uint32_t>(position));
        }
        multi_index->clear_fake_field_pos();
        for (const auto position : unnamed) {
            multi_index->add_fake_field_pos(position);
        }
        if (unnamed.contains(0)) {
            multi_index->set_name("index");
        }
    } else if (mismatches.index_at(0)) {
        auto* index = common->mutable_index();
        index->set_name("index");
        index->set_is_int(false);
        index->set_fake_name(true);
    }
    if (mismatches.series_name()) {
        common->set_name("");
        common->set_has_name(false);
    }
}

// Fold the normalization metadata over every schema. Append and concat are treated alike: a disagreement over a
// timezone clears it rather than letting the newer schema win, and a disagreement over a RangeIndex's start or
// step resets it to start=0/step=1.
NormalizationMetadata combine_norm_metadata(
        std::span<const OutputSchema> schemas, RequiredNameMismatches& mismatches, const SchemaCombineOptions& options
) {
    auto result = schemas.front().norm_metadata_;
    for (const auto& schema : schemas.subspan(1)) {
        result = accumulate_norm_metadata(result, schema.norm_metadata_, mismatches, options);
    }
    return result;
}

// What is known about the sort order of every schema's data laid end to end.
SortedValue combine_sorted(std::span<const OutputSchema> schemas) {
    auto result = schemas.front().stream_descriptor().sorted();
    for (const auto& schema : schemas.subspan(1)) {
        result = deduce_sorted(result, schema.stream_descriptor().sorted());
    }
    return result;
}
} // namespace

SortedValue deduce_sorted(SortedValue existing_frame, SortedValue input_frame) {
    constexpr auto UNKNOWN = SortedValue::UNKNOWN;
    constexpr auto ASCENDING = SortedValue::ASCENDING;
    constexpr auto DESCENDING = SortedValue::DESCENDING;
    constexpr auto UNSORTED = SortedValue::UNSORTED;

    SortedValue final_state;
    switch (existing_frame) {
    case UNKNOWN:
        final_state = input_frame == UNSORTED ? UNSORTED : UNKNOWN;
        break;
    case ASCENDING:
        if (input_frame == UNKNOWN) {
            final_state = UNKNOWN;
        } else if (input_frame != ASCENDING) {
            final_state = UNSORTED;
        } else {
            final_state = ASCENDING;
        }
        break;
    case DESCENDING:
        if (input_frame == UNKNOWN) {
            final_state = UNKNOWN;
        } else if (input_frame != DESCENDING) {
            final_state = UNSORTED;
        } else {
            final_state = DESCENDING;
        }
        break;
    default:
        final_state = UNSORTED;
        break;
    }
    return final_state;
}

OutputSchema combine_schema(std::span<const OutputSchema> schemas, const SchemaCombineOptions& options) {
    // Combining nothing has no answer, so a caller whose zero-row filter removed every schema has to deal with
    // that itself rather than asking here - concat of two empty symbols must not reach this point.
    util::check(!schemas.empty(), "Cannot combine an empty list of schemas");
    // The normalization metadata goes first because it is what decides which shapes may combine at all, and so
    // what the output's required fields are. Merging the descriptors first would report a mismatch in whichever
    // column names happen to disagree as a consequence, rather than in the kind of object or index that caused it.
    RequiredNameMismatches mismatches{options};
    auto norm = combine_norm_metadata(schemas, mismatches, options);
    const auto info = required_fields_info(norm);

    // The stream id is deliberately left unset: which symbol a schema belongs to is not something the combine can
    // decide, and for a multi-symbol join there is no answer. tsd_from_schema supplies it.
    StreamDescriptor out{StreamId{}, combine_index_descriptors(schemas, options)};
    out.set_sorted(combine_sorted(schemas));
    add_required_fields(out, schemas, info, mismatches, options);
    add_data_columns(out, schemas, info, options);
    // Whatever only the normalization metadata reveals - a RangeIndex name, a Series name - is applied here, in
    // one place, so that the descriptor field names and the metadata cannot end up disagreeing.
    apply_required_name_mismatches(norm, info, mismatches);
    return OutputSchema{std::move(out), std::move(norm)};
}

OutputSchema schema_from_tsd(const TimeseriesDescriptor& tsd) {
    return {tsd.as_stream_descriptor(), tsd.normalization()};
}

OutputSchema schema_from_input_frame(const pipelines::InputFrame& frame) {
    // compute_desc_for_tsd rather than desc() because the result describes what will be stored, and an Arrow string
    // column holds 32 bit offsets in the frame but always 64 bit on disk.
    return {frame.compute_desc_for_tsd(), frame.norm_meta};
}

TimeseriesDescriptor tsd_from_schema(OutputSchema&& schema, size_t total_rows, pipelines::InputFrame& frame) {
    auto [descriptor, norm_meta, _] = schema.release();
    descriptor.set_id(frame.desc().id());
    return make_timeseries_descriptor(
            total_rows,
            std::move(descriptor),
            std::move(norm_meta),
            std::move(frame.user_meta),
            std::nullopt,
            frame.bucketize_dynamic
    );
}

} // namespace arcticdb
