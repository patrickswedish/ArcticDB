/* Copyright 2026 Man Group Operations Limited
 *
 * Use of this software is governed by the Business Source License 1.1 included in the file licenses/BSL.txt.
 *
 * As of the Change Date specified in that file, in accordance with the Business Source License, use of this software
 * will be governed by the Apache License, version 2.0.
 */

#include <google/protobuf/util/message_differencer.h>

#include <gtest/gtest.h>
#include <arcticdb/processing/schema_combine.hpp>

using namespace arcticdb;
using namespace arcticdb::entity;
using namespace google::protobuf::util;
using NormalizationMetadata = arcticdb::proto::descriptors::NormalizationMetadata;

namespace {

// Build a timeseries-indexed DataFrame OutputSchema with the given index field and data columns.
OutputSchema timeseries_df(
        const std::string& index_name, const std::vector<std::pair<std::string, DataType>>& columns,
        const std::string& tz = ""
) {
    StreamDescriptor desc{StreamId{}, IndexDescriptorImpl{IndexDescriptor::Type::TIMESTAMP, 1}};
    desc.add_scalar_field(DataType::NANOSECONDS_UTC64, index_name);
    for (const auto& [name, type] : columns) {
        desc.add_scalar_field(type, name);
    }
    NormalizationMetadata norm;
    auto* index = norm.mutable_df()->mutable_common()->mutable_index();
    index->set_is_physically_stored(true);
    index->set_name(index_name);
    index->set_tz(tz);
    return {std::move(desc), std::move(norm)};
}

// A two-level multiindexed DataFrame: a timestamp level followed by one of the given type. The norm metadata
// field count is one less than the number of levels, as _normalization.py records it.
OutputSchema multiindex_df(
        DataType level_type, const std::vector<std::pair<std::string, DataType>>& columns,
        const std::string& level_0_name = "dt", const std::string& level_1_name = "lvl",
        const std::vector<uint32_t>& unnamed_levels = {}
) {
    StreamDescriptor desc{StreamId{}, IndexDescriptorImpl{IndexDescriptor::Type::TIMESTAMP, 2}};
    desc.add_scalar_field(DataType::NANOSECONDS_UTC64, level_0_name);
    desc.add_scalar_field(level_type, level_1_name);
    for (const auto& [name, type] : columns) {
        desc.add_scalar_field(type, name);
    }
    NormalizationMetadata norm;
    auto* multi_index = norm.mutable_df()->mutable_common()->mutable_multi_index();
    multi_index->set_field_count(1);
    multi_index->set_name(level_0_name);
    for (const auto position : unnamed_levels) {
        multi_index->add_fake_field_pos(position);
    }
    return {std::move(desc), std::move(norm)};
}

// A timeseries-indexed Series. Its value column counts as a required field, so it sits alongside the index
// levels rather than among the data columns.
OutputSchema timeseries_series(const std::string& index_name, const std::string& series_name, DataType value_type) {
    StreamDescriptor desc{StreamId{}, IndexDescriptorImpl{IndexDescriptor::Type::TIMESTAMP, 1}};
    desc.add_scalar_field(DataType::NANOSECONDS_UTC64, index_name);
    desc.add_scalar_field(value_type, series_name);
    NormalizationMetadata norm;
    auto* common = norm.mutable_series()->mutable_common();
    common->mutable_index()->set_is_physically_stored(true);
    common->mutable_index()->set_name(index_name);
    common->set_name(series_name);
    common->set_has_name(true);
    return {std::move(desc), std::move(norm)};
}

// A multiindexed DataFrame with one level per name. The norm metadata field count is one less than the number
// of levels, as _normalization.py records it; unnamed_levels are the positions it records as having no name.
OutputSchema multiindex_df_levels(
        const std::vector<std::string>& level_names, const std::vector<uint32_t>& unnamed_levels
) {
    StreamDescriptor desc{
            StreamId{}, IndexDescriptorImpl{IndexDescriptor::Type::TIMESTAMP, static_cast<uint32_t>(level_names.size())}
    };
    desc.add_scalar_field(DataType::NANOSECONDS_UTC64, level_names.front());
    for (size_t idx = 1; idx < level_names.size(); ++idx) {
        desc.add_scalar_field(DataType::INT32, level_names[idx]);
    }
    desc.add_scalar_field(DataType::FLOAT64, "a");
    NormalizationMetadata norm;
    auto* multi_index = norm.mutable_df()->mutable_common()->mutable_multi_index();
    multi_index->set_field_count(static_cast<uint32_t>(level_names.size() - 1));
    multi_index->set_name(level_names.front());
    for (const auto position : unnamed_levels) {
        multi_index->add_fake_field_pos(position);
    }
    return {std::move(desc), std::move(norm)};
}

// A RangeIndexed Series. The row count index is not a stored field, so the only required field is the value
// column, and the index's name lives solely in the normalization metadata.
OutputSchema rowcount_series(const std::string& index_name, const std::string& series_name, DataType value_type) {
    StreamDescriptor desc{StreamId{}, IndexDescriptorImpl{IndexDescriptor::Type::ROWCOUNT, 0}};
    desc.add_scalar_field(value_type, series_name);
    NormalizationMetadata norm;
    auto* common = norm.mutable_series()->mutable_common();
    common->mutable_index()->set_is_physically_stored(false);
    common->mutable_index()->set_name(index_name);
    // A real RangeIndex always has a non-zero step; a step of zero is how an empty index is recognised.
    common->mutable_index()->set_step(1);
    common->set_name(series_name);
    common->set_has_name(true);
    return {std::move(desc), std::move(norm)};
}

// Positions of the multiindex levels the output treats as unnamed. Sorted, as they are accumulated in an
// unordered set.
std::vector<uint32_t> fake_field_pos_of(const OutputSchema& schema) {
    const auto& common = schema.norm_metadata_.has_series() ? schema.norm_metadata_.series().common()
                                                            : schema.norm_metadata_.df().common();
    std::vector<uint32_t> positions{
            common.multi_index().fake_field_pos().begin(), common.multi_index().fake_field_pos().end()
    };
    std::ranges::sort(positions);
    return positions;
}

// What a zero-row pandas write produces when the library has empty types enabled: an empty index descriptor
// and no index field at all.
OutputSchema empty_index_df(const std::vector<std::pair<std::string, DataType>>& columns) {
    StreamDescriptor desc{StreamId{}, IndexDescriptorImpl{IndexDescriptor::Type::EMPTY, 0}};
    for (const auto& [name, type] : columns) {
        desc.add_scalar_field(type, name);
    }
    NormalizationMetadata norm;
    norm.mutable_df()->mutable_common()->mutable_index()->set_is_physically_stored(false);
    return {std::move(desc), std::move(norm)};
}

// A RangeIndexed DataFrame. The row count index is not stored as a field, so its name exists only here.
OutputSchema rowcount_df(
        const std::vector<std::pair<std::string, DataType>>& columns, const std::string& index_name = ""
) {
    StreamDescriptor desc{StreamId{}, IndexDescriptorImpl{IndexDescriptor::Type::ROWCOUNT, 0}};
    for (const auto& [name, type] : columns) {
        desc.add_scalar_field(type, name);
    }
    NormalizationMetadata norm;
    auto* index = norm.mutable_df()->mutable_common()->mutable_index();
    index->set_is_physically_stored(false);
    index->set_name(index_name);
    // A real RangeIndex always has a non-zero step; a step of zero is how an empty index is recognised.
    index->set_step(1);
    return {std::move(desc), std::move(norm)};
}

OutputSchema combine(std::vector<OutputSchema> schemas, const SchemaCombineOptions& options) {
    return combine_schema(schemas, options);
}

std::vector<std::pair<std::string, DataType>> columns_of(const OutputSchema& schema) {
    std::vector<std::pair<std::string, DataType>> out;
    for (const auto& field : schema.stream_descriptor().fields()) {
        out.emplace_back(std::string(field.name()), field.type().data_type());
    }
    return out;
}

} // namespace

TEST(CombineSchema, ConcatOuterUnionOfColumns) {
    auto base = timeseries_df("ts", {{"a", DataType::FLOAT64}, {"b", DataType::FLOAT64}});
    auto other = timeseries_df("ts", {{"b", DataType::FLOAT64}, {"c", DataType::FLOAT64}});
    auto combined = combine({base, other}, concat_options(JoinType::OUTER));
    // Union in base-first, then other-unique order.
    std::vector<std::pair<std::string, DataType>> expected{
            {"ts", DataType::NANOSECONDS_UTC64},
            {"a", DataType::FLOAT64},
            {"b", DataType::FLOAT64},
            {"c", DataType::FLOAT64}
    };
    ASSERT_EQ(columns_of(combined), expected);
}

TEST(CombineSchema, ConcatInnerIntersectionOfColumns) {
    auto base = timeseries_df("ts", {{"a", DataType::FLOAT64}, {"b", DataType::FLOAT64}});
    auto other = timeseries_df("ts", {{"b", DataType::FLOAT64}, {"c", DataType::FLOAT64}});
    auto combined = combine({base, other}, concat_options(JoinType::INNER));
    std::vector<std::pair<std::string, DataType>> expected{
            {"ts", DataType::NANOSECONDS_UTC64}, {"b", DataType::FLOAT64}
    };
    ASSERT_EQ(columns_of(combined), expected);
}

TEST(CombineSchema, ConcatOuterTypePromotion) {
    auto base = timeseries_df("ts", {{"a", DataType::INT32}});
    auto other = timeseries_df("ts", {{"a", DataType::INT64}});
    auto combined = combine({base, other}, concat_options(JoinType::OUTER));
    ASSERT_EQ(combined.stream_descriptor().field(1).type().data_type(), DataType::INT64);
}

TEST(CombineSchema, ConcatMismatchedIndexNameReconciledToFake) {
    auto base = timeseries_df("ts1", {{"a", DataType::FLOAT64}});
    auto other = timeseries_df("ts2", {{"a", DataType::FLOAT64}});
    auto combined = combine({base, other}, concat_options(JoinType::OUTER));
    // Scalar index name mismatch reconciles the index field to "index".
    ASSERT_EQ(combined.stream_descriptor().field(0).name(), "index");
}

// Renaming a subset of the multiindex levels reconciles only those levels, and records them as unnamed so
// that the read side reproduces an unnamed level rather than inventing a name. Ported from
// test_join_schemas.cpp::AddIndexFieldsTest::MultiIndexNonMatchingNames, which covers add_index_fields.
// Python equivalent: test_symbol_concatenation.py::test_symbol_concat_differently_named_multiindexes.
TEST(CombineSchema, ConcatRenamedMultiIndexLevelsReconciledToFake) {
    const std::vector<std::pair<std::string, DataType>> columns{{"a", DataType::FLOAT64}};
    const auto both_named = multiindex_df(DataType::INT32, columns);
    const auto first_renamed = multiindex_df(DataType::INT32, columns, "ts", "lvl");
    const auto second_renamed = multiindex_df(DataType::INT32, columns, "dt", "level2");
    const auto both_renamed = multiindex_df(DataType::INT32, columns, "ts", "level2");

    // Result must not depend on the ordering of the inputs, so check each pair both ways round.
    const auto combine_both_ways = [&](const OutputSchema& lhs, const OutputSchema& rhs, auto&& assertions) {
        assertions(combine({lhs, rhs}, concat_options(JoinType::OUTER)));
        assertions(combine({rhs, lhs}, concat_options(JoinType::OUTER)));
    };

    // Only level 0 differs: it takes the name "index", level 1 keeps its own.
    combine_both_ways(both_named, first_renamed, [](const OutputSchema& combined) {
        ASSERT_EQ(combined.stream_descriptor().field(0).name(), "index");
        ASSERT_EQ(combined.stream_descriptor().field(1).name(), "lvl");
        ASSERT_EQ(fake_field_pos_of(combined), std::vector<uint32_t>{0});
        ASSERT_EQ(combined.norm_metadata_.df().common().multi_index().name(), "index");
    });

    // Only level 1 differs: level 0 keeps its name and level 1 takes the __fkidx__ scheme.
    combine_both_ways(both_named, second_renamed, [](const OutputSchema& combined) {
        ASSERT_EQ(combined.stream_descriptor().field(0).name(), "dt");
        ASSERT_EQ(combined.stream_descriptor().field(1).name(), "__fkidx__1");
        ASSERT_EQ(fake_field_pos_of(combined), std::vector<uint32_t>{1});
    });

    // Both differ.
    combine_both_ways(both_named, both_renamed, [](const OutputSchema& combined) {
        ASSERT_EQ(combined.stream_descriptor().field(0).name(), "index");
        ASSERT_EQ(combined.stream_descriptor().field(1).name(), "__fkidx__1");
        ASSERT_EQ(fake_field_pos_of(combined), (std::vector<uint32_t>{0, 1}));
    });

    // Identical names are left alone.
    combine_both_ways(both_named, both_named, [](const OutputSchema& combined) {
        ASSERT_EQ(combined.stream_descriptor().field(0).name(), "dt");
        ASSERT_EQ(combined.stream_descriptor().field(1).name(), "lvl");
        ASSERT_TRUE(fake_field_pos_of(combined).empty());
    });
}

// A Series' value column is a required field, so a differing series name is reconciled the same way as a
// mismatched index level. The name is dropped from the normalization metadata, which is where the read side
// takes it from, so the descriptor's placeholder name is not user visible.
// Python equivalent: test_symbol_concatenation.py::test_symbol_concat_with_series.
TEST(CombineSchema, ConcatRenamedSeriesValueColumnDropsTheName) {
    const auto series_a = timeseries_series("ts", "a", DataType::FLOAT64);
    const auto series_b = timeseries_series("ts", "b", DataType::FLOAT64);

    for (auto schemas :
         {std::vector<OutputSchema>{series_a, series_b}, std::vector<OutputSchema>{series_b, series_a}}) {
        auto combined = combine(schemas, concat_options(JoinType::OUTER));
        ASSERT_EQ(combined.stream_descriptor().field(0).name(), "ts");
        ASSERT_EQ(combined.stream_descriptor().field(1).name(), "__fkidx__1");
        ASSERT_FALSE(combined.norm_metadata_.series().common().has_name());
    }

    // Matching names are preserved.
    auto combined = combine({series_a, series_a}, concat_options(JoinType::OUTER));
    ASSERT_EQ(combined.stream_descriptor().field(1).name(), "a");
    ASSERT_TRUE(combined.norm_metadata_.series().common().has_name());
    ASSERT_EQ(combined.norm_metadata_.series().common().name(), "a");
}

// A level every schema already records as unnamed agrees, so it contributes no mismatch - and it must still
// come out unnamed. _normalization.py:874 reads fake_field_pos to decide a level has no name, so losing the
// position would make the level read back literally named "__fkidx__1".
TEST(CombineSchema, AlreadyUnnamedMultiIndexLevelsStayUnnamed) {
    const std::vector<std::pair<std::string, DataType>> columns{{"a", DataType::FLOAT64}};
    const auto unnamed_level_1 = multiindex_df(DataType::INT32, columns, "dt", "__fkidx__1", {1});

    // Nothing disagrees at all, so there are no mismatching positions to drive the reconcile.
    auto combined = combine({unnamed_level_1, unnamed_level_1}, concat_options(JoinType::OUTER));
    ASSERT_EQ(fake_field_pos_of(combined), std::vector<uint32_t>{1});
    ASSERT_EQ(combined.stream_descriptor().field(1).name(), "__fkidx__1");

    // Level 0 disagrees, so there is a mismatch, and level 1 must not be lost while it is applied.
    const auto renamed_level_0 = multiindex_df(DataType::INT32, columns, "ts", "__fkidx__1", {1});
    combined = combine({unnamed_level_1, renamed_level_0}, concat_options(JoinType::OUTER));
    ASSERT_EQ(fake_field_pos_of(combined), (std::vector<uint32_t>{0, 1}));

    // And appending them is fine, since nothing about the names disagrees.
    combined = combine({unnamed_level_1, unnamed_level_1}, append_options(true));
    ASSERT_EQ(fake_field_pos_of(combined), std::vector<uint32_t>{1});
}

// One level that every schema agrees is unnamed, alongside one they disagree about. Applying the disagreement
// must not drop the level that agreed, which contributes no mismatching position of its own.
TEST(CombineSchema, AgreedUnnamedLevelSurvivesAlongsideADisagreeingOne) {
    // Level 1 is unnamed in both. Level 2 is unnamed in the first only, so only it disagrees.
    const auto both = multiindex_df_levels({"dt", "__fkidx__1", "__fkidx__2"}, {1, 2});
    const auto only_level_1 = multiindex_df_levels({"dt", "__fkidx__1", "lvl2"}, {1});

    auto combined = combine({both, only_level_1}, concat_options(JoinType::OUTER));
    ASSERT_EQ(fake_field_pos_of(combined), (std::vector<uint32_t>{1, 2}));

    // Independent of the ordering of the inputs.
    combined = combine({only_level_1, both}, concat_options(JoinType::OUTER));
    ASSERT_EQ(fake_field_pos_of(combined), (std::vector<uint32_t>{1, 2}));
}

// For a RangeIndexed Series the only required field is the value column, so a series-name disagreement is
// recorded at position 0 - the same position a scalar index would occupy. Reconciling it must not be mistaken
// for an index-name disagreement and overwrite the index metadata, which both schemas agree on.
TEST(CombineSchema, RowCountSeriesNameMismatchLeavesTheIndexAlone) {
    const auto series_a = rowcount_series("idx", "a", DataType::FLOAT64);
    const auto series_b = rowcount_series("idx", "b", DataType::FLOAT64);

    auto combined = combine({series_a, series_b}, concat_options(JoinType::OUTER));
    const auto& common = combined.norm_metadata_.series().common();
    // The series name disagreed, so it is dropped.
    ASSERT_FALSE(common.has_name());
    // The index name did not disagree, so it must survive untouched.
    ASSERT_EQ(common.index().name(), "idx");
    ASSERT_FALSE(common.index().fake_name());
}

// The required fields have to describe the same shape of index in every schema, so these are rejected outright
// rather than merged. The shape of the index and the kind of object are normalization concerns, not schema ones,
// so they raise as such. Each disagreement gets its own message, and the message names the operation.
TEST(CombineSchema, IncompatibleRequiredFieldShapesRaise) {
    const std::vector<std::pair<std::string, DataType>> columns{{"a", DataType::FLOAT64}};
    const auto two_levels = multiindex_df_levels({"dt", "lvl"}, {});
    const auto three_levels = multiindex_df_levels({"dt", "lvl", "lvl2"}, {});
    const auto scalar_index = timeseries_df("dt", columns);
    const auto series = timeseries_series("dt", "v", DataType::FLOAT64);

    for (const auto& options : {concat_options(JoinType::OUTER), append_options(true)}) {
        // Differing multi-index level counts.
        ASSERT_THROW(combine({two_levels, three_levels}, options), NormalizationException);
        ASSERT_THROW(combine({three_levels, two_levels}, options), NormalizationException);
        // Multi-indexed against not.
        ASSERT_THROW(combine({two_levels, scalar_index}, options), NormalizationException);
        ASSERT_THROW(combine({scalar_index, two_levels}, options), NormalizationException);
        // A Series against a DataFrame.
        ASSERT_THROW(combine({scalar_index, series}, options), NormalizationException);
        ASSERT_THROW(combine({series, scalar_index}, options), NormalizationException);
        // A multi-indexed DataFrame against a scalar-indexed Series, which differs in both respects.
        ASSERT_THROW(combine({two_levels, series}, options), NormalizationException);
    }
}

// Arrow normalization metadata records neither the Series/DataFrame distinction nor multi-index levels, so it
// cannot disagree about the shape: its leading fields line up with the base schema's required fields.
TEST(CombineSchema, ArrowSchemaHasNoShapeToDisagreeAbout) {
    const auto series = timeseries_series("ts", "col", DataType::INT64);
    auto arrow = timeseries_df("ts", {{"col", DataType::INT64}});
    arrow.norm_metadata_.mutable_experimental_arrow()->set_has_index(true);

    for (auto schemas : {std::vector<OutputSchema>{series, arrow}, std::vector<OutputSchema>{arrow, series}}) {
        auto combined = combine(schemas, concat_options(JoinType::OUTER));
        ASSERT_EQ(
                columns_of(combined),
                (std::vector<std::pair<std::string, DataType>>{
                        {"ts", DataType::NANOSECONDS_UTC64}, {"col", DataType::INT64}
                })
        );
    }
}

// The operation the caller asked for is what the error should name, so that a failed append does not report
// itself as a failed join.
TEST(CombineSchema, ErrorMessagesNameTheOperation) {
    const auto base = timeseries_df("ts", {{"a", DataType::UTF_DYNAMIC64}});
    const auto other = timeseries_df("ts", {{"a", DataType::INT64}});
    const auto message_for = [&](const SchemaCombineOptions& options) {
        try {
            combine({base, other}, options);
        } catch (const SchemaException& exception) {
            return std::string{exception.what()};
        }
        return std::string{};
    };
    ASSERT_NE(message_for(append_options(true)).find("append"), std::string::npos);
    ASSERT_NE(message_for(update_options(true)).find("update"), std::string::npos);
    ASSERT_NE(message_for(concat_options(JoinType::OUTER)).find("concat"), std::string::npos);
}

TEST(CombineSchema, AppendRejectsRenamedRequiredFields) {
    const std::vector<std::pair<std::string, DataType>> columns{{"a", DataType::FLOAT64}};
    const auto options = append_options(true);
    // Multiindex level, either position. Reported as an index incompatibility rather than a descriptor mismatch:
    // the level names are what keeps the normalization metadata in step with the data.
    ASSERT_THROW(
            combine({multiindex_df(DataType::INT32, columns), multiindex_df(DataType::INT32, columns, "ts", "lvl")},
                    options),
            NormalizationException
    );
    ASSERT_THROW(
            combine({multiindex_df(DataType::INT32, columns), multiindex_df(DataType::INT32, columns, "dt", "level2")},
                    options),
            NormalizationException
    );
    // Series value column. Matches the message asserted by
    // test_append.py::test_append_series_with_different_column_name_throws.
    ASSERT_THROW(
            combine({timeseries_series("ts", "a", DataType::FLOAT64), timeseries_series("ts", "b", DataType::FLOAT64)},
                    options),
            SchemaException
    );
}

TEST(CombineSchema, AppendStaticSameColumnsSucceeds) {
    auto base = timeseries_df("ts", {{"a", DataType::FLOAT64}, {"b", DataType::INT64}});
    auto other = timeseries_df("ts", {{"a", DataType::FLOAT64}, {"b", DataType::INT64}});
    auto combined = combine({base, other}, append_options(false));
    ASSERT_EQ(columns_of(combined), columns_of(base));
}

TEST(CombineSchema, AppendStaticMissingColumnRaises) {
    auto base = timeseries_df("ts", {{"a", DataType::FLOAT64}, {"b", DataType::INT64}});
    auto other = timeseries_df("ts", {{"a", DataType::FLOAT64}});
    ASSERT_THROW(combine({base, other}, append_options(false)), SchemaException);
}

TEST(CombineSchema, AppendStaticEmptyToConcretePromotion) {
    auto base = timeseries_df("ts", {{"a", DataType::EMPTYVAL}});
    auto other = timeseries_df("ts", {{"a", DataType::FLOAT64}});
    auto combined = combine({base, other}, append_options(false));
    ASSERT_EQ(combined.stream_descriptor().field(1).type().data_type(), DataType::FLOAT64);
}

TEST(CombineSchema, AppendStaticFixedToDynamicStringPromotion) {
    auto base = timeseries_df("ts", {{"a", DataType::UTF_FIXED64}});
    auto other = timeseries_df("ts", {{"a", DataType::UTF_DYNAMIC64}});
    auto combined = combine({base, other}, append_options(false));
    ASSERT_EQ(combined.stream_descriptor().field(1).type().data_type(), DataType::UTF_DYNAMIC64);
}

TEST(CombineSchema, AppendMismatchedIndexNameRaises) {
    auto base = timeseries_df("ts1", {{"a", DataType::FLOAT64}});
    auto other = timeseries_df("ts2", {{"a", DataType::FLOAT64}});
    ASSERT_THROW(combine({base, other}, append_options(false)), SchemaException);
}

TEST(CombineSchema, AppendDynamicKeepsUnionAndPromotes) {
    auto base = timeseries_df("ts", {{"a", DataType::INT32}});
    auto other = timeseries_df("ts", {{"a", DataType::INT64}, {"b", DataType::FLOAT64}});
    auto combined = combine({base, other}, append_options(true));
    std::vector<std::pair<std::string, DataType>> expected{
            {"ts", DataType::NANOSECONDS_UTC64}, {"a", DataType::INT64}, {"b", DataType::FLOAT64}
    };
    ASSERT_EQ(columns_of(combined), expected);
}

// Mixed-signedness 64-bit ints have no type representing both exactly. Concat falls back to float64 for data
// columns; append refuses; and required fields refuse under both because index levels and the Series value
// column must keep their values exact.
TEST(CombineSchema, ConcatPromotesMixedSignednessDataColumnToFloat64) {
    auto base = timeseries_df("ts", {{"a", DataType::UINT64}});
    auto other = timeseries_df("ts", {{"a", DataType::INT64}});
    for (auto join_type : {JoinType::OUTER, JoinType::INNER}) {
        auto combined = combine({base, other}, concat_options(join_type));
        ASSERT_EQ(combined.stream_descriptor().field(1).type().data_type(), DataType::FLOAT64);
    }
}

TEST(CombineSchema, AppendRejectsMixedSignednessDataColumn) {
    auto base = timeseries_df("ts", {{"a", DataType::UINT64}});
    auto other = timeseries_df("ts", {{"a", DataType::INT64}});
    ASSERT_THROW(combine({base, other}, append_options(true)), SchemaException);
}

TEST(CombineSchema, RequiredFieldsNeverTakeTheFloat64Fallback) {
    auto base = multiindex_df(DataType::UINT64, {{"a", DataType::FLOAT64}});
    auto other = multiindex_df(DataType::INT64, {{"a", DataType::FLOAT64}});
    ASSERT_THROW(combine({base, other}, concat_options(JoinType::OUTER)), SchemaException);
    ASSERT_THROW(combine({base, other}, append_options(true)), SchemaException);
    // A level pair that does have an exact common type still promotes.
    auto promotable = multiindex_df(DataType::INT32, {{"a", DataType::FLOAT64}});
    auto combined = combine({promotable, other}, concat_options(JoinType::OUTER));
    ASSERT_EQ(combined.stream_descriptor().field(1).type().data_type(), DataType::INT64);
}

// An empty index describes zero rows and says nothing about what the index would have been - it has no name, no
// timezone and occupies no descriptor field - so the concrete index it is combined with decides, in either order.
// The empty schema's fields are all data columns, as it has no index level to account for.
TEST(CombineSchema, EmptyIndexTakesOnTheConcreteIndexItIsCombinedWith) {
    auto empty = empty_index_df({{"a", DataType::EMPTYVAL}});
    auto timeseries = timeseries_df("ts", {{"a", DataType::FLOAT64}});
    for (const auto& options :
         {concat_options(JoinType::OUTER), concat_options(JoinType::INNER), append_options(true)}) {
        for (auto schemas :
             {std::vector<OutputSchema>{empty, timeseries}, std::vector<OutputSchema>{timeseries, empty}}) {
            auto combined = combine(schemas, options);
            ASSERT_EQ(combined.stream_descriptor().index().type(), IndexDescriptor::Type::TIMESTAMP);
            ASSERT_EQ(
                    columns_of(combined),
                    (std::vector<std::pair<std::string, DataType>>{
                            {"ts", DataType::NANOSECONDS_UTC64}, {"a", DataType::FLOAT64}
                    })
            );
            ASSERT_TRUE(combined.norm_metadata_.df().common().index().is_physically_stored());
        }
    }
}

// A column only the empty schema has is still a column: it takes part in the join like any other, and an outer
// join keeps it. Callers that want a zero-row frame to contribute nothing drop it before combining.
TEST(CombineSchema, EmptyIndexSchemaContributesItsDataColumns) {
    auto empty = empty_index_df({{"a", DataType::EMPTYVAL}, {"only_empty", DataType::EMPTYVAL}});
    auto timeseries = timeseries_df("ts", {{"a", DataType::FLOAT64}});

    auto outer = combine({timeseries, empty}, concat_options(JoinType::OUTER));
    ASSERT_EQ(
            columns_of(outer),
            (std::vector<std::pair<std::string, DataType>>{
                    {"ts", DataType::NANOSECONDS_UTC64}, {"a", DataType::FLOAT64}, {"only_empty", DataType::EMPTYVAL}
            })
    );

    auto inner = combine({timeseries, empty}, concat_options(JoinType::INNER));
    ASSERT_EQ(
            columns_of(inner),
            (std::vector<std::pair<std::string, DataType>>{
                    {"ts", DataType::NANOSECONDS_UTC64}, {"a", DataType::FLOAT64}
            })
    );
}

// Every schema being empty-indexed is the degenerate case where no symbol has any rows. There are no shapes to
// reconcile, so it combines rather than being rejected - concat of nothing but empty symbols still has to work.
TEST(CombineSchema, AllEmptyIndicesCombineToAnEmptyIndex) {
    auto empty_a = empty_index_df({{"a", DataType::EMPTYVAL}});
    auto empty_b = empty_index_df({{"b", DataType::EMPTYVAL}});

    auto combined = combine({empty_a, empty_b}, concat_options(JoinType::OUTER));
    ASSERT_EQ(combined.stream_descriptor().index().type(), IndexDescriptor::Type::EMPTY);
    ASSERT_EQ(
            columns_of(combined),
            (std::vector<std::pair<std::string, DataType>>{{"a", DataType::EMPTYVAL}, {"b", DataType::EMPTYVAL}})
    );

    ASSERT_TRUE(columns_of(combine({empty_a, empty_b}, concat_options(JoinType::INNER))).empty());
}

// A RangeIndex has no descriptor field, so its name lives only in the normalization metadata. Append used to
// let the last write silently win; it now raises, like every other required-field name (Monday 9797097831).
TEST(CombineSchema, RowCountIndexNameMismatchIsReconciledForConcatAndRaisesForAppend) {
    auto named = rowcount_df({{"a", DataType::FLOAT64}}, "index_name_1");
    auto renamed = rowcount_df({{"a", DataType::FLOAT64}}, "index_name_2");

    auto combined = combine({named, renamed}, concat_options(JoinType::OUTER));
    const auto& index = combined.norm_metadata_.df().common().index();
    ASSERT_EQ(index.name(), "index");
    ASSERT_TRUE(index.fake_name());

    ASSERT_THROW(combine({named, renamed}, append_options(true)), SchemaException);
    // Matching names are left alone.
    ASSERT_EQ(
            combine({named, named}, append_options(true)).norm_metadata_.df().common().index().name(), "index_name_1"
    );
}

// Append used to let the new frame's timezone overwrite the existing one; both operations now clear it, since
// neither timezone can be said to describe the combined data (Monday 12029540807).
TEST(CombineSchema, MismatchedTimezoneIsCleared) {
    auto london = timeseries_df("ts", {{"a", DataType::FLOAT64}}, "Europe/London");
    auto new_york = timeseries_df("ts", {{"a", DataType::FLOAT64}}, "America/New_York");
    for (const auto& options : {concat_options(JoinType::OUTER), append_options(true)}) {
        auto combined = combine({london, new_york}, options);
        ASSERT_EQ(combined.norm_metadata_.df().common().index().tz(), "");
    }
    // A shared timezone survives.
    auto combined = combine({london, london}, append_options(true));
    ASSERT_EQ(combined.norm_metadata_.df().common().index().tz(), "Europe/London");
}

TEST(CombineSchema, IncompatibleIndexTypesRaise) {
    auto timeseries = timeseries_df("ts", {{"a", DataType::FLOAT64}});
    auto rowcount = rowcount_df({{"a", DataType::FLOAT64}});
    ASSERT_THROW(combine({timeseries, rowcount}, concat_options(JoinType::OUTER)), NormalizationException);
    ASSERT_THROW(combine({rowcount, timeseries}, append_options(true)), NormalizationException);
}

// An inner join drops any column missing from any schema, so two schemas disagreeing irreconcilably about a
// column does not matter if a third schema lacks it. The clash must only be reported for surviving columns.
TEST(CombineSchema, InnerJoinIgnoresIncompatibleTypesOnDroppedColumns) {
    auto schema_0 = timeseries_df("ts", {{"common", DataType::FLOAT64}, {"a", DataType::UTF_DYNAMIC64}});
    auto schema_1 = timeseries_df("ts", {{"common", DataType::FLOAT64}, {"a", DataType::INT64}});
    auto schema_2 = timeseries_df("ts", {{"common", DataType::FLOAT64}});

    auto combined = combine({schema_0, schema_1, schema_2}, concat_options(JoinType::INNER));
    std::vector<std::pair<std::string, DataType>> expected{
            {"ts", DataType::NANOSECONDS_UTC64}, {"common", DataType::FLOAT64}
    };
    ASSERT_EQ(columns_of(combined), expected);

    // An outer join keeps "a", so there the clash does matter.
    ASSERT_THROW(combine({schema_0, schema_1, schema_2}, concat_options(JoinType::OUTER)), SchemaException);
}

TEST(CombineSchema, ThreeSchemasKeepFirstSeenColumnOrder) {
    auto schema_0 = timeseries_df("ts", {{"a", DataType::FLOAT64}});
    auto schema_1 = timeseries_df("ts", {{"c", DataType::FLOAT64}, {"b", DataType::FLOAT64}});
    auto schema_2 = timeseries_df("ts", {{"b", DataType::FLOAT64}, {"d", DataType::FLOAT64}});
    auto combined = combine({schema_0, schema_1, schema_2}, concat_options(JoinType::OUTER));
    std::vector<std::pair<std::string, DataType>> expected{
            {"ts", DataType::NANOSECONDS_UTC64},
            {"a", DataType::FLOAT64},
            {"c", DataType::FLOAT64},
            {"b", DataType::FLOAT64},
            {"d", DataType::FLOAT64}
    };
    ASSERT_EQ(columns_of(combined), expected);
}
