/* Copyright 2026 Man Group Operations Limited
 *
 * Use of this software is governed by the Business Source License 1.1 included in the file licenses/BSL.txt.
 *
 * As of the Change Date specified in that file, in accordance with the Business Source License, use of this software
 * will be governed by the Apache License, version 2.0.
 */

#pragma once

#include <arcticdb/entity/stream_descriptor.hpp>
#include <arcticdb/entity/timeseries_descriptor.hpp>

#include <span>
#include <string_view>

namespace arcticdb {

namespace pipelines {
struct InputFrame;
}

// How a column present in some schemas but missing from others is treated.
//   Strict - every schema must carry the same non-index columns in the same order (static append/update).
//   Drop  - keep only the columns present in all of them (concat inner join).
//   Keep  - keep the union of columns (dynamic append/update, concat outer join).
enum class MissingColumnPolicy { Strict, Drop, Keep };

// How the type of a column present in more than one schema is reconciled.
//   Static         - Static schema: promote empty->concrete and fixed->dynamic string
//   Dynamic        - Dynamic schema: promote via (has_valid_common_type)
//   MostPermissive - as Dynamic, but promotes to float64 when no exact common type exists
//                    for integral types. E.g. int32 + uint64 -> float64. Used for concat.
enum class TypePromotionPolicy { Static, Dynamic, MostPermissive };

// How a mismatch in the names of the required (index / Series) fields is treated.
//   Raise              - the required field names must match (append/update).
//   ReconcileToUnnamed - reconcile mismatched names to unnamed (concat).
enum class RequiredNameMismatchPolicy { Raise, ReconcileToUnnamed };

enum NormalizationOperation : uint8_t {
    APPEND,
    UPDATE,
    CONCAT,
};

// The operation named as the user would recognise it, so that a failed append does not report itself as a failed join.
constexpr std::string_view operation_name(NormalizationOperation operation) {
    switch (operation) {
    case APPEND:
        return "append";
    case UPDATE:
        return "update";
    case CONCAT:
        return "concat";
    }
    return "combine";
}

struct SchemaCombineOptions {
    MissingColumnPolicy missing_column;
    TypePromotionPolicy type_promotion;
    RequiredNameMismatchPolicy name_mismatch;
    NormalizationOperation operation;

    [[nodiscard]] std::string_view name() const { return operation_name(operation); }
};

// Append and update share their column mechanics exactly; they differ only in the index guards, which live in
// the callers. Two names rather than one so that the operation reported on failure is the one the user asked for.
inline SchemaCombineOptions append_or_update_options(bool dynamic_schema, NormalizationOperation operation) {
    const auto missing_column = dynamic_schema ? MissingColumnPolicy::Keep : MissingColumnPolicy::Strict;
    const auto type_promotion = dynamic_schema ? TypePromotionPolicy::Dynamic : TypePromotionPolicy::Static;
    return {missing_column, type_promotion, RequiredNameMismatchPolicy::Raise, operation};
}

inline SchemaCombineOptions append_options(bool dynamic_schema) {
    return append_or_update_options(dynamic_schema, APPEND);
}

inline SchemaCombineOptions update_options(bool dynamic_schema) {
    return append_or_update_options(dynamic_schema, UPDATE);
}

// Multi-symbol join utilities
enum class JoinType : uint8_t { OUTER, INNER };

inline SchemaCombineOptions concat_options(JoinType join_type) {
    const auto missing_column = join_type == JoinType::OUTER ? MissingColumnPolicy::Keep : MissingColumnPolicy::Drop;
    return {missing_column,
            TypePromotionPolicy::MostPermissive,
            RequiredNameMismatchPolicy::ReconcileToUnnamed,
            NormalizationOperation::CONCAT};
}

// Combine schemas into one. Resolves differences according to SchemaCombineOptions. The first schema is the
// base: its column order leads the output, and for append/update it is the existing symbol's schema.
// Takes all the schemas at once rather than folding pairwise because an inner join cannot decide whether two
// incompatible types matter until it knows which columns survive into the output.
entity::OutputSchema combine_schema(std::span<const entity::OutputSchema> schemas, const SchemaCombineOptions& options);

// What is known about the sort order of two frames laid end to end.
SortedValue deduce_sorted(SortedValue existing_frame, SortedValue input_frame);

// The two forms a schema is otherwise held in, as an OutputSchema, so that combine_schema is the only thing that
// has to know how to reconcile them.
entity::OutputSchema schema_from_tsd(const TimeseriesDescriptor& tsd);
entity::OutputSchema schema_from_input_frame(const pipelines::InputFrame& frame);

// Assemble the metadata for an index key from a combined schema plus the row count, which is the one thing a
// schema does not carry. The stream id comes from the frame, as combine_schema leaves the output's id unset.
TimeseriesDescriptor tsd_from_schema(entity::OutputSchema&& schema, size_t total_rows, pipelines::InputFrame& frame);

} // namespace arcticdb
