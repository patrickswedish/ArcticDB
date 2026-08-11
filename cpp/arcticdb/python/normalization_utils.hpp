/* Copyright 2026 Man Group Operations Limited
 *
 * Use of this software is governed by the Business Source License 1.1 included in the file licenses/BSL.txt.
 *
 * As of the Change Date specified in that file, in accordance with the Business Source License, use of this software
 * will be governed by the Apache License, version 2.0.
 */

#pragma once

#include <vector>
#include <unordered_set>
#include <arcticdb/entity/descriptors.hpp>
#include <arcticdb/entity/timeseries_descriptor.hpp>

namespace arcticdb {

namespace entity {
struct OutputSchema;
}

namespace pipelines {
struct InputFrame;
} // namespace pipelines

/// In case both indexes are row-ranged sanity checks will be performed:
/// * Both indexes must have the same step
/// * The new index must start at the point where the old one ends
/// If the checks above pass update the new normalization index so that it spans the whole index (old + new).
/// A no-op for input types that have no pandas index, such as an ndarray or a pickled object.
/// @throws In case the row-ranged indexes are incompatible
void update_rowcount_normalization_data(
        const proto::descriptors::NormalizationMetadata& old_norm, proto::descriptors::NormalizationMetadata& new_norm,
        size_t old_length
);
} // namespace arcticdb
