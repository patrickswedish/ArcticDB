#pragma once

#include <arcticdb/pipeline/input_frame.hpp>
#include <arcticdb/processing/schema_combine.hpp>
#include <arcticdb/python/normalization_utils.hpp>
#include <arcticdb/entity/timeseries_descriptor.hpp>

namespace arcticdb {

struct StreamDescriptorMismatch : ArcticSpecificException<ErrorCode::E_DESCRIPTOR_MISMATCH> {
    StreamDescriptorMismatch(
            const char* preamble, const StreamId& stream_id, const StreamDescriptor& existing,
            const StreamDescriptor& new_val, NormalizationOperation operation
    );
};

bool index_names_match(const StreamDescriptor& df_in_store_descriptor, const StreamDescriptor& new_df_descriptor);

bool columns_match(
        const StreamDescriptor& df_in_store_descriptor, const StreamDescriptor& new_df_descriptor,
        const bool convert_int_to_float = false
);

/// The checks and the schema merge for append and update, in one place: raises if the new frame cannot be combined
/// with the existing symbol, and otherwise returns the schema the combined data will have. Callers keep the
/// operational guards - pickled, sortedness, index contiguity - and the row counts, which this knows nothing about.
entity::OutputSchema combine_schema_with_frame(
        NormalizationOperation operation, bool dynamic_schema, const TimeseriesDescriptor& existing_tsd,
        const pipelines::InputFrame& new_frame
);
} // namespace arcticdb
