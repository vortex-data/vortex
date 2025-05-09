use datafusion_common::stats::Precision;
use datafusion_common::{ColumnStatistics, Result as DFResult, ScalarValue, Statistics};
use itertools::Itertools;
use vortex_array::arrays::ChunkedArray;
use vortex_array::nbytes::NBytes;
use vortex_array::stats::{Stat, StatsProvider};
use vortex_array::{Array, ToCanonical};
use vortex_dtype::FieldNames;
use vortex_error::{VortexExpect, VortexResult};

use crate::PrecisionExt;

pub(crate) fn chunked_array_df_stats(
    array: &ChunkedArray,
    projection: FieldNames,
) -> DFResult<Statistics> {
    // Swizzle the chunked of struct in to struct of chunked.
    let array = array.to_struct()?;

    let mut nbytes: usize = 0;
    let column_statistics = projection
        .iter()
        .map(|name| array.field_by_name(name))
        .map_ok(|arr| {
            nbytes += arr.nbytes();
            ColumnStatistics {
                null_count: arr
                    .statistics()
                    .get_as::<u64>(Stat::NullCount)
                    .map(|n| n.map(|n| n as usize))
                    .to_df(),

                max_value: arr
                    .statistics()
                    .get(Stat::Max)
                    .map(|n| {
                        n.into_scalar(array.dtype().clone()).map(|n| {
                            ScalarValue::try_from(n)
                                .vortex_expect("cannot convert scalar to df scalar")
                        })
                    })
                    .to_df(),

                min_value: arr
                    .statistics()
                    .get(Stat::Min)
                    .map(|n| {
                        n.into_scalar(array.dtype().clone()).map(|n| {
                            ScalarValue::try_from(n)
                                .vortex_expect("cannot convert scalar to df scalar")
                        })
                    })
                    .to_df(),
                distinct_count: Precision::Absent,
                sum_value: arr
                    .statistics()
                    .get(Stat::Sum)
                    .map(|n| {
                        n.into_scalar(array.dtype().clone()).map(|n| {
                            ScalarValue::try_from(n)
                                .vortex_expect("cannot convert scalar to df scalar")
                        })
                    })
                    .to_df(),
            }
        })
        .collect::<VortexResult<Vec<_>>>()?;

    Ok(Statistics {
        num_rows: Precision::Exact(array.len()),
        total_byte_size: Precision::Exact(nbytes),
        column_statistics,
    })
}
