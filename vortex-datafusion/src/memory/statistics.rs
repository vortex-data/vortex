use datafusion_common::stats::Precision;
use datafusion_common::{ColumnStatistics, Result as DFResult, ScalarValue, Statistics};
use itertools::Itertools;
use vortex_array::arrays::ChunkedArray;
use vortex_array::stats::{Stat, Statistics as _};
use vortex_array::variants::StructArrayTrait;
use vortex_dtype::FieldNames;
use vortex_error::{VortexExpect, VortexResult};

use crate::converter::directional_bound_to_df_precision;

pub(crate) fn chunked_array_df_stats(
    array: &ChunkedArray,
    projection: FieldNames,
) -> DFResult<Statistics> {
    let mut nbytes: usize = 0;
    let column_statistics = projection
        .iter()
        .map(|name| array.maybe_null_field_by_name(name))
        .map_ok(|arr| {
            nbytes += arr.nbytes();
            ColumnStatistics {
                null_count: directional_bound_to_df_precision(
                    arr.statistics()
                        .get_as::<u64>(Stat::NullCount)
                        .map(|n| n.map(|n| n as usize)),
                ),
                max_value: directional_bound_to_df_precision(arr.get_stat(Stat::Max).map(|n| {
                    n.into_scalar(array.dtype().clone()).map(|n| {
                        ScalarValue::try_from(n).vortex_expect("cannot convert scalar to df scalar")
                    })
                })),
                min_value: directional_bound_to_df_precision(arr.get_stat(Stat::Min).map(|n| {
                    n.into_scalar(array.dtype().clone()).map(|n| {
                        ScalarValue::try_from(n).vortex_expect("cannot convert scalar to df scalar")
                    })
                })),
                distinct_count: Precision::Absent,
                sum_value: Precision::Absent,
            }
        })
        .collect::<VortexResult<Vec<_>>>()?;

    Ok(Statistics {
        num_rows: Precision::Exact(array.len()),
        total_byte_size: Precision::Exact(nbytes),
        column_statistics,
    })
}
