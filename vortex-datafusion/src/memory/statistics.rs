use datafusion_common::stats::Precision;
use datafusion_common::{ColumnStatistics, Result as DFResult, ScalarValue, Statistics};
use itertools::Itertools;
use vortex_array::array::ChunkedArray;
use vortex_array::stats::Stat;
use vortex_array::variants::StructArrayTrait;
use vortex_dtype::FieldNames;
use vortex_error::{vortex_err, VortexExpect, VortexResult};
use vortex_scalar::Scalar;

pub(crate) fn chunked_array_df_stats(
    array: &ChunkedArray,
    projection: FieldNames,
) -> DFResult<Statistics> {
    let mut nbytes: usize = 0;
    let column_statistics = projection
        .iter()
        .map(|name| {
            array
                .maybe_null_field_by_name(name)
                .ok_or_else(|| vortex_err!("Projection references unknown field {name}"))
        })
        .map_ok(|arr| {
            nbytes += arr.nbytes();
            ColumnStatistics {
                null_count: arr
                    .statistics()
                    .get_as::<u64>(Stat::NullCount)
                    .map(|n| n as usize)
                    .map(Precision::Exact)
                    .unwrap_or(Precision::Absent),
                max_value: arr
                    .statistics()
                    .get(Stat::Max)
                    .map(|n| Scalar::new(array.dtype().clone(), n))
                    .map(|n| {
                        ScalarValue::try_from(n).vortex_expect("cannot convert scalar to df scalar")
                    })
                    .map(Precision::Exact)
                    .unwrap_or(Precision::Absent),
                min_value: arr
                    .statistics()
                    .get(Stat::Min)
                    .map(|n| Scalar::new(array.dtype().clone(), n))
                    .map(|n| {
                        ScalarValue::try_from(n).vortex_expect("cannot convert scalar to df scalar")
                    })
                    .map(Precision::Exact)
                    .unwrap_or(Precision::Absent),
                distinct_count: Precision::Absent,
            }
        })
        .collect::<VortexResult<Vec<_>>>()?;

    Ok(Statistics {
        num_rows: Precision::Exact(array.len()),
        total_byte_size: Precision::Exact(nbytes),
        column_statistics,
    })
}
