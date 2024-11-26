use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use datafusion::functions_aggregate::min_max::{MaxAccumulator, MinAccumulator};
use datafusion_common::stats::Precision;
use datafusion_common::ColumnStatistics;
use datafusion_expr::Accumulator;
use vortex_array::array::StructArray;
use vortex_array::variants::StructArrayTrait as _;
use vortex_array::IntoCanonical;
use vortex_error::VortexResult;

pub fn array_to_col_statistics(array: StructArray) -> VortexResult<ColumnStatistics> {
    let mut stats = ColumnStatistics::new_unknown();

    if let Some(null_count_array) = array.field_by_name("null_count") {
        let array = null_count_array.into_canonical()?.into_arrow()?;
        let array = array.as_primitive::<UInt64Type>();

        let null_count = array.iter().map(|v| v.unwrap_or_default()).sum::<u64>();
        stats.null_count = Precision::Exact(null_count as usize);
    }

    if let Some(max_value_array) = array.field_by_name("max") {
        let array = max_value_array.into_canonical()?.into_arrow()?;
        let mut acc = MaxAccumulator::try_new(array.data_type())?;
        acc.update_batch(&[array])?;

        let max_val = acc.evaluate()?;
        stats.max_value = Precision::Exact(max_val)
    }

    if let Some(min_value_array) = array.field_by_name("min") {
        let array = min_value_array.into_canonical()?.into_arrow()?;
        let mut acc = MinAccumulator::try_new(array.data_type())?;
        acc.update_batch(&[array])?;

        let max_val = acc.evaluate()?;
        stats.min_value = Precision::Exact(max_val)
    }

    Ok(stats)
}
