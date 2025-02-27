use vortex_error::VortexResult;

use crate::Array;
use crate::stats::{Precision, Stat};

pub fn null_count(array: &dyn Array) -> VortexResult<usize> {
    if let Some(Precision::Exact(invalid_count)) =
        array.statistics().get_as::<usize>(Stat::NullCount)
    {
        return Ok(invalid_count);
    }

    let null_count = array.validity_mask()?.false_count();
    assert!(
        null_count <= array.len(),
        "Invalid count exceeds array length"
    );

    array
        .statistics()
        .set(Stat::NullCount, Precision::exact(null_count));

    Ok(null_count)
}
