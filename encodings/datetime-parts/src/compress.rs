use vortex_array::array::{PrimitiveArray, TemporalArray};
use vortex_array::compute::unary::try_cast;
use vortex_array::{ArrayDType as _, ArrayData, ArrayLen, IntoArrayData, IntoArrayVariant};
use vortex_datetime_dtype::TimeUnit;
use vortex_dtype::{DType, PType};
use vortex_error::{vortex_bail, VortexResult};

pub struct TemporalParts {
    pub days: ArrayData,
    pub seconds: ArrayData,
    pub subseconds: ArrayData,
}

/// Compress a `TemporalArray` into day, second, and subsecond components.
///
/// Splitting the components by granularity creates more small values, which enables better
/// cascading compression.
pub fn split_temporal(array: TemporalArray) -> VortexResult<TemporalParts> {
    let temporal_values = array.temporal_values().into_primitive()?;
    let validity = temporal_values.validity();

    // After this operation, timestamps will be non-nullable PrimitiveArray<i64>
    let timestamps = try_cast(
        &temporal_values,
        &DType::Primitive(PType::I64, temporal_values.dtype().nullability()),
    )?
    .into_primitive()?;

    let divisor = match array.temporal_metadata().time_unit() {
        TimeUnit::Ns => 1_000_000_000,
        TimeUnit::Us => 1_000_000,
        TimeUnit::Ms => 1_000,
        TimeUnit::S => 1,
        TimeUnit::D => vortex_bail!(InvalidArgument: "Cannot compress day-level data"),
    };

    let length = timestamps.len();
    let mut days = Vec::with_capacity(length);
    let mut seconds = Vec::with_capacity(length);
    let mut subsecond = Vec::with_capacity(length);

    for &t in timestamps.maybe_null_slice::<i64>().iter() {
        days.push(t / (86_400 * divisor));
        seconds.push((t % (86_400 * divisor)) / divisor);
        subsecond.push((t % (86_400 * divisor)) % divisor);
    }

    Ok(TemporalParts {
        days: PrimitiveArray::from_vec(days, validity).into_array(),
        seconds: PrimitiveArray::from(seconds).into_array(),
        subseconds: PrimitiveArray::from(subsecond).into_array(),
    })
}
