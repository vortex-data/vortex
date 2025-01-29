use vortex_array::array::{PrimitiveArray, TemporalArray};
use vortex_array::compute::try_cast;
use vortex_array::{ArrayData, IntoArrayData, IntoArrayVariant};
use vortex_buffer::BufferMut;
use vortex_datetime_dtype::TimeUnit;
use vortex_dtype::{DType, PType};
use vortex_error::{vortex_bail, VortexError, VortexResult};

use crate::DateTimePartsArray;

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
    let mut days = BufferMut::with_capacity(length);
    let mut seconds = BufferMut::with_capacity(length);
    let mut subsecond = BufferMut::with_capacity(length);

    for &t in timestamps.as_slice::<i64>().iter() {
        days.push(t / (86_400 * divisor));
        seconds.push((t % (86_400 * divisor)) / divisor);
        subsecond.push((t % (86_400 * divisor)) % divisor);
    }

    Ok(TemporalParts {
        days: PrimitiveArray::new(days, validity).into_array(),
        seconds: seconds.into_array(),
        subseconds: subsecond.into_array(),
    })
}

impl TryFrom<TemporalArray> for DateTimePartsArray {
    type Error = VortexError;

    fn try_from(array: TemporalArray) -> Result<Self, Self::Error> {
        let ext_dtype = array.ext_dtype();
        let TemporalParts {
            days,
            seconds,
            subseconds,
        } = split_temporal(array)?;
        DateTimePartsArray::try_new(DType::Extension(ext_dtype), days, seconds, subseconds)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::array::{PrimitiveArray, TemporalArray};
    use vortex_array::validity::Validity;
    use vortex_array::{IntoArrayData as _, IntoArrayVariant as _};
    use vortex_buffer::buffer;
    use vortex_datetime_dtype::TimeUnit;

    use crate::{split_temporal, TemporalParts};

    #[rstest]
    #[case(Validity::NonNullable)]
    #[case(Validity::AllValid)]
    #[case(Validity::AllInvalid)]
    #[case(Validity::from_iter([true, false, true]))]
    fn test_split_temporal(#[case] validity: Validity) {
        let milliseconds = PrimitiveArray::new(
            buffer![
                86_400i64,            // element with only day component
                86_400i64 + 1000,     // element with day + second components
                86_400i64 + 1000 + 1, // element with day + second + sub-second components
            ],
            validity.clone(),
        )
        .into_array();
        let temporal_array =
            TemporalArray::new_timestamp(milliseconds, TimeUnit::Ms, Some("UTC".to_string()));
        let TemporalParts {
            days,
            seconds,
            subseconds,
        } = split_temporal(temporal_array).unwrap();
        assert_eq!(days.into_primitive().unwrap().validity(), validity);
        assert_eq!(
            seconds.into_primitive().unwrap().validity(),
            Validity::NonNullable
        );
        assert_eq!(
            subseconds.into_primitive().unwrap().validity(),
            Validity::NonNullable
        );
    }
}
