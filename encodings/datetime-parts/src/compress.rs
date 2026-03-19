// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::vtable::ValidityHelper;
use vortex_buffer::BufferMut;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::DateTimePartsArray;
use crate::timestamp::SECONDS_PER_DAY;

pub struct TemporalParts {
    pub days: ArrayRef,
    pub seconds: ArrayRef,
    pub subseconds: ArrayRef,
}

/// Compress a `TemporalArray` into day, second, and subseconds components.
///
/// Splitting the components by granularity creates more small values, which enables better
/// cascading compression.
pub fn split_temporal(array: TemporalArray) -> VortexResult<TemporalParts> {
    let time_unit = array.temporal_metadata().time_unit();

    let divisor: i64 = match time_unit {
        TimeUnit::Nanoseconds => 1_000_000_000,
        TimeUnit::Microseconds => 1_000_000,
        TimeUnit::Milliseconds => 1_000,
        TimeUnit::Seconds => 1,
        TimeUnit::Days => vortex_bail!("Cannot handle day-level data"),
    };
    let ticks_per_day = SECONDS_PER_DAY * divisor;

    let temporal_values = array.temporal_values().to_primitive();

    // After this operation, timestamps will be a PrimitiveArray<i64>
    let timestamps = temporal_values
        .clone()
        .into_array()
        .cast(DType::Primitive(
            PType::I64,
            temporal_values.dtype().nullability(),
        ))?
        .to_primitive();

    let ts_slice = timestamps.as_slice::<i64>();
    let length = ts_slice.len();
    let mut days = BufferMut::with_capacity(length);
    let mut seconds = BufferMut::with_capacity(length);
    let mut subseconds = BufferMut::with_capacity(length);

    for &ts in ts_slice {
        days.push(ts / ticks_per_day);
        seconds.push((ts % ticks_per_day) / divisor);
        subseconds.push((ts % ticks_per_day) % divisor);
    }

    Ok(TemporalParts {
        days: PrimitiveArray::new(days, temporal_values.validity().clone()).into_array(),
        seconds: seconds.into_array(),
        subseconds: subseconds.into_array(),
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
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::ToCanonical;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::TemporalArray;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::validity::Validity;
    use vortex_array::vtable::ValidityHelper;
    use vortex_buffer::buffer;

    use crate::TemporalParts;
    use crate::split_temporal;

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
            TemporalArray::new_timestamp(milliseconds, TimeUnit::Milliseconds, Some("UTC".into()));
        let TemporalParts {
            days,
            seconds,
            subseconds,
        } = split_temporal(temporal_array).unwrap();

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert!(
            days.to_primitive()
                .validity()
                .mask_eq(&validity, &mut ctx)
                .unwrap()
        );
        assert!(matches!(
            seconds.to_primitive().validity(),
            Validity::NonNullable
        ));
        assert!(matches!(
            subseconds.to_primitive().validity(),
            Validity::NonNullable
        ));
    }
}
