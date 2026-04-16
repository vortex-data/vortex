// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
#[expect(deprecated)]
use vortex_array::ToCanonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_buffer::BufferMut;
use vortex_error::VortexError;
use vortex_error::VortexResult;

use crate::DateTimePartsData;
use crate::timestamp;
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
    #[expect(deprecated)]
    let temporal_values = array.temporal_values().to_primitive();

    // After this operation, timestamps will be a PrimitiveArray<i64>
    #[expect(deprecated)]
    let timestamps = temporal_values
        .clone()
        .into_array()
        .cast(DType::Primitive(
            PType::I64,
            temporal_values.dtype().nullability(),
        ))?
        .to_primitive();

    let length = timestamps.len();
    let mut days = BufferMut::with_capacity(length);
    let mut seconds = BufferMut::with_capacity(length);
    let mut subseconds = BufferMut::with_capacity(length);

    for &ts in timestamps.as_slice::<i64>() {
        let ts_parts = timestamp::split(ts, array.temporal_metadata().time_unit())?;
        days.push(ts_parts.days);
        seconds.push(ts_parts.seconds);
        subseconds.push(ts_parts.subseconds);
    }

    Ok(TemporalParts {
        days: PrimitiveArray::new(days, temporal_values.validity()?).into_array(),
        seconds: seconds.into_array(),
        subseconds: subseconds.into_array(),
    })
}

impl TryFrom<TemporalArray> for DateTimePartsData {
    type Error = VortexError;

    fn try_from(array: TemporalArray) -> Result<Self, Self::Error> {
        let ext_dtype = array.ext_dtype();
        let TemporalParts {
            days,
            seconds,
            subseconds,
        } = split_temporal(array)?;
        DateTimePartsData::validate(
            &DType::Extension(ext_dtype),
            &days,
            &seconds,
            &subseconds,
            days.len(),
        )?;
        Ok(DateTimePartsData {})
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    #[expect(deprecated)]
    use vortex_array::ToCanonical;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::TemporalArray;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;

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
        #[expect(deprecated)]
        let days_prim = days.to_primitive();
        assert!(
            days_prim
                .validity()
                .vortex_expect("days validity should be derivable")
                .mask_eq(&validity, &mut ctx)
                .unwrap()
        );
        #[expect(deprecated)]
        let seconds_prim = seconds.to_primitive();
        assert!(matches!(
            seconds_prim
                .validity()
                .vortex_expect("seconds validity should be derivable"),
            Validity::NonNullable
        ));
        #[expect(deprecated)]
        let subseconds_prim = subseconds.to_primitive();
        assert!(matches!(
            subseconds_prim
                .validity()
                .vortex_expect("subseconds validity should be derivable"),
            Validity::NonNullable
        ));
    }
}
