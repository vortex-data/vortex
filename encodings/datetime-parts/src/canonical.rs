// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::extension::datetime::Timestamp;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::DateTimePartsArray;
use crate::array::DateTimePartsArrayExt;

/// Decode an [Array] into a [TemporalArray].
///
/// Enforces that the passed array is actually a [DateTimePartsArray] with proper metadata.
pub fn decode_to_temporal(
    array: &DateTimePartsArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<TemporalArray> {
    let DType::Extension(ext) = array.dtype().clone() else {
        vortex_panic!(Compute: "expected dtype to be DType::Extension variant")
    };

    let Some(options) = ext.metadata_opt::<Timestamp>() else {
        vortex_panic!(Compute: "must decode TemporalMetadata from extension metadata");
    };

    let divisor = match options.unit {
        TimeUnit::Nanoseconds => 1_000_000_000,
        TimeUnit::Microseconds => 1_000_000,
        TimeUnit::Milliseconds => 1_000,
        TimeUnit::Seconds => 1,
        TimeUnit::Days => vortex_panic!(InvalidArgument: "cannot decode into TimeUnit::D"),
    };

    let days_buf = array
        .days()
        .cast(DType::Primitive(PType::I64, array.dtype().nullability()))
        .vortex_expect("must be able to cast days to i64")
        .execute::<PrimitiveArray>(ctx)?;

    // We start with the days component, which is always present.
    // And then add the seconds and subseconds components.
    // We split this into separate passes because often the seconds and/org subseconds components
    // are constant.
    let mut values: BufferMut<i64> = days_buf
        .into_buffer::<i64>()
        .map_each_in_place(|d| d * 86_400 * divisor);

    if let Some(seconds) = array.seconds().as_constant() {
        let seconds = seconds
            .as_primitive()
            .as_::<i64>()
            .vortex_expect("non-nullable");
        let seconds = seconds * divisor;
        for v in values.iter_mut() {
            *v += seconds;
        }
    } else {
        let seconds_buf = array.seconds().clone().execute::<PrimitiveArray>(ctx)?;
        match_each_integer_ptype!(seconds_buf.ptype(), |S| {
            for (v, second) in values.iter_mut().zip(seconds_buf.as_slice::<S>()) {
                let second: i64 = second.as_();
                *v += second * divisor;
            }
        });
    }

    if let Some(subseconds) = array.subseconds().as_constant() {
        let subseconds = subseconds
            .as_primitive()
            .as_::<i64>()
            .vortex_expect("non-nullable");
        for v in values.iter_mut() {
            *v += subseconds;
        }
    } else {
        let subseconds_buf = array.subseconds().clone().execute::<PrimitiveArray>(ctx)?;
        match_each_integer_ptype!(subseconds_buf.ptype(), |S| {
            for (v, subseconds) in values.iter_mut().zip(subseconds_buf.as_slice::<S>()) {
                let subseconds: i64 = subseconds.as_();
                *v += subseconds;
            }
        });
    }

    Ok(TemporalArray::new_timestamp(
        PrimitiveArray::new(values.freeze(), array.validity()?).into_array(),
        options.unit,
        options.tz.clone(),
    ))
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::TemporalArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::DateTimeParts;
    use crate::canonical::decode_to_temporal;

    #[rstest]
    #[case(Validity::NonNullable)]
    #[case(Validity::AllValid)]
    #[case(Validity::AllInvalid)]
    #[case(Validity::from_iter([true, true, false, false, true, true]))]
    fn test_decode_to_temporal(#[case] validity: Validity) -> VortexResult<()> {
        let milliseconds = PrimitiveArray::new(
            buffer![
                86_400i64, // element with only day component
                -86_400i64,
                86_400i64 + 1000, // element with day + second components
                -86_400i64 - 1000,
                86_400i64 + 1000 + 1, // element with day + second + sub-second components
                -86_400i64 - 1000 - 1
            ],
            validity.clone(),
        );
        let mut ctx = ExecutionCtx::new(VortexSession::empty());
        let date_times = DateTimeParts::try_from_temporal(
            TemporalArray::new_timestamp(
                milliseconds.clone().into_array(),
                TimeUnit::Milliseconds,
                Some("UTC".into()),
            ),
            &mut ctx,
        )?;

        assert!(
            date_times
                .as_array()
                .validity()?
                .mask_eq(&validity, &mut ctx)?
        );

        let primitive_values = decode_to_temporal(&date_times, &mut ctx)?
            .temporal_values()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;

        assert_arrays_eq!(primitive_values, milliseconds);
        assert!(
            primitive_values
                .validity()
                .unwrap()
                .mask_eq(&validity, &mut ctx)?
        );
        Ok(())
    }
}
