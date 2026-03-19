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
use vortex_array::vtable::ValidityHelper;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::DateTimePartsArray;

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

    // Validity is carried by the days component. Capture it before consuming the buffer.
    let validity = days_buf.validity().clone();

    let seconds_per_day: i64 = 86_400;
    let ticks_per_day: i64 = seconds_per_day * divisor;

    let seconds_const = array
        .seconds()
        .as_constant()
        .map(|s| s.as_primitive().as_::<i64>().vortex_expect("non-nullable"));
    let subseconds_const = array
        .subseconds()
        .as_constant()
        .map(|s| s.as_primitive().as_::<i64>().vortex_expect("non-nullable"));

    // Fused single-pass when both seconds and subseconds are constant (common case).
    let values: BufferMut<i64> =
        if let (Some(sec), Some(subsec)) = (seconds_const, subseconds_const) {
            let constant_offset = sec * divisor + subsec;
            days_buf
                .into_buffer_mut::<i64>()
                .map_each_in_place(|d| d * ticks_per_day + constant_offset)
        } else {
            let mut vals = days_buf
                .into_buffer_mut::<i64>()
                .map_each_in_place(|d| d * ticks_per_day);
            add_seconds_component(array, &mut vals, divisor, seconds_const, ctx)?;
            add_subseconds_component(array, &mut vals, subseconds_const, ctx)?;
            vals
        };

    Ok(TemporalArray::new_timestamp(
        PrimitiveArray::new(values.freeze(), validity).into_array(),
        options.unit,
        options.tz.clone(),
    ))
}

/// Add the seconds component to the values buffer.
fn add_seconds_component(
    array: &DateTimePartsArray,
    values: &mut BufferMut<i64>,
    divisor: i64,
    seconds_const: Option<i64>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    if let Some(seconds) = seconds_const {
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
    Ok(())
}

/// Add the subseconds component to the values buffer.
fn add_subseconds_component(
    array: &DateTimePartsArray,
    values: &mut BufferMut<i64>,
    subseconds_const: Option<i64>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    if let Some(subseconds) = subseconds_const {
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
    Ok(())
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
    use vortex_array::vtable::ValidityHelper;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::DateTimePartsArray;
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
        let date_times = DateTimePartsArray::try_from(TemporalArray::new_timestamp(
            milliseconds.clone().into_array(),
            TimeUnit::Milliseconds,
            Some("UTC".into()),
        ))
        .unwrap();

        let mut ctx = ExecutionCtx::new(VortexSession::empty());

        assert!(date_times.validity()?.mask_eq(&validity, &mut ctx)?);

        let primitive_values = decode_to_temporal(&date_times, &mut ctx)?
            .temporal_values()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;

        assert_arrays_eq!(primitive_values, milliseconds);
        assert!(primitive_values.validity().mask_eq(&validity, &mut ctx)?);
        Ok(())
    }
}
