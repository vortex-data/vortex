// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::DType;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::extension::datetime::Timestamp;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::array::DateTimePartsParts;

/// Decode [`DateTimePartsParts`] into a [`TemporalArray`].
pub fn decode_to_temporal(
    parts: DateTimePartsParts,
    dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<TemporalArray> {
    let DType::Extension(ext) = dtype else {
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

    // Days is guaranteed Primitive by require_child.
    let days = parts.days.as_::<Primitive>();
    let validity = days.validity()?;

    let mut values: BufferMut<i64> = match_each_integer_ptype!(days.ptype(), |D| {
        BufferMut::from_iter(days.as_slice::<D>().iter().map(|d| {
            let d: i64 = d.as_();
            d * 86_400 * divisor
        }))
    });

    // Seconds/subseconds may be Constant — handle the fast path.
    if let Some(seconds) = parts.seconds.as_constant() {
        let seconds = seconds
            .as_primitive()
            .as_::<i64>()
            .vortex_expect("non-nullable");
        let seconds = seconds * divisor;
        for v in values.iter_mut() {
            *v += seconds;
        }
    } else {
        let seconds_buf = parts.seconds.execute::<PrimitiveArray>(ctx)?;
        match_each_integer_ptype!(seconds_buf.ptype(), |S| {
            for (v, second) in values.iter_mut().zip(seconds_buf.as_slice::<S>()) {
                let second: i64 = second.as_();
                *v += second * divisor;
            }
        });
    }

    if let Some(subseconds) = parts.subseconds.as_constant() {
        let subseconds = subseconds
            .as_primitive()
            .as_::<i64>()
            .vortex_expect("non-nullable");
        for v in values.iter_mut() {
            *v += subseconds;
        }
    } else {
        let subseconds_buf = parts.subseconds.execute::<PrimitiveArray>(ctx)?;
        match_each_integer_ptype!(subseconds_buf.ptype(), |S| {
            for (v, subsecond) in values.iter_mut().zip(subseconds_buf.as_slice::<S>()) {
                let subsecond: i64 = subsecond.as_();
                *v += subsecond;
            }
        });
    }

    Ok(TemporalArray::new_timestamp(
        PrimitiveArray::new(values.freeze(), validity).into_array(),
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
    use crate::array::DateTimePartsArraySlotsExt;
    use crate::array::DateTimePartsParts;
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

        assert!(date_times.as_array().validity()?.mask_eq(
            &validity,
            milliseconds.len(),
            &mut ctx
        )?);

        let dtype = date_times.dtype().clone();
        let parts = DateTimePartsParts {
            days: date_times.days().clone(),
            seconds: date_times.seconds().clone(),
            subseconds: date_times.subseconds().clone(),
        };

        let primitive_values = decode_to_temporal(parts, &dtype, &mut ctx)?
            .temporal_values()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;

        assert_arrays_eq!(primitive_values, milliseconds);
        Ok(())
    }
}
