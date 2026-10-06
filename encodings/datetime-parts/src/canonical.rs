// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_array::IntoArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::extension::datetime::Timestamp;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::array::DateTimePartsParts;

/// Decode [`DateTimePartsParts`] into a [`TemporalArray`].
///
/// The days must be a primitive array. The seconds and subseconds must each be a constant or a
/// primitive array.
pub fn decode_to_temporal(parts: DateTimePartsParts, dtype: &DType) -> VortexResult<TemporalArray> {
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
    let days = parts.days.downcast::<Primitive>();
    let validity = days.validity()?;

    let mut values = days_to_values(days, 86_400 * divisor);

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
        let seconds_buf = parts.seconds.as_::<Primitive>();
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
        let subseconds_buf = parts.subseconds.as_::<Primitive>();
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

/// Scales the days into `i64` values of the target time unit.
///
/// The days buffer is scaled in place when it is `i64` and nothing else holds it.
fn days_to_values(days: PrimitiveArray, day_scale: i64) -> BufferMut<i64> {
    if days.ptype() != PType::I64 {
        return match_each_integer_ptype!(days.ptype(), |D| {
            BufferMut::from_iter(days.as_slice::<D>().iter().map(|d| {
                let d: i64 = d.as_();
                d * day_scale
            }))
        });
    }

    match days.try_into_buffer_mut::<i64>() {
        Ok(mut values) => {
            for v in values.iter_mut() {
                *v *= day_scale;
            }
            values
        }
        Err(days) => BufferMut::from_iter(days.iter().map(|d| d * day_scale)),
    }
}

#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
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

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

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
        let mut ctx = SESSION.create_execution_ctx();
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

        let primitive_values = decode_to_temporal(parts, &dtype)?
            .temporal_values()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;

        assert_arrays_eq!(primitive_values, milliseconds, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::unique(false)]
    #[case::shared(true)]
    fn test_decode_to_temporal_i64_days(#[case] shared: bool) -> VortexResult<()> {
        let days = PrimitiveArray::new(buffer![0i64, 1, -1, 2], Validity::NonNullable);
        let kept_days = shared.then(|| days.clone());
        let dtype = TemporalArray::new_timestamp(
            PrimitiveArray::new(buffer![0i64; 4], Validity::NonNullable).into_array(),
            TimeUnit::Milliseconds,
            Some("UTC".into()),
        )
        .dtype()
        .clone();
        let parts = DateTimePartsParts {
            days: days.into_array(),
            seconds: ConstantArray::new(1i64, 4).into_array(),
            subseconds: ConstantArray::new(2i64, 4).into_array(),
        };

        let mut ctx = SESSION.create_execution_ctx();
        let primitive_values = decode_to_temporal(parts, &dtype)?
            .temporal_values()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;

        let expected = PrimitiveArray::from_iter([1_002i64, 86_401_002, -86_398_998, 172_801_002]);
        assert_arrays_eq!(primitive_values, expected, &mut ctx);

        if let Some(kept_days) = kept_days {
            let unchanged = PrimitiveArray::from_iter([0i64, 1, -1, 2]);
            assert_arrays_eq!(kept_days, unchanged, &mut ctx);
        }
        Ok(())
    }
}
