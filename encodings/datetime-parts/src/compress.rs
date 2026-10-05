// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::extension::datetime::TimeUnit;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::timestamp;
/// All parts are stored as i32. seconds and subseconds always fit in i32.
/// For days we have a upper day limitation due to "jiff" validation so days
/// also fit in i32.
pub struct TemporalParts {
    pub days: ArrayRef,
    pub seconds: ArrayRef,
    pub subseconds: ArrayRef,
}

/// Compress a `TemporalArray` into day, second, and subseconds components.
///
/// Splitting the components by granularity creates more small values, which enables better
/// cascading compression.
pub fn split_temporal(array: TemporalArray, ctx: &mut ExecutionCtx) -> VortexResult<TemporalParts> {
    let time_unit = array.temporal_metadata().time_unit();
    if matches!(time_unit, TimeUnit::Days) {
        vortex_bail!("Cannot handle day-level data");
    }

    let temporal_values = array
        .temporal_values()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;

    // After this operation, timestamps will be a PrimitiveArray<i64>
    let timestamps = temporal_values
        .clone()
        .into_array()
        .cast(DType::Primitive(
            PType::I64,
            temporal_values.dtype().nullability(),
        ))?
        .execute::<PrimitiveArray>(ctx)?;

    let length = timestamps.len();

    // If we don't [..length], compiler can't infer all 3 or 4 slices are the
    // same length, and zip() in split_slice_* checks iterator boundaries.
    let timestamps = &timestamps.as_slice::<i64>()[..length];

    let mut days: BufferMut<i32> = BufferMut::with_capacity(timestamps.len());
    let mut seconds: BufferMut<i32> = BufferMut::with_capacity(timestamps.len());
    let days_ptr = &mut days.spare_capacity_mut()[..length];
    let seconds_ptr = &mut seconds.spare_capacity_mut()[..length];

    if matches!(time_unit, TimeUnit::Seconds) {
        split_slice_seconds(days_ptr, seconds_ptr, timestamps);

        // SAFETY: all items in [0; length) are filled in split_slice_seconds
        unsafe {
            days.set_len(length);
            seconds.set_len(length);
        }

        return Ok(TemporalParts {
            days: PrimitiveArray::new(days.freeze(), temporal_values.validity()?).into_array(),
            seconds: seconds.into_array(),
            subseconds: ConstantArray::new(0, length).into_array(),
        });
    }

    let mut subseconds: BufferMut<i32> = BufferMut::with_capacity(timestamps.len());
    let subseconds_ptr = &mut subseconds.spare_capacity_mut()[..length];

    match time_unit {
        TimeUnit::Nanoseconds => {
            split_slice::<1_000_000_000>(days_ptr, seconds_ptr, subseconds_ptr, timestamps)
        }
        TimeUnit::Microseconds => {
            split_slice::<1_000_000>(days_ptr, seconds_ptr, subseconds_ptr, timestamps)
        }
        TimeUnit::Milliseconds => {
            split_slice::<1_000>(days_ptr, seconds_ptr, subseconds_ptr, timestamps)
        }
        _ => unreachable!("Handled before"),
    };

    // SAFETY: all items in [0; length) are filled in split_slice
    unsafe {
        days.set_len(length);
        seconds.set_len(length);
        subseconds.set_len(length);
    }

    Ok(TemporalParts {
        days: PrimitiveArray::new(days.freeze(), temporal_values.validity()?).into_array(),
        seconds: seconds.into_array(),
        subseconds: subseconds.into_array(),
    })
}

#[inline]
fn split_slice<const DIVISOR: i64>(
    days: &mut [MaybeUninit<i32>],
    seconds: &mut [MaybeUninit<i32>],
    subseconds: &mut [MaybeUninit<i32>],
    timestamps: &[i64],
) {
    // Computing chunks of 4 elements lets LLVM optimize stores into
    // a 16-byte vector store per chunk.
    let length = timestamps.len();
    let (timestamps, timestamps_rem) = timestamps.as_chunks::<4>();
    let (days, days_rem) = days[..length].as_chunks_mut::<4>();
    let (seconds, seconds_rem) = seconds[..length].as_chunks_mut::<4>();
    let (subseconds, subseconds_rem) = subseconds[..length].as_chunks_mut::<4>();

    let chunks = timestamps.iter().zip(days).zip(seconds).zip(subseconds);
    for (((timestamp, day), second), subsecond) in chunks {
        let mut day_buf = [0i32; 4];
        let mut second_buf = [0i32; 4];
        let mut subsecond_buf = [0i32; 4];
        for k in 0..4 {
            let parts = timestamp::split_with_divisor::<DIVISOR>(timestamp[k]);
            day_buf[k] = parts.days;
            second_buf[k] = parts.seconds;
            subsecond_buf[k] = parts.subseconds;
        }
        for k in 0..4 {
            day[k].write(day_buf[k]);
            second[k].write(second_buf[k]);
            subsecond[k].write(subsecond_buf[k]);
        }
    }

    let remainder = timestamps_rem
        .iter()
        .zip(days_rem)
        .zip(seconds_rem)
        .zip(subseconds_rem);
    for (((&ts, day), second), subsecond) in remainder {
        let parts = timestamp::split_with_divisor::<DIVISOR>(ts);
        day.write(parts.days);
        second.write(parts.seconds);
        subsecond.write(parts.subseconds);
    }
}

#[inline]
fn split_slice_seconds(
    days: &mut [MaybeUninit<i32>],
    seconds: &mut [MaybeUninit<i32>],
    timestamps: &[i64],
) {
    for ((day, second), ts) in days.iter_mut().zip(seconds).zip(timestamps) {
        let parts = timestamp::split_with_divisor::<1>(*ts);
        day.write(parts.days);
        second.write(parts.seconds);
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
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
        let mut ctx = array_session().create_execution_ctx();
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
        } = split_temporal(temporal_array, &mut ctx).unwrap();

        let days_prim = days.execute::<PrimitiveArray>(&mut ctx).unwrap();
        assert!(
            days_prim
                .validity()
                .vortex_expect("days validity should be derivable")
                .mask_eq(&validity, days_prim.len(), &mut ctx)
                .unwrap()
        );
        let seconds_prim = seconds.execute::<PrimitiveArray>(&mut ctx).unwrap();
        assert!(matches!(
            seconds_prim
                .validity()
                .vortex_expect("seconds validity should be derivable"),
            Validity::NonNullable
        ));
        let subseconds_prim = subseconds.execute::<PrimitiveArray>(&mut ctx).unwrap();
        assert!(matches!(
            subseconds_prim
                .validity()
                .vortex_expect("subseconds validity should be derivable"),
            Validity::NonNullable
        ));
    }
}
