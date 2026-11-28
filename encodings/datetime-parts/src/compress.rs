// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Splits temporal arrays into components for independent compression.
//!
//! The split keeps days in `i32` when possible and widens only that component to `i64` for extreme
//! second or millisecond timestamps. The split checks the computed days as it writes the narrow
//! buffer. If any day exceeds `i32`, it checks the non-null days before keeping a wide buffer.
//! Null slots do not restrict the range of an array.

use std::mem::MaybeUninit;

use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::min_max::MinMaxResult;
use vortex_array::aggregate_fn::fns::min_max::min_max;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::extension::datetime::TimeUnit;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::timestamp;
use crate::timestamp::temporal_kind;

/// The day, second, and subsecond components of a temporal array.
pub struct TemporalParts {
    /// Days since the Unix epoch (or `0` for `Time` values), carrying the validity of the
    /// original temporal array.
    ///
    /// Uses `i32` when all non-null days fit, otherwise `i64`.
    pub days: ArrayRef,
    /// Seconds within the day, with non-nullable storage.
    pub seconds: ArrayRef,
    /// Subseconds within the second, with non-nullable storage.
    pub subseconds: ArrayRef,
}

/// The day, second, and scaled subsecond components of a temporal array.
pub struct ScaledTemporalParts {
    /// Days since the Unix epoch (or `0` for `Time` values), carrying the validity of the
    /// original temporal array.
    ///
    /// Uses `i32` when all non-null days fit, otherwise `i64`.
    pub days: ArrayRef,
    /// Seconds within the day, with non-nullable storage.
    pub seconds: ArrayRef,
    /// Subseconds within the second divided by [`Self::subseconds_divisor`], with non-nullable
    /// storage.
    pub subseconds: ArrayRef,
    /// Power-of-1000 scaling divisor applied to [`Self::subseconds`] (`1`, `1_000`, or
    /// `1_000_000`).
    pub subseconds_divisor: u32,
}

/// Splits a [`TemporalArray`] (`Timestamp` or `Time`) into day, second, and unscaled subsecond
/// components (`subseconds_divisor = 1`).
///
/// Splitting the components by granularity creates more small values, which enables better
/// cascading compression.
///
/// # Errors
///
/// Returns an error for unsupported temporal types, day units, or if the storage cannot be
/// executed.
pub fn split_temporal(array: TemporalArray, ctx: &mut ExecutionCtx) -> VortexResult<TemporalParts> {
    let ScaledTemporalParts {
        days,
        seconds,
        subseconds,
        ..
    } = split_temporal_impl(array, false, ctx)?;
    Ok(TemporalParts {
        days,
        seconds,
        subseconds,
    })
}

/// Splits a [`TemporalArray`] (`Timestamp` or `Time`) into day, second, and optionally scaled
/// subsecond components.
///
/// For `Microseconds` and `Nanoseconds` arrays, inspects only valid (non-null) rows to detect
/// whether all subseconds are zero or evenly divisible by `1_000_000` (for `Nanoseconds`) or
/// `1_000` (for `Microseconds` and `Nanoseconds`). When all valid subseconds are zero, emits a
/// constant-zero subseconds array with `subseconds_divisor = 1`.
///
/// # Errors
///
/// Returns an error for unsupported temporal types, day units, or if the storage cannot be
/// executed.
pub fn split_temporal_v2(
    array: TemporalArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ScaledTemporalParts> {
    split_temporal_impl(array, true, ctx)
}

fn split_temporal_impl(
    array: TemporalArray,
    detect_divisor: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ScaledTemporalParts> {
    let time_unit = temporal_kind(array.dtype())?.unit();

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
        let all_days_fit_i32 = split_slice_seconds(days_ptr, seconds_ptr, timestamps);

        // SAFETY: `split_slice_seconds` initialized all `length` elements of both buffers.
        unsafe {
            days.set_len(length);
            seconds.set_len(length);
        }

        return Ok(ScaledTemporalParts {
            days: widen_days_if_needed(
                PrimitiveArray::new(days.freeze(), temporal_values.validity()?),
                timestamps,
                time_unit,
                all_days_fit_i32,
                ctx,
            )?,
            seconds: seconds.into_array(),
            subseconds: ConstantArray::new(0, length).into_array(),
            subseconds_divisor: 1,
        });
    }

    let mut subseconds: BufferMut<i32> = BufferMut::with_capacity(timestamps.len());
    let subseconds_ptr = &mut subseconds.spare_capacity_mut()[..length];

    let all_days_fit_i32 = match time_unit {
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

    // SAFETY: `split_slice` initialized all `length` elements of the three buffers.
    unsafe {
        days.set_len(length);
        seconds.set_len(length);
        subseconds.set_len(length);
    }

    let validity = temporal_values.validity()?;
    let (subseconds_array, subseconds_divisor) =
        if detect_divisor && matches!(time_unit, TimeUnit::Microseconds | TimeUnit::Nanoseconds) {
            let mask = validity.execute_mask(length, ctx)?;
            match detect_subseconds_divisor(subseconds.as_slice(), &mask, time_unit) {
                DetectedSubseconds::AllZero => (ConstantArray::new(0i32, length).into_array(), 1),
                DetectedSubseconds::Scaled(divisor) => {
                    let scaled = match divisor {
                        1_000_000 => scale_subseconds::<1_000_000>(subseconds, &mask),
                        1_000 => scale_subseconds::<1_000>(subseconds, &mask),
                        _ => unreachable!("only 1_000 and 1_000_000 are valid scaled divisors"),
                    };
                    (scaled.into_array(), divisor)
                }
                DetectedSubseconds::Unscaled => (subseconds.into_array(), 1),
            }
        } else {
            (subseconds.into_array(), 1)
        };

    Ok(ScaledTemporalParts {
        days: widen_days_if_needed(
            PrimitiveArray::new(days.freeze(), validity),
            timestamps,
            time_unit,
            all_days_fit_i32,
            ctx,
        )?,
        seconds: seconds.into_array(),
        subseconds: subseconds_array,
        subseconds_divisor,
    })
}

enum DetectedSubseconds {
    /// Every valid row has subseconds == 0 (including empty or all-null arrays).
    AllZero,
    /// At least one valid row has non-zero subseconds, and every valid row is divisible by the
    /// returned power-of-1000 divisor (`1_000` or `1_000_000`).
    Scaled(u32),
    /// Valid rows require full resolution (`subseconds_divisor = 1`).
    Unscaled,
}

fn detect_subseconds_divisor(
    subseconds: &[i32],
    mask: &Mask,
    time_unit: TimeUnit,
) -> DetectedSubseconds {
    let check_1m = matches!(time_unit, TimeUnit::Nanoseconds);
    let mut any_nonzero = false;
    let mut div_1m = check_1m;
    let mut div_1k = true;

    let mut visit = |s: i32| -> bool {
        if s != 0 {
            any_nonzero = true;
            if s % 1_000 != 0 {
                div_1k = false;
                div_1m = false;
                return false;
            }
            if div_1m && s % 1_000_000 != 0 {
                div_1m = false;
            }
        }
        true
    };

    match mask {
        Mask::AllFalse(_) => return DetectedSubseconds::AllZero,
        Mask::AllTrue(_) => {
            for &s in subseconds {
                if !visit(s) {
                    break;
                }
            }
        }
        Mask::Values(values) => {
            let _ = values.bit_buffer().try_for_each_set_index(|idx| {
                if visit(subseconds[idx]) {
                    Ok(())
                } else {
                    Err(())
                }
            });
        }
    }

    if !any_nonzero {
        DetectedSubseconds::AllZero
    } else if div_1m {
        DetectedSubseconds::Scaled(1_000_000)
    } else if div_1k {
        DetectedSubseconds::Scaled(1_000)
    } else {
        DetectedSubseconds::Unscaled
    }
}

fn scale_subseconds<const DIVISOR: i32>(
    mut subseconds: BufferMut<i32>,
    mask: &Mask,
) -> BufferMut<i32> {
    match mask {
        Mask::AllTrue(_) => {
            for s in subseconds.as_mut_slice() {
                *s /= DIVISOR;
            }
            subseconds
        }
        Mask::Values(values) => {
            let src = subseconds.as_slice();
            let mut out = BufferMut::<i32>::zeroed(src.len());
            let dst = out.as_mut_slice();
            values.bit_buffer().for_each_set_index(|idx| {
                dst[idx] = src[idx] / DIVISOR;
            });
            out
        }
        Mask::AllFalse(_) => unreachable!("AllFalse returns DetectedSubseconds::AllZero"),
    }
}

/// Keeps widened days only when a non-null day exceeds the `i32` range.
///
/// The split loops may truncate days while writing the candidate `i32` buffer. Keep that buffer
/// only if all days fit or the non-null bounds confirm that only null slots were truncated.
fn widen_days_if_needed(
    days: PrimitiveArray,
    timestamps: &[i64],
    time_unit: TimeUnit,
    all_days_fit_i32: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if all_days_fit_i32 {
        return Ok(days.into_array());
    }

    let ticks_per_day = match time_unit {
        TimeUnit::Seconds => timestamp::SECONDS_PER_DAY,
        TimeUnit::Milliseconds => timestamp::SECONDS_PER_DAY * 1_000,
        _ => unreachable!("microsecond and nanosecond days always fit i32"),
    };
    let wide_days = PrimitiveArray::new(
        Buffer::from_iter(timestamps.iter().map(|ts| ts / ticks_per_day)),
        days.validity()?,
    )
    .into_array();

    // The split checks null slots too. Their values must not force the days buffer to widen.
    let fits = match min_max(&wide_days, ctx, NumericalAggregateOpts::default())? {
        Some(MinMaxResult { min, max }) => [min, max]
            .iter()
            .all(|bound| bound.as_primitive().as_opt::<i32>().is_some()),
        None => true,
    };

    Ok(if fits { days.into_array() } else { wide_days })
}

/// Adds 2^31 to the days and converts to unsigned, placing in-range days below 2^32.
///
/// Days below `i32::MIN` become large unsigned values; days above `i32::MAX` reach at least 2^32.
/// Both set a bit above bit 31. The input must come from splitting an `i64` timestamp, whose days
/// leave enough room for the bias addition in `i64`.
#[inline]
fn day_offset(days: i64) -> u64 {
    (days - i64::from(i32::MIN)).cast_unsigned()
}

/// Initializes all three output slices and reports whether the days fit `i32`.
///
/// The slices must have the same length.
#[inline]
fn split_slice<const DIVISOR: i64>(
    days: &mut [MaybeUninit<i32>],
    seconds: &mut [MaybeUninit<i32>],
    subseconds: &mut [MaybeUninit<i32>],
    timestamps: &[i64],
) -> bool {
    // OR preserves any high bits from `day_offset`, so the final high bits are zero only if every
    // day fits i32. `widen_days_if_needed` checks this result before keeping the narrow buffer.
    let mut day_range_bits = 0u64;

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
            day_range_bits |= day_offset(parts.days);
            day_buf[k] = parts.days.as_();
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
        day_range_bits |= day_offset(parts.days);
        day.write(parts.days.as_());
        second.write(parts.seconds);
        subsecond.write(parts.subseconds);
    }

    day_range_bits >> 32 == 0
}

/// Initializes the days and seconds slices for second timestamps.
///
/// The slices must have the same length. Returns whether all days fit `i32`.
#[inline]
fn split_slice_seconds(
    days: &mut [MaybeUninit<i32>],
    seconds: &mut [MaybeUninit<i32>],
    timestamps: &[i64],
) -> bool {
    // OR preserves any high bits from `day_offset`, so a later in-range day cannot hide overflow.
    // `widen_days_if_needed` checks this result before keeping the narrow buffer.
    let mut day_range_bits = 0u64;
    for ((day, second), ts) in days.iter_mut().zip(seconds).zip(timestamps) {
        let parts = timestamp::split_with_divisor::<1>(*ts);
        day_range_bits |= day_offset(parts.days);
        day.write(parts.days.as_());
        second.write(parts.seconds);
    }

    day_range_bits >> 32 == 0
}

#[cfg(test)]
mod tests {
    use std::iter;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::TemporalArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::PType;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::serde::SerializedArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::ByteBufferMut;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use super::day_offset;
    use crate::DateTimeParts;
    use crate::TemporalParts;
    use crate::array::DateTimePartsArraySlotsExt;
    use crate::split_temporal;
    use crate::timestamp::SECONDS_PER_DAY;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    #[rstest]
    #[case::empty(&[])]
    #[case::zero(&[0])]
    #[case::boundaries(&[i64::from(i32::MIN), i64::from(i32::MAX)])]
    #[case::below_min(&[i64::from(i32::MIN) - 1])]
    #[case::above_max(&[i64::from(i32::MAX) + 1])]
    #[case::both_overflows(&[i64::from(i32::MIN) - 1, 0, i64::from(i32::MAX) + 1, 0])]
    fn test_day_range_check(#[case] days: &[i64]) -> VortexResult<()> {
        let day_range_bits = days.iter().fold(0u64, |bits, &day| bits | day_offset(day));
        assert_eq!(
            day_range_bits >> 32 == 0,
            days.iter().all(|&day| i32::try_from(day).is_ok())
        );

        Ok(())
    }

    #[rstest]
    #[case(Validity::NonNullable)]
    #[case(Validity::AllValid)]
    #[case(Validity::AllInvalid)]
    #[case(Validity::from_iter([true, false, true]))]
    fn test_split_temporal(#[case] validity: Validity) {
        let mut ctx = SESSION.create_execution_ctx();
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

    #[rstest]
    #[case::seconds(TimeUnit::Seconds)]
    #[case::milliseconds(TimeUnit::Milliseconds)]
    #[case::microseconds(TimeUnit::Microseconds)]
    #[case::nanoseconds(TimeUnit::Nanoseconds)]
    fn test_empty_timestamps_stay_narrow(#[case] unit: TimeUnit) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let temporal = TemporalArray::new_timestamp(
            PrimitiveArray::from_iter(iter::empty::<i64>()).into_array(),
            unit,
            None,
        );
        let parts = DateTimeParts::try_from_temporal(temporal.clone(), &mut ctx)?;
        assert_eq!(parts.days().dtype().as_ptype(), PType::I32);
        assert_arrays_eq!(parts, temporal, &mut ctx);

        Ok(())
    }

    #[rstest]
    #[case::seconds(TimeUnit::Seconds, PType::I64)]
    #[case::milliseconds(TimeUnit::Milliseconds, PType::I64)]
    #[case::microseconds(TimeUnit::Microseconds, PType::I32)]
    #[case::nanoseconds(TimeUnit::Nanoseconds, PType::I32)]
    fn test_extreme_timestamps_roundtrip(
        #[case] unit: TimeUnit,
        #[case] days_ptype: PType,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let temporal = TemporalArray::new_timestamp(
            PrimitiveArray::from_option_iter([
                Some(i64::MAX), // Largest storage value.
                None,           // Null row.
                Some(i64::MIN), // Smallest storage value.
                Some(0),        // Unix epoch.
            ])
            .into_array(),
            unit,
            Some("UTC".into()),
        );
        let expected = temporal.clone().into_array();

        let parts = DateTimeParts::try_from_temporal(temporal, &mut ctx)?;
        assert_eq!(parts.days().dtype().as_ptype(), days_ptype);
        assert_arrays_eq!(&parts, &expected, &mut ctx);
        for index in 0..expected.len() {
            assert_eq!(
                parts.execute_scalar(index, &mut ctx)?,
                expected.execute_scalar(index, &mut ctx)?,
            );
        }

        let array_ctx = ArrayContext::empty();
        let serialized = parts.clone().into_array().serialize(
            &array_ctx,
            &SESSION,
            &SerializeOptions::default(),
        )?;
        let mut bytes = ByteBufferMut::empty();
        for buffer in serialized {
            bytes.extend_from_slice(&buffer);
        }
        let decoded = SerializedArray::try_from(bytes.freeze())?.decode(
            expected.dtype(),
            expected.len(),
            &ReadContext::new(array_ctx.to_ids()),
            &SESSION,
        )?;
        assert_eq!(
            decoded.as_::<DateTimeParts>().days().dtype().as_ptype(),
            days_ptype
        );
        assert_arrays_eq!(decoded, expected, &mut ctx);

        Ok(())
    }

    #[rstest]
    #[case::seconds(TimeUnit::Seconds, 1)]
    #[case::milliseconds(TimeUnit::Milliseconds, 1_000)]
    fn test_days_at_i32_bounds(
        #[case] unit: TimeUnit,
        #[case] ticks_per_second: i64,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let ticks_per_day = SECONDS_PER_DAY * ticks_per_second;
        let max = i64::from(i32::MAX) * ticks_per_day + ticks_per_day - 1;
        let min = i64::from(i32::MIN) * ticks_per_day - ticks_per_day + 1;

        let temporal = |values: Vec<i64>| {
            TemporalArray::new_timestamp(PrimitiveArray::from_iter(values).into_array(), unit, None)
        };
        let fits = temporal(vec![max, 0, min, max, min, 0, max, min, max]);
        let parts = DateTimeParts::try_from_temporal(fits.clone(), &mut ctx)?;
        assert_eq!(parts.days().dtype().as_ptype(), PType::I32);
        assert_arrays_eq!(parts, fits, &mut ctx);

        // One past each bound, in the 4-wide chunks and in the remainder.
        for (index, past) in [(1, max + 1), (8, min - 1)] {
            let mut values = vec![0; 9];
            values[index] = past;
            let temporal = temporal(values);
            let parts = DateTimeParts::try_from_temporal(temporal.clone(), &mut ctx)?;
            assert_eq!(parts.days().dtype().as_ptype(), PType::I64);
            assert_arrays_eq!(parts, temporal, &mut ctx);
        }

        Ok(())
    }

    #[rstest]
    #[case::seconds(TimeUnit::Seconds)]
    #[case::milliseconds(TimeUnit::Milliseconds)]
    fn test_out_of_range_null_slots_split(
        #[case] unit: TimeUnit,
        #[values(Validity::from_iter([true, false, true, false, true]), Validity::AllInvalid)]
        validity: Validity,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::new(
            buffer![
                0i64,     // Valid epoch.
                i64::MAX, // Null slot outside the days range.
                86_400,   // Valid next day.
                i64::MIN, // Null slot outside the days range.
                1,        // Valid next second.
            ],
            validity,
        );
        let temporal = TemporalArray::new_timestamp(values.into_array(), unit, None);
        let parts = DateTimeParts::try_from_temporal(temporal.clone(), &mut ctx)?;
        assert_eq!(parts.days().dtype().as_ptype(), PType::I32);
        assert_arrays_eq!(parts, temporal, &mut ctx);

        Ok(())
    }

    #[test]
    fn test_split_v2_ignores_dirty_null_slots_for_divisor() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Valid rows are all ms-precision in a µs timestamp; null slots hold arbitrary
        // non-multiple garbage (including negative values).
        let values = PrimitiveArray::new(
            buffer![
                1_700_000_000_000_000i64,
                1_700_000_000_123_456, // null slot: not multiple of 1_000
                1_700_000_001_500_000,
                -987_654_321, // null slot: negative non-multiple
                -86_400_250_000,
            ],
            Validity::from_iter([true, false, true, false, true]),
        );
        let temporal = TemporalArray::new_timestamp(
            values.into_array(),
            TimeUnit::Microseconds,
            Some("UTC".into()),
        );
        let parts = DateTimeParts::try_from_temporal_v2(temporal.clone(), &mut ctx)?;
        assert_eq!(parts.subseconds_divisor(), 1_000);
        // Null slots in scaled subseconds are zeroed so garbage does not leak.
        let sub_prim = parts
            .subseconds()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(sub_prim.as_slice::<i32>(), &[0, 0, 500, 0, -250]);
        assert_arrays_eq!(parts, temporal, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_split_v2_all_zero_valid_subseconds_stays_v1_divisor() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::new(
            buffer![
                1_700_000_000_000_000i64,
                1_700_000_000_123_456, // null slot with non-zero garbage
                1_700_000_005_000_000,
            ],
            Validity::from_iter([true, false, true]),
        );
        let temporal =
            TemporalArray::new_timestamp(values.into_array(), TimeUnit::Microseconds, None);
        let parts = DateTimeParts::try_from_temporal_v2(temporal.clone(), &mut ctx)?;
        assert_eq!(parts.subseconds_divisor(), 1);
        assert!(parts.subseconds().as_constant().is_some());
        assert_arrays_eq!(parts, temporal, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::all_null(Validity::AllInvalid)]
    #[case::empty(Validity::NonNullable)]
    fn test_split_v2_empty_and_all_null_keep_divisor_one(
        #[case] validity: Validity,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = if matches!(validity, Validity::AllInvalid) {
            PrimitiveArray::new(buffer![123i64, 456, 789], validity)
        } else {
            PrimitiveArray::from_iter(iter::empty::<i64>())
        };
        let temporal =
            TemporalArray::new_timestamp(values.into_array(), TimeUnit::Nanoseconds, None);
        let parts = DateTimeParts::try_from_temporal_v2(temporal.clone(), &mut ctx)?;
        assert_eq!(parts.subseconds_divisor(), 1);
        assert_arrays_eq!(parts, temporal, &mut ctx);
        Ok(())
    }
}
