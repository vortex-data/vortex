// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Add;

use num_traits::AsPrimitive;
use num_traits::ToPrimitive;
use vortex_buffer::BitBuffer;
use vortex_compute::lane_kernels::IndexedFoldExt;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::Mask;
use vortex_mask::MaskValues;

use super::SumState;
use super::checked_add_i64;
use super::checked_add_u64;
use crate::ExecutionCtx;
use crate::arrays::PrimitiveArray;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::match_each_native_ptype;

/// Number of elements summed without an overflow check. Chosen so that a chunk of values narrower
/// than 64 bits cannot overflow the 64-bit accumulator: `2^16 * (2^32 - 1) < 2^64`.
const SUM_CHUNK: usize = 1 << 16;

pub(crate) fn accumulate_primitive(
    inner: &mut SumState,
    p: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
    skip_nans: bool,
) -> VortexResult<bool> {
    match p.as_ref().validity()?.execute_mask(p.as_ref().len(), ctx)? {
        Mask::AllFalse(_) => Ok(false),
        Mask::AllTrue(_) => accumulate_primitive_all(inner, p, skip_nans),
        Mask::Values(valid) => accumulate_primitive_valid(inner, p, &valid, skip_nans),
    }
}

fn accumulate_primitive_all(
    inner: &mut SumState,
    p: &PrimitiveArray,
    skip_nans: bool,
) -> VortexResult<bool> {
    match inner {
        SumState::Unsigned(acc) => match_each_native_ptype!(p.ptype(),
            unsigned: |T| { Ok(sum_unsigned_all(acc, p.as_slice::<T>())) },
            signed: |_T| { vortex_panic!("unsigned sum state with signed input") },
            floating: |_T| { vortex_panic!("unsigned sum state with float input") }
        ),
        SumState::Signed(acc) => match_each_native_ptype!(p.ptype(),
            unsigned: |_T| { vortex_panic!("signed sum state with unsigned input") },
            signed: |T| { Ok(sum_signed_all(acc, p.as_slice::<T>())) },
            floating: |_T| { vortex_panic!("signed sum state with float input") }
        ),
        SumState::Float(acc) => match_each_native_ptype!(p.ptype(),
            unsigned: |_T| { vortex_panic!("float sum state with unsigned input") },
            signed: |_T| { vortex_panic!("float sum state with signed input") },
            floating: |T| {
                sum_float_all(acc, p.as_slice::<T>(), skip_nans);
                Ok(false)
            }
        ),
        SumState::Decimal(_) => {
            vortex_panic!("decimal sum state with primitive input")
        }
    }
}

/// Sum the values of a float slice into an `f64` accumulator. When `skip_nans` is set, NaN values
/// are skipped to match the scalar `sum` semantics; otherwise any NaN poisons the accumulator to
/// NaN. Floats cannot overflow the accumulator, so this never reports saturation.
pub(crate) fn sum_float_all<T: NativePType>(acc: &mut f64, slice: &[T], skip_nans: bool) {
    if skip_nans {
        for &v in slice {
            if !v.is_nan() {
                *acc += ToPrimitive::to_f64(&v).vortex_expect("float to f64");
            }
        }
    } else {
        for &v in slice {
            *acc += ToPrimitive::to_f64(&v).vortex_expect("float to f64");
        }
    }
}

/// Sum all values into a `u64` accumulator. For types narrower than 64 bits, values are summed in
/// chunks of [`SUM_CHUNK`] with a single checked add per chunk, which lets the inner loop vectorize
/// to packed widening adds. `u64` input keeps a per-element checked add since a chunk of `u64`s
/// could itself overflow. Returns `true` on overflow.
pub(crate) fn sum_unsigned_all<T>(acc: &mut u64, slice: &[T]) -> bool
where
    T: NativePType + AsPrimitive<u64>,
{
    if T::PTYPE == PType::U64 {
        for &v in slice {
            if checked_add_u64(acc, v.as_()) {
                return true;
            }
        }
        return false;
    }
    for chunk in slice.chunks(SUM_CHUNK) {
        let chunk_sum: u64 = chunk.iter().map(|&v| v.as_()).sum();
        if checked_add_u64(acc, chunk_sum) {
            return true;
        }
    }
    false
}

/// Signed counterpart of [`sum_unsigned_all`].
pub(crate) fn sum_signed_all<T>(acc: &mut i64, slice: &[T]) -> bool
where
    T: NativePType + AsPrimitive<i64>,
{
    if T::PTYPE == PType::I64 {
        for &v in slice {
            if checked_add_i64(acc, v.as_()) {
                return true;
            }
        }
        return false;
    }
    for chunk in slice.chunks(SUM_CHUNK) {
        let chunk_sum: i64 = chunk.iter().map(|&v| v.as_()).sum();
        if checked_add_i64(acc, chunk_sum) {
            return true;
        }
    }
    false
}

/// Sum the valid elements of `p`, whose validity is `valid`.
fn accumulate_primitive_valid(
    inner: &mut SumState,
    p: &PrimitiveArray,
    valid: &MaskValues,
    skip_nans: bool,
) -> VortexResult<bool> {
    let valid = valid.bit_buffer();
    match inner {
        SumState::Unsigned(acc) => match_each_native_ptype!(p.ptype(),
            unsigned: |T| { Ok(sum_unsigned_masked(acc, p.as_slice::<T>(), valid)) },
            signed: |_T| { vortex_panic!("unsigned sum state with signed input") },
            floating: |_T| { vortex_panic!("unsigned sum state with float input") }
        ),
        SumState::Signed(acc) => match_each_native_ptype!(p.ptype(),
            unsigned: |_T| { vortex_panic!("signed sum state with unsigned input") },
            signed: |T| { Ok(sum_signed_masked(acc, p.as_slice::<T>(), valid)) },
            floating: |_T| { vortex_panic!("signed sum state with float input") }
        ),
        SumState::Float(acc) => match_each_native_ptype!(p.ptype(),
            unsigned: |_T| { vortex_panic!("float sum state with unsigned input") },
            signed: |_T| { vortex_panic!("float sum state with signed input") },
            floating: |T| {
                sum_float_masked(acc, p.as_slice::<T>(), valid, skip_nans);
                Ok(false)
            }
        ),
        SumState::Decimal(_) => {
            vortex_panic!("decimal sum state with primitive input")
        }
    }
}

/// Sum the values of `values` that `valid` marks valid into a `u64` accumulator. Returns `true` on
/// overflow.
///
/// Values narrower than 64 bits read null lanes as zero, so the sum vectorizes the same way as
/// [`sum_unsigned_all`] however the valid values are spread. `u64` values check every element
/// for overflow, so they sum each contiguous run of valid values through [`sum_unsigned_all`].
fn sum_unsigned_masked<T>(acc: &mut u64, values: &[T], valid: &BitBuffer) -> bool
where
    T: NativePType + AsPrimitive<u64>,
{
    if T::PTYPE == PType::U64 {
        return valid
            .set_slices()
            .any(|(start, end)| sum_unsigned_all(acc, &values[start..end]));
    }
    sum_chunks_masked(values, valid, |chunk_sum| checked_add_u64(acc, chunk_sum))
}

/// Signed counterpart of [`sum_unsigned_masked`].
fn sum_signed_masked<T>(acc: &mut i64, values: &[T], valid: &BitBuffer) -> bool
where
    T: NativePType + AsPrimitive<i64>,
{
    if T::PTYPE == PType::I64 {
        return valid
            .set_slices()
            .any(|(start, end)| sum_signed_all(acc, &values[start..end]));
    }
    sum_chunks_masked(values, valid, |chunk_sum| checked_add_i64(acc, chunk_sum))
}

/// Sum the values of `values` that `valid` marks valid into an `f64` accumulator, skipping NaN
/// values when `skip_nans` is set, as [`sum_float_all`] does.
///
/// Null and skipped lanes read as zero. The values are added in order, as by [`sum_float_all`],
/// and adding zero leaves the accumulator unchanged: it starts at positive zero, so it is never
/// negative zero, the one value adding zero would change. The result is therefore exactly that of
/// summing the valid values alone, without walking their runs.
fn sum_float_masked<T: NativePType>(
    acc: &mut f64,
    values: &[T],
    valid: &BitBuffer,
    skip_nans: bool,
) {
    *acc = values.fold_masked(valid, T::zero(), *acc, |sum, v| {
        if skip_nans && v.is_nan() {
            sum
        } else {
            sum + ToPrimitive::to_f64(&v).vortex_expect("float to f64")
        }
    });
}

/// Sum the valid values of an integer slice narrower than 64 bits in chunks of [`SUM_CHUNK`],
/// reading null lanes as zero, and hand each chunk's sum to `add`, which returns `true` on
/// overflow. A chunk cannot overflow its 64-bit sum, as in [`sum_unsigned_all`].
fn sum_chunks_masked<T, S>(values: &[T], valid: &BitBuffer, mut add: impl FnMut(S) -> bool) -> bool
where
    T: NativePType + AsPrimitive<S>,
    S: NativePType + Add<Output = S>,
{
    (0..values.len()).step_by(SUM_CHUNK).any(|start| {
        let end = (start + SUM_CHUNK).min(values.len());
        let chunk_sum = values[start..end].fold_masked(
            &valid.slice(start..end),
            T::zero(),
            S::zero(),
            |sum, v| sum + v.as_(),
        );
        add(chunk_sum)
    })
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::NumericalAggregateOpts;
    use crate::aggregate_fn::fns::sum::Sum;
    use crate::aggregate_fn::fns::sum::sum;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType;
    use crate::expr::stats::Precision;
    use crate::expr::stats::Stat;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn sum_i32() -> VortexResult<()> {
        let arr = PrimitiveArray::new(buffer![1i32, 2, 3, 4], Validity::NonNullable).into_array();
        let result = sum(&arr, &mut SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(10));
        Ok(())
    }

    /// The null rows hold the type's maximum, which the masked kernel must not add, in a full
    /// 64-row word and in the partial word after it.
    #[test]
    fn sum_ignores_values_under_nulls() -> VortexResult<()> {
        let values: Buffer<u8> = (0..100u8)
            .map(|i| if i % 3 == 0 { u8::MAX } else { i })
            .collect();
        let validity = Validity::from_iter((0..100).map(|i| i % 3 != 0));
        let arr = PrimitiveArray::new(values, validity).into_array();
        let expected: u64 = (0..100u64).filter(|i| i % 3 != 0).sum();
        let result = sum(&arr, &mut SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(expected));
        Ok(())
    }

    /// The null rows hold infinities and NaNs, which the masked float kernel must not add, and
    /// the valid NaN is skipped by default.
    #[test]
    fn float_sum_ignores_values_under_nulls() -> VortexResult<()> {
        let values: Buffer<f32> = (0..100u16)
            .map(|i| match i % 3 {
                0 if i % 2 == 0 => f32::INFINITY,
                0 => f32::NAN,
                _ if i == 50 => f32::NAN,
                _ => f32::from(i) - 50.0,
            })
            .collect();
        let validity = Validity::from_iter((0..100).map(|i| i % 3 != 0));
        let arr = PrimitiveArray::new(values, validity).into_array();
        let expected: f64 = (0..100u16)
            .filter(|&i| i % 3 != 0 && i != 50)
            .map(|i| f64::from(i) - 50.0)
            .sum();
        let result = sum(&arr, &mut SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(expected));
        Ok(())
    }

    #[test]
    fn sum_u8() -> VortexResult<()> {
        let arr = PrimitiveArray::new(buffer![10u8, 20, 30], Validity::NonNullable).into_array();
        let result = sum(&arr, &mut SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(60));
        Ok(())
    }

    #[test]
    fn sum_f64() -> VortexResult<()> {
        let arr =
            PrimitiveArray::new(buffer![1.5f64, 2.5, 3.0], Validity::NonNullable).into_array();
        let result = sum(&arr, &mut SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(7.0));
        Ok(())
    }

    #[test]
    fn sum_with_nulls() -> VortexResult<()> {
        let arr = PrimitiveArray::from_option_iter([Some(2i32), None, Some(4)]).into_array();
        let result = sum(&arr, &mut SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(6));
        Ok(())
    }

    #[test]
    fn sum_multiple_null_runs() -> VortexResult<()> {
        // Several disjoint valid runs separated by nulls exercise the per-run fold.
        let arr = PrimitiveArray::from_option_iter([
            Some(1i32),
            Some(2),
            None,
            None,
            Some(3),
            None,
            Some(4),
            Some(5),
            Some(6),
        ])
        .into_array();
        let result = sum(&arr, &mut SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(21));
        Ok(())
    }

    #[test]
    fn sum_all_null() -> VortexResult<()> {
        let arr = PrimitiveArray::from_option_iter([None::<i32>, None, None]).into_array();
        let result = sum(&arr, &mut SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(0));
        Ok(())
    }

    #[test]
    fn sum_all_invalid_float() -> VortexResult<()> {
        let arr = PrimitiveArray::from_option_iter::<f32, _>([None, None, None]).into_array();
        let result = sum(&arr, &mut SESSION.create_execution_ctx())?;
        assert_eq!(result, Scalar::primitive(0f64, Nullable));
        Ok(())
    }

    #[test]
    fn sum_buffer_i32() -> VortexResult<()> {
        let arr = buffer![1, 1, 1, 1].into_array();
        let result = sum(&arr, &mut SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().as_::<i32>(), Some(4));
        Ok(())
    }

    #[test]
    fn sum_buffer_f64() -> VortexResult<()> {
        let arr = buffer![1., 1., 1., 1.].into_array();
        let result = sum(&arr, &mut SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().as_::<f32>(), Some(4.));
        Ok(())
    }

    #[test]
    fn sum_empty_produces_zero() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, NumericalAggregateOpts::default(), dtype)?;
        let result = acc.finish()?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(0));
        Ok(())
    }

    #[test]
    fn sum_empty_f64_produces_zero() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, NumericalAggregateOpts::default(), dtype)?;
        let result = acc.finish()?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(0.0));
        Ok(())
    }

    #[test]
    fn sum_f64_with_nan() -> VortexResult<()> {
        let arr = PrimitiveArray::new(
            buffer![1.0f64, f64::NAN, 2.0, f64::NAN, 3.0],
            Validity::NonNullable,
        )
        .into_array();
        let result = sum(&arr, &mut SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(6.0));
        Ok(())
    }

    #[test]
    fn sum_f32_with_nan() -> VortexResult<()> {
        let arr =
            PrimitiveArray::new(buffer![1.0f32, f32::NAN, 4.0], Validity::NonNullable).into_array();
        let result = sum(&arr, &mut SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(5.0));
        Ok(())
    }

    #[test]
    fn sum_f64_with_nan_and_nulls() -> VortexResult<()> {
        let arr = PrimitiveArray::from_option_iter([Some(1.0f64), None, Some(f64::NAN), Some(3.0)])
            .into_array();
        let result = sum(&arr, &mut SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(4.0));
        Ok(())
    }

    #[test]
    fn sum_all_nan() -> VortexResult<()> {
        let arr =
            PrimitiveArray::new(buffer![f64::NAN, f64::NAN], Validity::NonNullable).into_array();
        let result = sum(&arr, &mut SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(0.0));
        Ok(())
    }

    /// Sum an array with explicit [`NumericalAggregateOpts`] (test-only helper).
    fn sum_with_options(
        arr: &crate::ArrayRef,
        options: NumericalAggregateOpts,
    ) -> VortexResult<Scalar> {
        let mut acc = Accumulator::try_new(Sum, options, arr.dtype().clone())?;
        acc.accumulate(arr, &mut SESSION.create_execution_ctx())?;
        acc.finish()
    }

    #[test]
    fn sum_f64_with_nan_not_skipping() -> VortexResult<()> {
        let arr =
            PrimitiveArray::new(buffer![1.0f64, f64::NAN, 2.0], Validity::NonNullable).into_array();
        let result = sum_with_options(&arr, NumericalAggregateOpts::include_nans())?;
        assert!(result.as_primitive().typed_value::<f64>().unwrap().is_nan());
        Ok(())
    }

    #[test]
    fn sum_f64_without_nan_not_skipping() -> VortexResult<()> {
        let arr =
            PrimitiveArray::new(buffer![1.0f64, 2.0, 3.0], Validity::NonNullable).into_array();
        let result = sum_with_options(&arr, NumericalAggregateOpts::include_nans())?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(6.0));
        Ok(())
    }

    #[test]
    fn sum_not_skipping_with_exact_nan_count_stat() -> VortexResult<()> {
        // The exact NaN count agrees with the batch; including NaNs must poison the sum.
        let arr = PrimitiveArray::new(buffer![1.0f64, 2.0, 3.0, f64::NAN], Validity::NonNullable)
            .into_array();
        arr.statistics()
            .set(Stat::NaNCount, Precision::Exact(ScalarValue::from(1u64)));
        let result = sum_with_options(&arr, NumericalAggregateOpts::include_nans())?;
        assert!(result.as_primitive().typed_value::<f64>().unwrap().is_nan());
        Ok(())
    }

    #[test]
    fn sum_not_skipping_with_exact_sum_when_nan_free() -> VortexResult<()> {
        // With an exact NaNCount of zero, the exact Sum stat is usable as-is.
        let arr =
            PrimitiveArray::new(buffer![1.0f64, 2.0, 3.0], Validity::NonNullable).into_array();
        arr.statistics()
            .set(Stat::NaNCount, Precision::Exact(ScalarValue::from(0u64)));
        arr.statistics()
            .set(Stat::Sum, Precision::Exact(ScalarValue::from(6.0f64)));
        let result = sum_with_options(&arr, NumericalAggregateOpts::include_nans())?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(6.0));
        Ok(())
    }

    #[test]
    fn sum_constant_nan() -> VortexResult<()> {
        let arr = ConstantArray::new(f64::NAN, 4).into_array();
        // NaN constants are skipped by default and poison the sum otherwise.
        let result = sum_with_options(&arr, NumericalAggregateOpts::default())?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(0.0));

        let result = sum_with_options(&arr, NumericalAggregateOpts::include_nans())?;
        assert!(result.as_primitive().typed_value::<f64>().unwrap().is_nan());
        Ok(())
    }

    #[test]
    fn sum_f64_with_infinity() -> VortexResult<()> {
        let batch = PrimitiveArray::new(
            buffer![1.0f64, f64::INFINITY, f64::NEG_INFINITY, 2.0],
            Validity::NonNullable,
        )
        .into_array();
        let acc = sum(&batch, &mut SESSION.create_execution_ctx())?;
        // INFINITY + NEG_INFINITY = NaN, which is treated as saturated
        assert!(acc.as_primitive().typed_value::<f64>().unwrap().is_nan());

        let mut acc = Accumulator::try_new(
            Sum,
            NumericalAggregateOpts::default(),
            DType::Primitive(PType::F64, Nullability::NonNullable),
        )?;
        acc.accumulate(&batch, &mut SESSION.create_execution_ctx())?;
        assert!(acc.is_saturated());
        Ok(())
    }

    #[test]
    fn sum_checked_overflow() -> VortexResult<()> {
        let arr = PrimitiveArray::new(buffer![i64::MAX, 1i64], Validity::NonNullable).into_array();
        let result = sum(&arr, &mut SESSION.create_execution_ctx())?;
        assert!(result.is_null());
        Ok(())
    }

    #[test]
    fn sum_checked_overflow_is_saturated() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, NumericalAggregateOpts::default(), dtype)?;
        assert!(!acc.is_saturated());

        let batch =
            PrimitiveArray::new(buffer![i64::MAX, 1i64], Validity::NonNullable).into_array();
        acc.accumulate(&batch, &mut SESSION.create_execution_ctx())?;
        assert!(acc.is_saturated());

        // finish resets state, clearing saturation
        drop(acc.finish()?);
        assert!(!acc.is_saturated());
        Ok(())
    }
}
