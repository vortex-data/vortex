// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use num_traits::ToPrimitive;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::AllOr;

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

pub(super) fn accumulate_primitive(
    inner: &mut SumState,
    p: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
    skip_nans: bool,
) -> VortexResult<bool> {
    let mask = p.as_ref().validity()?.execute_mask(p.as_ref().len(), ctx)?;
    match mask.slices() {
        AllOr::None => Ok(false),
        AllOr::All => accumulate_primitive_all(inner, p, skip_nans),
        AllOr::Some(slices) => accumulate_primitive_valid(inner, p, slices, skip_nans),
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
        SumState::Decimal { .. } => vortex_panic!("decimal sum state with primitive input"),
    }
}

/// Sum the values of a float slice into an `f64` accumulator. When `skip_nans` is set, NaN values
/// are skipped to match the scalar `sum` semantics; otherwise any NaN poisons the accumulator to
/// NaN. Floats cannot overflow the accumulator, so this never reports saturation.
pub(super) fn sum_float_all<T: NativePType>(acc: &mut f64, slice: &[T], skip_nans: bool) {
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
pub(super) fn sum_unsigned_all<T>(acc: &mut u64, slice: &[T]) -> bool
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
pub(super) fn sum_signed_all<T>(acc: &mut i64, slice: &[T]) -> bool
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

/// Sum the valid elements, described as contiguous `[start, end)` runs of set validity bits. Each
/// run is a slice of fully-valid values, so it reuses the same vectorized reduction as the
/// all-valid path instead of a per-element validity branch.
fn accumulate_primitive_valid(
    inner: &mut SumState,
    p: &PrimitiveArray,
    slices: &[(usize, usize)],
    skip_nans: bool,
) -> VortexResult<bool> {
    match inner {
        SumState::Unsigned(acc) => match_each_native_ptype!(p.ptype(),
            unsigned: |T| {
                let values = p.as_slice::<T>();
                for &(start, end) in slices {
                    if sum_unsigned_all(acc, &values[start..end]) {
                        return Ok(true);
                    }
                }
                Ok(false)
            },
            signed: |_T| { vortex_panic!("unsigned sum state with signed input") },
            floating: |_T| { vortex_panic!("unsigned sum state with float input") }
        ),
        SumState::Signed(acc) => match_each_native_ptype!(p.ptype(),
            unsigned: |_T| { vortex_panic!("signed sum state with unsigned input") },
            signed: |T| {
                let values = p.as_slice::<T>();
                for &(start, end) in slices {
                    if sum_signed_all(acc, &values[start..end]) {
                        return Ok(true);
                    }
                }
                Ok(false)
            },
            floating: |_T| { vortex_panic!("signed sum state with float input") }
        ),
        SumState::Float(acc) => match_each_native_ptype!(p.ptype(),
            unsigned: |_T| { vortex_panic!("float sum state with unsigned input") },
            signed: |_T| { vortex_panic!("float sum state with signed input") },
            floating: |T| {
                let values = p.as_slice::<T>();
                for &(start, end) in slices {
                    sum_float_all(acc, &values[start..end], skip_nans);
                }
                Ok(false)
            }
        ),
        SumState::Decimal { .. } => vortex_panic!("decimal sum state with primitive input"),
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
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

    #[test]
    fn sum_i32() -> VortexResult<()> {
        let arr = PrimitiveArray::new(buffer![1i32, 2, 3, 4], Validity::NonNullable).into_array();
        let result = sum(&arr, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(10));
        Ok(())
    }

    #[test]
    fn sum_u8() -> VortexResult<()> {
        let arr = PrimitiveArray::new(buffer![10u8, 20, 30], Validity::NonNullable).into_array();
        let result = sum(&arr, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(60));
        Ok(())
    }

    #[test]
    fn sum_f64() -> VortexResult<()> {
        let arr =
            PrimitiveArray::new(buffer![1.5f64, 2.5, 3.0], Validity::NonNullable).into_array();
        let result = sum(&arr, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(7.0));
        Ok(())
    }

    #[test]
    fn sum_with_nulls() -> VortexResult<()> {
        let arr = PrimitiveArray::from_option_iter([Some(2i32), None, Some(4)]).into_array();
        let result = sum(&arr, &mut LEGACY_SESSION.create_execution_ctx())?;
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
        let result = sum(&arr, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(21));
        Ok(())
    }

    #[test]
    fn sum_all_null() -> VortexResult<()> {
        let arr = PrimitiveArray::from_option_iter([None::<i32>, None, None]).into_array();
        let result = sum(&arr, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(0));
        Ok(())
    }

    #[test]
    fn sum_all_invalid_float() -> VortexResult<()> {
        let arr = PrimitiveArray::from_option_iter::<f32, _>([None, None, None]).into_array();
        let result = sum(&arr, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(result, Scalar::primitive(0f64, Nullable));
        Ok(())
    }

    #[test]
    fn sum_buffer_i32() -> VortexResult<()> {
        let arr = buffer![1, 1, 1, 1].into_array();
        let result = sum(&arr, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().as_::<i32>(), Some(4));
        Ok(())
    }

    #[test]
    fn sum_buffer_f64() -> VortexResult<()> {
        let arr = buffer![1., 1., 1., 1.].into_array();
        let result = sum(&arr, &mut LEGACY_SESSION.create_execution_ctx())?;
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
        let result = sum(&arr, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(6.0));
        Ok(())
    }

    #[test]
    fn sum_f32_with_nan() -> VortexResult<()> {
        let arr =
            PrimitiveArray::new(buffer![1.0f32, f32::NAN, 4.0], Validity::NonNullable).into_array();
        let result = sum(&arr, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(5.0));
        Ok(())
    }

    #[test]
    fn sum_f64_with_nan_and_nulls() -> VortexResult<()> {
        let arr = PrimitiveArray::from_option_iter([Some(1.0f64), None, Some(f64::NAN), Some(3.0)])
            .into_array();
        let result = sum(&arr, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(4.0));
        Ok(())
    }

    #[test]
    fn sum_all_nan() -> VortexResult<()> {
        let arr =
            PrimitiveArray::new(buffer![f64::NAN, f64::NAN], Validity::NonNullable).into_array();
        let result = sum(&arr, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(0.0));
        Ok(())
    }

    /// Sum an array with explicit [`NumericalAggregateOpts`] (test-only helper).
    fn sum_with_options(
        arr: &crate::ArrayRef,
        options: NumericalAggregateOpts,
    ) -> VortexResult<Scalar> {
        let mut acc = Accumulator::try_new(Sum, options, arr.dtype().clone())?;
        acc.accumulate(arr, &mut LEGACY_SESSION.create_execution_ctx())?;
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
    fn sum_not_skipping_shortcircuits_on_exact_nan_count_stat() -> VortexResult<()> {
        // The array has no NaNs; a planted exact NaNCount stat proves the NaN poisoning came
        // from the stat rather than a scan.
        let arr =
            PrimitiveArray::new(buffer![1.0f64, 2.0, 3.0], Validity::NonNullable).into_array();
        arr.statistics()
            .set(Stat::NaNCount, Precision::Exact(ScalarValue::from(1u64)));
        let result = sum_with_options(&arr, NumericalAggregateOpts::include_nans())?;
        assert!(result.as_primitive().typed_value::<f64>().unwrap().is_nan());
        Ok(())
    }

    #[test]
    fn sum_not_skipping_uses_cached_sum_when_nan_free() -> VortexResult<()> {
        // With an exact NaNCount of zero, the planted exact Sum stat is usable as-is.
        let arr =
            PrimitiveArray::new(buffer![1.0f64, 2.0, 3.0], Validity::NonNullable).into_array();
        arr.statistics()
            .set(Stat::NaNCount, Precision::Exact(ScalarValue::from(0u64)));
        arr.statistics()
            .set(Stat::Sum, Precision::Exact(ScalarValue::from(42.0f64)));
        let result = sum_with_options(&arr, NumericalAggregateOpts::include_nans())?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(42.0));
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
        let acc = sum(&batch, &mut LEGACY_SESSION.create_execution_ctx())?;
        // INFINITY + NEG_INFINITY = NaN, which is treated as saturated
        assert!(acc.as_primitive().typed_value::<f64>().unwrap().is_nan());

        let mut acc = Accumulator::try_new(
            Sum,
            NumericalAggregateOpts::default(),
            DType::Primitive(PType::F64, Nullability::NonNullable),
        )?;
        acc.accumulate(&batch, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert!(acc.is_saturated());
        Ok(())
    }

    #[test]
    fn sum_checked_overflow() -> VortexResult<()> {
        let arr = PrimitiveArray::new(buffer![i64::MAX, 1i64], Validity::NonNullable).into_array();
        let result = sum(&arr, &mut LEGACY_SESSION.create_execution_ctx())?;
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
        acc.accumulate(&batch, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert!(acc.is_saturated());

        // finish resets state, clearing saturation
        drop(acc.finish()?);
        assert!(!acc.is_saturated());
        Ok(())
    }
}
