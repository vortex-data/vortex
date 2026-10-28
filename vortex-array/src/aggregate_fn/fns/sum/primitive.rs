// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Add;
use std::ops::BitOr;

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

/// Partial sums of a sum of integers narrower than 64 bits, each widened to a 64-bit lane.
const NARROW_SUM_LANES: usize = 16;

/// Partial sums of a float sum. Sixteen `f64` lanes fill eight NEON, four AVX2 or two AVX-512
/// registers, enough to hide the latency of a vector add on each.
const FLOAT_SUM_LANES: usize = 16;

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
    sum_float(acc, slice, None, skip_nans);
}

/// Sum the values of a float slice, or only its `valid` values, into an `f64` accumulator.
///
/// The values are added in [`FLOAT_SUM_LANES`] independent partial sums, so the sum vectorizes
/// although float addition does not reassociate. The result can differ from an in-order sum only
/// in rounding.
///
/// With `skip_nans`, the values are first summed as they are: a sum that is not NaN saw no NaN.
/// Only a NaN sum, which a NaN or infinities of both signs make, is summed again reading NaNs as
/// zero. A select per lane to skip NaNs on top of the select for null lanes leaves the masked
/// fold scalar on NEON.
fn sum_float<T: NativePType>(acc: &mut f64, values: &[T], valid: Option<&BitBuffer>, skip_nans: bool) {
    let sum = fold_float(values, valid, |v| v);
    *acc += if skip_nans && sum.is_nan() {
        fold_float(values, valid, non_nan)
    } else {
        sum
    };
}

/// Sum `read` of each value, or of each `valid` value, into [`FLOAT_SUM_LANES`] partial sums.
#[inline]
fn fold_float<T: NativePType>(
    values: &[T],
    valid: Option<&BitBuffer>,
    read: impl Fn(f64) -> f64,
) -> f64 {
    let add = |sum: f64, v: T| sum + read(ToPrimitive::to_f64(&v).vortex_expect("float to f64"));
    match valid {
        None => values.fold_lanes::<FLOAT_SUM_LANES, _, _, _>(0.0, add, |a, b| a + b),
        Some(valid) => values.fold_lanes_masked::<FLOAT_SUM_LANES, _, _, _>(
            valid,
            T::zero(),
            0.0,
            add,
            |a, b| a + b,
        ),
    }
}

/// `v`, or zero when `v` is NaN.
#[inline]
fn non_nan(v: f64) -> f64 {
    if v.is_nan() { 0.0 } else { v }
}

/// Sum all values into a `u64` accumulator. For types narrower than 64 bits, values are summed in
/// chunks of [`SUM_CHUNK`] by [`narrow_sum`] with a single checked add per chunk. A chunk of `u64`s could itself overflow, so `u64` input is summed by
/// [`sum_wide`]. Returns `true` on overflow.
pub(crate) fn sum_unsigned_all<T>(acc: &mut u64, slice: &[T]) -> bool
where
    T: NativePType + AsPrimitive<u64>,
{
    if T::PTYPE == PType::U64 {
        return sum_wide(acc, slice, None);
    }
    for chunk in slice.chunks(SUM_CHUNK) {
        let chunk_sum = narrow_sum::<T, u64>(chunk);
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
        return sum_wide(acc, slice, None);
    }
    for chunk in slice.chunks(SUM_CHUNK) {
        let chunk_sum = narrow_sum::<T, i64>(chunk);
        if checked_add_i64(acc, chunk_sum) {
            return true;
        }
    }
    false
}

/// Sum the valid elements.
///
/// Every kernel reads null lanes as zero, so the sum vectorizes the same way as the all-valid path
/// however the valid values are spread.
fn accumulate_primitive_valid(
    inner: &mut SumState,
    p: &PrimitiveArray,
    valid: &MaskValues,
    skip_nans: bool,
) -> VortexResult<bool> {
    match inner {
        SumState::Unsigned(acc) => match_each_native_ptype!(p.ptype(),
            unsigned: |T| {
                let values = p.as_slice::<T>();
                if T::PTYPE == PType::U64 {
                    return Ok(sum_wide(acc, values, Some(valid.bit_buffer())));
                }
                Ok(sum_masked(values, valid.bit_buffer(), |chunk_sum| {
                    checked_add_u64(acc, chunk_sum)
                }))
            },
            signed: |_T| { vortex_panic!("unsigned sum state with signed input") },
            floating: |_T| { vortex_panic!("unsigned sum state with float input") }
        ),
        SumState::Signed(acc) => match_each_native_ptype!(p.ptype(),
            unsigned: |_T| { vortex_panic!("signed sum state with unsigned input") },
            signed: |T| {
                let values = p.as_slice::<T>();
                if T::PTYPE == PType::I64 {
                    return Ok(sum_wide(acc, values, Some(valid.bit_buffer())));
                }
                Ok(sum_masked(values, valid.bit_buffer(), |chunk_sum| {
                    checked_add_i64(acc, chunk_sum)
                }))
            },
            floating: |_T| { vortex_panic!("signed sum state with float input") }
        ),
        SumState::Float(acc) => match_each_native_ptype!(p.ptype(),
            unsigned: |_T| { vortex_panic!("float sum state with unsigned input") },
            signed: |_T| { vortex_panic!("float sum state with signed input") },
            floating: |T| {
                sum_float(acc, p.as_slice::<T>(), Some(valid.bit_buffer()), skip_nans);
                Ok(false)
            }
        ),
        SumState::Decimal(_) => {
            vortex_panic!("decimal sum state with primitive input")
        }
    }
}

/// Sum the valid values of an integer slice narrower than 64 bits in chunks of [`SUM_CHUNK`],
/// reading null lanes as zero, and hand each chunk's sum to `add`, which returns `true` on
/// overflow. A chunk cannot overflow its 64-bit sum, as in [`sum_unsigned_all`].
fn sum_masked<T, S>(values: &[T], valid: &BitBuffer, mut add: impl FnMut(S) -> bool) -> bool
where
    T: NativePType + AsPrimitive<S>,
    S: NativePType + Add<Output = S>,
{
    (0..values.len()).step_by(SUM_CHUNK).any(|start| {
        let end = (start + SUM_CHUNK).min(values.len());
        let chunk_sum = values[start..end].fold_lanes_masked::<NARROW_SUM_LANES, _, _, _>(
            &valid.slice(start..end),
            T::zero(),
            S::zero(),
            |sum, v| sum + v.as_(),
            |a, b| a + b,
        );
        add(chunk_sum)
    })
}

/// Sum a chunk of at most [`SUM_CHUNK`] integers narrower than 64 bits, which cannot overflow `S`.
///
/// The compiler would vectorize an in-order sum on its own, but into a loop of two adds that runs
/// at the mercy of where it lands in the instruction cache: on x86-64 without SSE4.1 the same loop
/// ran 30% slower once it straddled a 64-byte boundary. [`NARROW_SUM_LANES`] partial sums make a
/// loop body large enough not to care, and are as fast or faster on every target, with one
/// exception: x86-64 without SSE4.1 has no vector sign extension, and widening signed values into
/// that many lanes costs more than the in-order loop does.
fn narrow_sum<T, S>(chunk: &[T]) -> S
where
    T: NativePType + AsPrimitive<S>,
    S: NativePType + Add<Output = S>,
{
    if NO_SIGN_EXTENSION && T::PTYPE.is_signed_int() {
        return chunk.iter().map(|&v| v.as_()).fold(S::zero(), |a, b| a + b);
    }
    chunk.fold_lanes::<NARROW_SUM_LANES, _, _, _>(S::zero(), |sum, v| sum + v.as_(), |a, b| a + b)
}

/// Whether the target lacks a vector sign extension of narrow integers to 64 bits.
const NO_SIGN_EXTENSION: bool = cfg!(all(target_arch = "x86_64", not(target_feature = "sse4.1")));

/// Sum 64-bit integers, or only the `valid` ones, into `acc`. Returns `true` on overflow.
///
/// A checked add per element does not vectorize, so the values are added with wrapping, and
/// evidence of each value's magnitude is ORed together alongside. Both reassociate, so the
/// compiler vectorizes the fold in lane order with the width and interleave it picks, which beats
/// independent per-lane partial sums on every target. If every value is small enough that
/// `values.len()` of them cannot overflow, which holds for most data, the sum is exact. Otherwise the values are summed again in
/// [`sum_exact`]. Either way the result overflows only if the exact sum does not fit, whatever the
/// order of the values.
fn sum_wide<T, S>(acc: &mut S, values: &[T], valid: Option<&BitBuffer>) -> bool
where
    T: NativePType + AsPrimitive<S>,
    S: WrappingSum,
{
    let Some(bound) = S::Bound::for_len(values.len()) else {
        return false;
    };

    let add = |(sum, evidence): (S, S), v: T| {
        let v = v.as_();
        (sum.wrapping_add(v), evidence | bound.evidence(v))
    };
    let zero = (S::zero(), S::zero());

    let (sum, evidence) = match valid {
        None => values.iter().fold(zero, |acc, &v| add(acc, v)),
        Some(valid) => values.fold_masked(valid, T::zero(), zero, add),
    };

    if !bound.holds(evidence) {
        return sum_exact(acc, values, valid);
    }
    S::checked_add_into(acc, sum)
}

/// Sum the values, or only the `valid` ones, into `acc` in 128-bit arithmetic, which no slice of
/// 64-bit integers can overflow. Returns `true` if the result does not fit `S`.
#[cold]
#[inline(never)]
fn sum_exact<T, S>(acc: &mut S, values: &[T], valid: Option<&BitBuffer>) -> bool
where
    T: NativePType,
    S: WrappingSum,
{
    let sum: i128 = match valid {
        None => values.iter().map(|&v| to_i128(v)).sum(),
        Some(valid) => valid.set_indices().map(|i| to_i128(values[i])).sum(),
    };

    match S::from(to_i128(*acc) + sum) {
        Some(total) => {
            *acc = total;
            false
        }
        None => true,
    }
}

fn to_i128<T: NativePType>(v: T) -> i128 {
    v.to_i128().vortex_expect("64-bit integers fit i128")
}

/// A 64-bit sum accumulator that adds with wrapping, with a bound on its values under which a
/// wrapping sum cannot overflow.
trait WrappingSum: NativePType + BitOr<Output = Self> {
    /// The magnitude bound for this type.
    type Bound: SumBound<Self>;

    /// Add with wrapping.
    fn wrapping_add(self, v: Self) -> Self;

    /// Checked add of `v` into `acc`, returning `true` on overflow.
    fn checked_add_into(acc: &mut Self, v: Self) -> bool;
}

/// A bound on the magnitude of the values of a sum, tested with one add and one OR per value.
trait SumBound<S>: Copy {
    /// The bound under which `len` values cannot overflow their sum, or `None` if `len` is zero.
    fn for_len(len: usize) -> Option<Self>;

    /// Evidence of the magnitude of `v`, to OR together over all values.
    fn evidence(self, v: S) -> S;

    /// Whether the ORed evidence shows every value is within the bound.
    fn holds(self, evidence: S) -> bool;
}

/// Signed values within `[-2^bits, 2^bits)`, where `len` of them sum to less than `2^63` in
/// magnitude. A value is within the bound iff adding `2^bits` leaves it in `[0, 2^(bits + 1))`.
#[derive(Clone, Copy)]
struct SignedBound {
    bits: u32,
}

impl SumBound<i64> for SignedBound {
    fn for_len(len: usize) -> Option<Self> {
        (len != 0).then(|| Self {
            bits: 63 - (usize::BITS - len.leading_zeros()),
        })
    }

    #[inline]
    fn evidence(self, v: i64) -> i64 {
        v.wrapping_add(1 << self.bits)
    }

    fn holds(self, evidence: i64) -> bool {
        evidence >> (self.bits + 1) == 0
    }
}

/// Unsigned values below `2^bits`, where `len` of them sum to less than `2^64`.
#[derive(Clone, Copy)]
struct UnsignedBound {
    bits: u32,
}

impl SumBound<u64> for UnsignedBound {
    fn for_len(len: usize) -> Option<Self> {
        (len != 0).then(|| Self {
            bits: 64 - (usize::BITS - len.leading_zeros()),
        })
    }

    #[inline]
    fn evidence(self, v: u64) -> u64 {
        v
    }

    fn holds(self, evidence: u64) -> bool {
        evidence >> self.bits == 0
    }
}

impl WrappingSum for i64 {
    type Bound = SignedBound;

    #[inline]
    fn wrapping_add(self, v: i64) -> i64 {
        i64::wrapping_add(self, v)
    }

    fn checked_add_into(acc: &mut i64, v: i64) -> bool {
        checked_add_i64(acc, v)
    }
}

impl WrappingSum for u64 {
    type Bound = UnsignedBound;

    #[inline]
    fn wrapping_add(self, v: u64) -> u64 {
        u64::wrapping_add(self, v)
    }

    fn checked_add_into(acc: &mut u64, v: u64) -> bool {
        checked_add_u64(acc, v)
    }
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

    /// Values past the magnitude bound take the exact sum, which overflows only if the total does
    /// not fit, however the values are ordered.
    #[test]
    fn sum_i64_near_the_limits_is_exact() -> VortexResult<()> {
        let fits =
            PrimitiveArray::new(buffer![i64::MAX, 1, i64::MIN, -1, 5], Validity::NonNullable)
                .into_array();
        let overflows =
            PrimitiveArray::new(buffer![i64::MAX - 1, 1, 1, 0], Validity::NonNullable).into_array();
        let underflows =
            PrimitiveArray::new(buffer![i64::MIN, 0, -1], Validity::NonNullable).into_array();

        let ctx = &mut SESSION.create_execution_ctx();
        assert_eq!(sum(&fits, ctx)?.as_primitive().typed_value::<i64>(), Some(4));
        assert!(sum(&overflows, ctx)?.is_null());
        assert!(sum(&underflows, ctx)?.is_null());
        Ok(())
    }

    /// The null rows hold values that would overflow, in a full 64-row word and in the partial
    /// word after it, on both the bounded and the exact path.
    #[test]
    fn sum_64_bit_ignores_values_under_nulls() -> VortexResult<()> {
        let validity = || Validity::from_iter((0..100).map(|i| i % 3 != 0));
        let unsigned: Buffer<u64> = (0..100u64)
            .map(|i| if i % 3 == 0 { u64::MAX } else { i })
            .collect();
        let signed: Buffer<i64> = (0..100i64)
            .map(|i| match i % 6 {
                0 => i64::MIN,
                3 => i64::MAX,
                _ => -i,
            })
            .collect();
        let large: Buffer<i64> = (0..100i64)
            .map(|i| if i % 3 == 0 { i64::MAX } else { i64::MAX / 100 })
            .collect();

        let ctx = &mut SESSION.create_execution_ctx();
        let unsigned = sum(&PrimitiveArray::new(unsigned, validity()).into_array(), ctx)?;
        let signed = sum(&PrimitiveArray::new(signed, validity()).into_array(), ctx)?;
        let large = sum(&PrimitiveArray::new(large, validity()).into_array(), ctx)?;

        let valid = (0..100i64).filter(|i| i % 3 != 0);
        assert_eq!(
            unsigned.as_primitive().typed_value::<u64>(),
            Some(valid.clone().map(i64::cast_unsigned).sum())
        );
        assert_eq!(
            signed.as_primitive().typed_value::<i64>(),
            Some(-valid.clone().sum::<i64>())
        );
        assert_eq!(
            large.as_primitive().typed_value::<i64>(),
            Some(valid.count() as i64 * (i64::MAX / 100))
        );
        Ok(())
    }

    /// NaNs under nulls are never read, and valid NaNs are skipped or poison the sum as asked.
    #[test]
    fn sum_masked_float_nans() -> VortexResult<()> {
        let values: Buffer<f64> = (0..100)
            .map(|i| match i % 5 {
                0 => f64::NAN,
                4 if i < 50 => f64::NAN,
                _ => f64::from(i),
            })
            .collect();
        let under_nulls = PrimitiveArray::new(
            values.clone(),
            Validity::from_iter((0..100).map(|i| i % 5 != 0 && !(i % 5 == 4 && i < 50))),
        )
        .into_array();
        let valid_nans =
            PrimitiveArray::new(values, Validity::from_iter((0..100).map(|i| i % 5 != 0)))
                .into_array();

        let expected: f64 = (0..100)
            .filter(|&i| i % 5 != 0 && !(i % 5 == 4 && i < 50))
            .map(f64::from)
            .sum();
        assert_eq!(
            sum_with_options(&under_nulls, NumericalAggregateOpts::include_nans())?
                .as_primitive()
                .typed_value::<f64>(),
            Some(expected)
        );
        assert_eq!(
            sum_with_options(&valid_nans, NumericalAggregateOpts::default())?
                .as_primitive()
                .typed_value::<f64>(),
            Some(expected)
        );
        assert!(
            sum_with_options(&valid_nans, NumericalAggregateOpts::include_nans())?
                .as_primitive()
                .typed_value::<f64>()
                .is_some_and(f64::is_nan)
        );
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
