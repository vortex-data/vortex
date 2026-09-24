// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow's decimal arithmetic rules, and their evaluation over [`DecimalValue`].
//!
//! Arrow derives the result of `(p1, s1) op (p2, s2)` as:
//!
//! | operator | result precision                    | result scale      |
//! | -------- | ----------------------------------- | ----------------- |
//! | Add, Sub | `max(p1 - s1, p2 - s2) + s + 1`     | `s = max(s1, s2)` |
//! | Mul      | `p1 + p2 + 1`                       | `s1 + s2`         |
//! | Div      | `p1 - s1 + s2 + s`                  | `s = s1 + 4`      |
//!
//! Precision saturates at [`MAX_PRECISION`]. Add and Sub first align both stored integers to the
//! result scale. Mul needs no alignment: the product of the stored integers already sits at the
//! summed scale. Div scales the dividend by `10^(s - s1 + s2)` up front, or the divisor for a
//! negative exponent, and truncates toward zero, which is what Arrow does.
//!
//! Alignment can outgrow the widest native width: scaling an operand by `10^e` overflows `i256`
//! once `p + e` passes [`MAX_PRECISION`], even for a result that would have fit the result
//! precision. That is reported as an overflow. The array kernels derive their working width the
//! same way, so both paths agree on which operations are representable.

use num_traits::CheckedAdd;
use num_traits::CheckedDiv;
use num_traits::CheckedMul;
use num_traits::CheckedSub;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::dtype::BigCast;
use crate::dtype::DecimalDType;
use crate::dtype::MAX_PRECISION;
use crate::dtype::MAX_SCALE;
use crate::dtype::NativeDecimalType;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalValue;
use crate::scalar::NumericOperator;

/// Derive the result decimal dtype of a numeric operation over two operands of `input`.
///
/// # Errors
///
/// See [`decimal_binary_result_dtype`].
#[cfg(any(test, feature = "_test-harness"))]
pub(crate) fn decimal_numeric_result_dtype(
    input: DecimalDType,
    op: NumericOperator,
) -> VortexResult<DecimalDType> {
    decimal_binary_result_dtype(input, input, op)
}

/// Derive the result decimal dtype of `lhs op rhs`, following Arrow's rules.
///
/// # Errors
///
/// Returns an error if the operation has no valid result type: a Mul whose summed scale is
/// unrepresentable — above [`MAX_SCALE`], or below the `i8::MIN` floor of the scale field — or a
/// Div whose precision would fall outside the legal range.
pub(crate) fn decimal_binary_result_dtype(
    lhs: DecimalDType,
    rhs: DecimalDType,
    op: NumericOperator,
) -> VortexResult<DecimalDType> {
    let (p1, s1) = widen(lhs);
    let (p2, s2) = widen(rhs);

    match op {
        NumericOperator::Add | NumericOperator::Sub => {
            // Keep the wider integral part and the finer scale. One carry digit suffices:
            // 2 * (10^p - 1) < 10^(p + 1).
            let result_scale = lhs.scale().max(rhs.scale());
            let result_precision =
                (p1 - s1).max(p2 - s2) + <i16 as From<i8>>::from(result_scale) + 1;
            DecimalDType::try_new(clamp_precision(result_precision), result_scale)
        }
        NumericOperator::Mul => {
            // Sum in i16 so a very negative sum cannot wrap or saturate into a legal-looking i8
            // scale. The SQL standard rejects a product whose scale cannot be represented rather
            // than rounding it away.
            let result_scale = s1 + s2;
            let Some(result_scale) = i8::try_from(result_scale)
                .ok()
                .filter(|scale| *scale <= MAX_SCALE)
            else {
                vortex_bail!(
                    "output scale {result_scale} of {lhs} {op} {rhs} is outside the \
                     representable scale range of {} to {MAX_SCALE}",
                    i8::MIN
                );
            };
            DecimalDType::try_new(clamp_precision(p1 + p2 + 1), result_scale)
        }
        NumericOperator::Div => {
            // Arrow follows Postgres and MySQL in adding a fixed four fractional digits.
            let result_scale = lhs.scale().saturating_add(4).min(MAX_SCALE);
            let result_precision = p1 - s1 + s2 + <i16 as From<i8>>::from(result_scale);
            if result_precision < 1 {
                vortex_bail!(
                    InvalidArgument:
                    "decimal division result precision {result_precision} is invalid"
                );
            }
            DecimalDType::try_new(clamp_precision(result_precision), result_scale)
        }
    }
}

fn widen(dtype: DecimalDType) -> (i16, i16) {
    (
        <i16 as From<u8>>::from(dtype.precision()),
        <i16 as From<i8>>::from(dtype.scale()),
    )
}

fn clamp_precision(precision: i16) -> u8 {
    u8::try_from(precision.clamp(0, <i16 as From<u8>>::from(MAX_PRECISION)))
        .vortex_expect("clamped precision fits u8")
}

/// Powers of ten `(lhs, rhs)` that align the stored values of `lhs op rhs` before `op` applies.
///
/// Add and Sub scale each operand up to the result scale. Div scales the dividend by
/// `10^(result_scale - s1 + s2)`, or the divisor for a negative exponent. Mul needs none.
pub(crate) fn decimal_align_exponents(
    lhs: DecimalDType,
    rhs: DecimalDType,
    result: DecimalDType,
    op: NumericOperator,
) -> (u32, u32) {
    let result_scale = <i16 as From<i8>>::from(result.scale());
    let (s1, s2) = (
        <i16 as From<i8>>::from(lhs.scale()),
        <i16 as From<i8>>::from(rhs.scale()),
    );
    let (lhs_exp, rhs_exp) = match op {
        NumericOperator::Add | NumericOperator::Sub => (result_scale - s1, result_scale - s2),
        NumericOperator::Mul => (0, 0),
        NumericOperator::Div => {
            let exp = result_scale - s1 + s2;
            if exp >= 0 { (exp, 0) } else { (0, -exp) }
        }
    };
    (
        <u32 as From<u16>>::from(lhs_exp.unsigned_abs()),
        <u32 as From<u16>>::from(rhs_exp.unsigned_abs()),
    )
}

/// Apply `op` to stored values of `lhs_dtype` and `rhs_dtype`, returning the result stored in
/// `result`'s width.
///
/// Returns `None` if the result overflows `result`'s precision, or if `op` is a division by zero.
pub(crate) fn checked_decimal_numeric(
    lhs: DecimalValue,
    rhs: DecimalValue,
    lhs_dtype: DecimalDType,
    rhs_dtype: DecimalDType,
    result: DecimalDType,
    op: NumericOperator,
) -> Option<DecimalValue> {
    let work = decimal_numeric_work_dtype(lhs_dtype, rhs_dtype, result, op);
    let (lhs_exp, rhs_exp) = decimal_align_exponents(lhs_dtype, rhs_dtype, result, op);
    match_each_decimal_value_type!(DecimalType::smallest_decimal_value_type(&work), |W| {
        let lhs = aligned_value::<W>(lhs, lhs_exp)?;
        let rhs = aligned_value::<W>(rhs, rhs_exp)?;
        let value = checked_at_width::<W>(lhs, rhs, op)?;
        DecimalValue::from(value).normalize(result)
    })
}

/// Pick a native width wide enough to hold every intermediate of an in-precision operation.
///
/// Each operand needs room for its precision plus its alignment exponent, and the result needs
/// room for its own precision. For Mul, and for Add and Sub over one scale, that is the result
/// precision.
pub(crate) fn decimal_numeric_work_dtype(
    lhs: DecimalDType,
    rhs: DecimalDType,
    result: DecimalDType,
    op: NumericOperator,
) -> DecimalDType {
    let (lhs_exp, rhs_exp) = decimal_align_exponents(lhs, rhs, result, op);
    let aligned = |dtype: DecimalDType, exp: u32| {
        <u32 as From<u8>>::from(dtype.precision()).saturating_add(exp)
    };
    let precision = aligned(lhs, lhs_exp)
        .max(aligned(rhs, rhs_exp))
        .max(<u32 as From<u8>>::from(result.precision()))
        .min(<u32 as From<u8>>::from(MAX_PRECISION));
    DecimalDType::new(
        u8::try_from(precision).vortex_expect("precision is at most MAX_PRECISION"),
        0,
    )
}

/// `value` at width `W`, scaled by `10^exp`, or `None` if that is not representable there.
fn aligned_value<W>(value: DecimalValue, exp: u32) -> Option<W>
where
    W: NativeDecimalType + CheckedAdd + CheckedMul,
{
    value
        .cast::<W>()?
        .checked_mul(&decimal_scale_factor::<W>(exp)?)
}

fn checked_at_width<W>(lhs: W, rhs: W, op: NumericOperator) -> Option<W>
where
    W: NativeDecimalType + CheckedAdd + CheckedSub + CheckedMul + CheckedDiv,
{
    match op {
        NumericOperator::Add => lhs.checked_add(&rhs),
        NumericOperator::Sub => lhs.checked_sub(&rhs),
        // The result scale is the sum of the operand scales, so the raw product is already
        // correctly scaled.
        NumericOperator::Mul => lhs.checked_mul(&rhs),
        NumericOperator::Div => lhs.checked_div(&rhs),
    }
}

/// `10^exp` at width `W`, or `None` if it is not representable there.
fn decimal_scale_factor<W>(exp: u32) -> Option<W>
where
    W: NativeDecimalType + CheckedAdd,
{
    let max = *W::MAX_BY_PRECISION.get(usize::try_from(exp).ok()?)?;
    max.checked_add(&<W as BigCast>::from(1_i8)?)
}
