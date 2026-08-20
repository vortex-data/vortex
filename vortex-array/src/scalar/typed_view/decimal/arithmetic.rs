// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow's decimal arithmetic rules, and their evaluation over [`DecimalValue`].
//!
//! Vortex coerces both operands of a decimal arithmetic expression to a single [`DecimalDType`],
//! so Arrow's general `(p1, s1) op (p2, s2)` formulas collapse to a function of one input type:
//!
//! | operator | result precision | result scale |
//! | -------- | ---------------- | ------------ |
//! | Add, Sub | `p + 1`          | `s`          |
//! | Mul      | `2p + 1`         | `2s`         |
//! | Div      | `p + s + 4`      | `s + 4`      |
//!
//! Precision saturates at [`MAX_PRECISION`]. Because Mul widens the scale, the product of the
//! stored integers already sits at the result scale and no rounding is needed. Div instead scales
//! the dividend up front and truncates toward zero, which is what Arrow does.
//!
//! Div is the one operator whose intermediate can outgrow the widest native width: scaling the
//! dividend by `10^result_scale` overflows `i256` once `p + result_scale` passes
//! [`MAX_PRECISION`], even for a quotient that would have fit the result precision. That is
//! reported as an overflow. The array kernels derive their working width the same way, so both
//! paths agree on which operations are representable.

use num_traits::CheckedAdd;
use num_traits::CheckedDiv;
use num_traits::CheckedMul;
use num_traits::CheckedSub;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

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
/// Returns an error if the operation has no valid result type: a Mul whose doubled scale is
/// unrepresentable — above [`MAX_SCALE`], or below the `i8::MIN` floor of the scale field — or a
/// Div whose precision would fall outside the legal range.
pub(crate) fn decimal_numeric_result_dtype(
    input: DecimalDType,
    op: NumericOperator,
) -> VortexResult<DecimalDType> {
    match op {
        NumericOperator::Add | NumericOperator::Sub => {
            // A p-digit Add/Sub result needs at most one carry digit:
            // 2 * (10^p - 1) < 10^(p + 1).
            Ok(DecimalDType::new(
                input.precision().saturating_add(1).min(MAX_PRECISION),
                input.scale(),
            ))
        }
        NumericOperator::Mul => {
            // Doubling the scale in i8 would saturate a very negative sum into a legal-looking
            // scale, so widen first. The SQL standard rejects a product whose scale cannot be
            // represented rather than rounding it away.
            let result_scale = <i16 as From<i8>>::from(input.scale()) * 2;
            let Some(result_scale) = i8::try_from(result_scale)
                .ok()
                .filter(|scale| *scale <= MAX_SCALE)
            else {
                vortex_bail!(
                    "output scale {result_scale} of {input} {op} {input} is outside the \
                     representable scale range of {} to {MAX_SCALE}",
                    i8::MIN
                );
            };
            let result_precision = input
                .precision()
                .saturating_add(input.precision().saturating_add(1))
                .min(MAX_PRECISION);
            DecimalDType::try_new(result_precision, result_scale)
        }
        NumericOperator::Div => {
            // Arrow follows Postgres and MySQL in adding a fixed four fractional digits. Its
            // precision formula `p1 - s1 + s2 + result_scale` simplifies to `p + result_scale`
            // once both operands share a dtype.
            let result_scale = input.scale().saturating_add(4).min(MAX_SCALE);
            let result_precision =
                <i16 as From<u8>>::from(input.precision()) + <i16 as From<i8>>::from(result_scale);
            let result_precision = u8::try_from(result_precision).map_err(|_| {
                vortex_err!(
                    InvalidArgument:
                    "decimal division result precision {result_precision} is invalid"
                )
            })?;
            DecimalDType::try_new(result_precision.min(MAX_PRECISION), result_scale)
        }
    }
}

/// Apply `op` to two stored values of `input`, returning the result stored in `result`'s width.
///
/// Returns `None` if the result overflows `result`'s precision, or if `op` is a division by zero.
pub(crate) fn checked_decimal_numeric(
    lhs: DecimalValue,
    rhs: DecimalValue,
    input: DecimalDType,
    result: DecimalDType,
    op: NumericOperator,
) -> Option<DecimalValue> {
    let work = decimal_numeric_work_dtype(input, result, op);
    match_each_decimal_value_type!(DecimalType::smallest_decimal_value_type(&work), |W| {
        let value = checked_at_width::<W>(lhs.cast::<W>()?, rhs.cast::<W>()?, result.scale(), op)?;
        DecimalValue::from(value).normalize(result)
    })
}

/// Pick a native width wide enough to hold every intermediate of an in-precision operation.
///
/// The result precision covers Add, Sub and Mul, whose intermediates are the result itself. Div
/// scales the dividend (or the divisor, for a negative result scale) by `10^result_scale` before
/// dividing, so it needs room for `p + |result_scale|` digits.
pub(crate) fn decimal_numeric_work_dtype(
    input: DecimalDType,
    result: DecimalDType,
    op: NumericOperator,
) -> DecimalDType {
    let precision = match op {
        NumericOperator::Add | NumericOperator::Sub | NumericOperator::Mul => result.precision(),
        NumericOperator::Div => input
            .precision()
            .saturating_add(result.scale().unsigned_abs())
            .max(result.precision())
            .min(MAX_PRECISION),
    };
    DecimalDType::new(precision, 0)
}

fn checked_at_width<W>(lhs: W, rhs: W, result_scale: i8, op: NumericOperator) -> Option<W>
where
    W: NativeDecimalType + CheckedAdd + CheckedSub + CheckedMul + CheckedDiv,
{
    match op {
        NumericOperator::Add => lhs.checked_add(&rhs),
        NumericOperator::Sub => lhs.checked_sub(&rhs),
        // The result scale is the sum of the operand scales, so the raw product is already
        // correctly scaled.
        NumericOperator::Mul => lhs.checked_mul(&rhs),
        NumericOperator::Div => {
            // Arrow scales the quotient by `10^(result_scale - lhs_scale + rhs_scale)`, which is
            // `10^result_scale` for equal operand dtypes. A negative exponent scales the divisor
            // instead of the dividend.
            let factor =
                decimal_scale_factor::<W>(<u32 as From<u8>>::from(result_scale.unsigned_abs()))?;
            if result_scale >= 0 {
                lhs.checked_mul(&factor)?.checked_div(&rhs)
            } else {
                lhs.checked_div(&rhs.checked_mul(&factor)?)
            }
        }
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
