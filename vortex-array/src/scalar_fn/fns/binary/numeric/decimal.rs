// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native execution of the arithmetic operators over decimal arrays.
//!
//! Both operands share a logical [`DecimalDType`] (equal precision and scale). Add and Sub apply
//! directly to the unscaled stored integers and are exact at that shared scale. The result reserves
//! one additional precision digit for a carry, as long as that digit does not push the result into
//! a wider working width (see [`result_decimal_dtype`]).
//!
//! Lanes execute at the working width of the result precision. Every lane is checked twice: once
//! for overflow of the working width, and once against the bounds implied by the result precision.
//! A valid lane failing either check is an error; invalid lanes never error.

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::checked::CheckedValues;
use super::checked::checked_all_lanes;
use super::checked::checked_valid_lanes;
use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::DecimalArray;
use crate::arrays::decimal::DecimalArrayExt;
use crate::dtype::BigCast;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::NativeDecimalType;
use crate::dtype::i256;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalValue;
use crate::scalar::NumericOperator;
use crate::scalar::Scalar;
use crate::validity::Validity;

/// The widest decimal precision that still fits a native `i128`.
const NATIVE_MAX_PRECISION: u8 = <i128 as NativeDecimalType>::MAX_PRECISION;

/// Derive the result decimal dtype for a numeric operation over same-typed operands.
pub(crate) fn result_decimal_dtype(
    input: DecimalDType,
    op: NumericOperator,
) -> VortexResult<DecimalDType> {
    match op {
        NumericOperator::Add | NumericOperator::Sub => {
            // A p-digit Add/Sub result needs at most one carry digit, because
            // 2 * (10^p - 1) < 10^(p + 1). Reserving that digit is free while it stays inside a
            // native integer, but 39 digits no longer fit an `i128`: promoting `Decimal(38, s)` —
            // by far the most common wide decimal — to `Decimal(39, s)` would put every lane of
            // this and every later operation through software 256-bit arithmetic for one digit.
            // Arrow saturates at its own type boundary for the same reason, so the precision
            // saturates here too and a sum that no longer fits is an overflow error rather than a
            // silently widened column.
            let precision = input.precision();
            let cap = if precision <= NATIVE_MAX_PRECISION {
                NATIVE_MAX_PRECISION
            } else {
                <i256 as NativeDecimalType>::MAX_PRECISION
            };
            Ok(DecimalDType::new(
                precision.saturating_add(1).min(cap),
                input.scale(),
            ))
        }
        NumericOperator::Mul | NumericOperator::Div => {
            vortex_bail!("numeric operator {op} is not yet supported for decimal arrays")
        }
    }
}

/// Execute a numeric operation between two decimal arrays sharing a decimal dtype.
pub(super) fn execute_numeric_decimal(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: NumericOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let decimal_dtype = lhs
        .dtype()
        .as_decimal_opt()
        .vortex_expect("inputs are both decimals");

    let result_decimal_dtype = result_decimal_dtype(*decimal_dtype, op)?;
    let result_dtype = DType::Decimal(
        result_decimal_dtype,
        lhs.dtype().nullability() | rhs.dtype().nullability(),
    );

    // Fast path for null constant arrays.
    if is_null_constant(lhs) || is_null_constant(rhs) {
        return Ok(null_result(&result_dtype, lhs.len()));
    }

    let Some(lhs) = DecimalOperand::try_new(lhs, ctx)? else {
        return Ok(null_result(&result_dtype, lhs.len()));
    };
    let Some(rhs) = DecimalOperand::try_new(rhs, ctx)? else {
        return Ok(null_result(&result_dtype, rhs.len()));
    };
    let len = lhs.len();
    debug_assert_eq!(len, rhs.len());

    let validity = lhs.validity().and(rhs.validity())?;
    let valid_rows = validity.execute_mask(len, ctx)?;

    match_each_decimal_value_type!(
        DecimalType::smallest_decimal_value_type(&result_decimal_dtype),
        |W| {
            execute_decimal_at_width::<W>(
                &lhs,
                &rhs,
                op,
                result_decimal_dtype,
                &result_dtype,
                validity,
                &valid_rows,
            )
        }
    )
}

fn is_null_constant(array: &ArrayRef) -> bool {
    array
        .as_opt::<Constant>()
        .is_some_and(|constant| constant.scalar().is_null())
}

fn null_result(dtype: &DType, len: usize) -> ArrayRef {
    ConstantArray::new(Scalar::null(dtype.clone()), len).into_array()
}

/// A decimal binary-operator operand: a canonical decimal array or a non-null constant.
enum DecimalOperand {
    Array {
        values: DecimalArray,
        validity: Validity,
    },
    Constant {
        value: DecimalValue,
        len: usize,
        validity: Validity,
    },
}

impl DecimalOperand {
    fn try_new(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Option<Self>> {
        let columnar = array.clone().execute::<Columnar>(ctx)?;

        match columnar {
            Columnar::Constant(array) => match array.scalar().as_decimal().decimal_value() {
                Some(value) => Ok(Some(Self::Constant {
                    value,
                    len: array.len(),
                    validity: if array.scalar().dtype().is_nullable() {
                        Validity::AllValid
                    } else {
                        Validity::NonNullable
                    },
                })),
                None => Ok(None),
            },
            Columnar::Canonical(array) => {
                let values = array.as_decimal().to_owned();
                let validity = values.validity()?;
                Ok(Some(Self::Array { values, validity }))
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Array { values, .. } => values.len(),
            Self::Constant { len, .. } => *len,
        }
    }

    fn validity(&self) -> Validity {
        match self {
            Self::Array { validity, .. } | Self::Constant { validity, .. } => validity.clone(),
        }
    }
}

/// Lane arithmetic for a decimal working width, phrased so that neither the overflow test nor the
/// precision test needs a branch.
///
/// The natural spelling — `checked_add` followed by `lower <= v && v <= upper` — costs four
/// three-way comparisons per lane. That is one instruction each for a native integer, but for
/// [`i256`] every one of them is a 256-bit lexicographic compare. Deriving overflow from the sign
/// bits instead, and folding the two bounds into a single unsigned comparison, removes all four.
trait DecimalLane: NativeDecimalType {
    /// `self + rhs` wrapped to `Self`, and whether the exact sum overflowed `Self`.
    fn overflowing_add(self, rhs: Self) -> (Self, bool);

    /// `self - rhs` wrapped to `Self`, and whether the exact difference overflowed `Self`.
    fn overflowing_sub(self, rhs: Self) -> (Self, bool);

    /// `self - origin` wrapped to `Self`.
    fn wrapping_sub(self, origin: Self) -> Self;

    /// `self <= other`, with both operands read as unsigned integers of the same width.
    fn le_unsigned(self, other: Self) -> bool;
}

macro_rules! impl_decimal_lane {
    ($T:ty, $U:ty) => {
        impl DecimalLane for $T {
            #[inline]
            fn overflowing_add(self, rhs: Self) -> (Self, bool) {
                <$T>::overflowing_add(self, rhs)
            }

            #[inline]
            fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
                <$T>::overflowing_sub(self, rhs)
            }

            #[inline]
            fn wrapping_sub(self, origin: Self) -> Self {
                <$T>::wrapping_sub(self, origin)
            }

            #[inline]
            #[expect(clippy::cast_sign_loss, reason = "reinterpreting the bit pattern")]
            fn le_unsigned(self, other: Self) -> bool {
                (self as $U) <= (other as $U)
            }
        }
    };
}

impl_decimal_lane!(i8, u8);
impl_decimal_lane!(i16, u16);
impl_decimal_lane!(i32, u32);
impl_decimal_lane!(i64, u64);
impl_decimal_lane!(i128, u128);

impl DecimalLane for i256 {
    #[inline]
    fn overflowing_add(self, rhs: Self) -> (Self, bool) {
        i256::overflowing_add(self, rhs)
    }

    #[inline]
    fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
        i256::overflowing_sub(self, rhs)
    }

    #[inline]
    fn wrapping_sub(self, origin: Self) -> Self {
        i256::wrapping_sub(&self, origin)
    }

    #[inline]
    #[expect(clippy::cast_sign_loss, reason = "reinterpreting the bit pattern")]
    fn le_unsigned(self, other: Self) -> bool {
        let (low, high) = self.to_parts();
        let (other_low, other_high) = other.to_parts();
        (high as u128, low) <= (other_high as u128, other_low)
    }
}

/// Per-execution bounds for checked decimal lane operations at working width `W`.
///
/// Native-width checked arithmetic only detects overflow of `W`, whose range may exceed the
/// declared decimal precision. In particular, `i256` can represent values outside precision 76,
/// so every native result must also be checked against these logical bounds.
struct DecimalValueBounds<W> {
    /// The inclusive lower stored-value bound implied by the result precision.
    lower_bound: W,
    /// The width of the in-precision range, `upper_bound - lower_bound`.
    span: W,
}

impl<W: DecimalLane> DecimalValueBounds<W> {
    fn new(dtype: DecimalDType) -> Self {
        let precision = dtype.precision() as usize;
        let lower_bound = W::MIN_BY_PRECISION[precision];
        Self {
            lower_bound,
            span: W::MAX_BY_PRECISION[precision].wrapping_sub(lower_bound),
        }
    }

    /// Bounds-check a candidate result against the result precision.
    ///
    /// Biasing by the lower bound turns the two-sided signed range test into a single unsigned
    /// one: `x` lies in `[lower, upper]` exactly when `x - lower`, read as an unsigned integer,
    /// is at most `upper - lower`. Values below the lower bound wrap around to a large unsigned
    /// value and fail the same comparison.
    #[inline]
    fn in_precision(&self, value: W) -> bool {
        value.wrapping_sub(self.lower_bound).le_unsigned(self.span)
    }
}

/// A checked fixed-point decimal operation on unscaled values at working width `W`.
trait CheckedDecimalOp {
    const ERROR: &'static str;

    fn apply<W: DecimalLane>(lhs: W, rhs: W, bounds: &DecimalValueBounds<W>) -> Option<W>;
}

struct CheckedDecimalAdd;

struct CheckedDecimalSub;

impl CheckedDecimalOp for CheckedDecimalAdd {
    const ERROR: &'static str = "decimal overflow in checked add";

    #[inline]
    fn apply<W: DecimalLane>(lhs: W, rhs: W, bounds: &DecimalValueBounds<W>) -> Option<W> {
        let (value, overflow) = lhs.overflowing_add(rhs);
        (!overflow & bounds.in_precision(value)).then_some(value)
    }
}

impl CheckedDecimalOp for CheckedDecimalSub {
    const ERROR: &'static str = "decimal overflow in checked sub";

    #[inline]
    fn apply<W: DecimalLane>(lhs: W, rhs: W, bounds: &DecimalValueBounds<W>) -> Option<W> {
        let (value, overflow) = lhs.overflowing_sub(rhs);
        (!overflow & bounds.in_precision(value)).then_some(value)
    }
}

fn execute_decimal_at_width<W>(
    lhs: &DecimalOperand,
    rhs: &DecimalOperand,
    op: NumericOperator,
    result_decimal_dtype: DecimalDType,
    result_dtype: &DType,
    validity: Validity,
    valid_rows: &Mask,
) -> VortexResult<ArrayRef>
where
    W: DecimalLane,
    DecimalValue: From<W>,
{
    macro_rules! execute_typed {
        ($Op:ty) => {
            execute_decimal_typed::<W, $Op>(
                lhs,
                rhs,
                result_decimal_dtype,
                result_dtype,
                validity,
                valid_rows,
            )
        };
    }

    match op {
        NumericOperator::Add => execute_typed!(CheckedDecimalAdd),
        NumericOperator::Sub => execute_typed!(CheckedDecimalSub),
        NumericOperator::Mul | NumericOperator::Div => vortex_bail!(
            "numeric operator {:?} is not yet supported for decimal arrays",
            op
        ),
    }
}

fn execute_decimal_typed<W, Op>(
    lhs: &DecimalOperand,
    rhs: &DecimalOperand,
    result_decimal_dtype: DecimalDType,
    result_dtype: &DType,
    validity: Validity,
    valid_rows: &Mask,
) -> VortexResult<ArrayRef>
where
    W: DecimalLane,
    DecimalValue: From<W>,
    Op: CheckedDecimalOp,
{
    let len = lhs.len();
    let bounds = DecimalValueBounds::<W>::new(result_decimal_dtype);

    let checked = match (lhs, rhs) {
        (DecimalOperand::Array { values: lhs, .. }, DecimalOperand::Array { values: rhs, .. }) => {
            checked_decimal_arrays::<W, Op>(lhs, rhs, &bounds, valid_rows)
        }
        (DecimalOperand::Array { values: lhs, .. }, DecimalOperand::Constant { value, .. }) => {
            let rhs = typed_constant::<W>(value);
            match_each_decimal_value_type!(lhs.values_type(), |L| {
                let lhs = lhs.buffer::<L>();
                checked_decimal_lanes::<W, _>(len, valid_rows, |idx| {
                    Op::apply(<W as BigCast>::from(lhs[idx])?, rhs, &bounds)
                })
            })
        }
        (DecimalOperand::Constant { value, .. }, DecimalOperand::Array { values: rhs, .. }) => {
            let lhs = typed_constant::<W>(value);
            match_each_decimal_value_type!(rhs.values_type(), |R| {
                let rhs = rhs.buffer::<R>();
                checked_decimal_lanes::<W, _>(len, valid_rows, |idx| {
                    Op::apply(lhs, <W as BigCast>::from(rhs[idx])?, &bounds)
                })
            })
        }
        (
            DecimalOperand::Constant { value: lhs, .. },
            DecimalOperand::Constant { value: rhs, .. },
        ) => {
            let lhs = typed_constant::<W>(lhs);
            let rhs = typed_constant::<W>(rhs);
            let value = Op::apply(lhs, rhs, &bounds)
                .ok_or_else(|| vortex_err!(InvalidArgument: "{}", Op::ERROR))?;
            return Ok(ConstantArray::new(
                Scalar::decimal(
                    DecimalValue::from(value),
                    result_decimal_dtype,
                    result_dtype.nullability(),
                ),
                len,
            )
            .into_array());
        }
    };

    if checked.failed {
        return Err(vortex_err!(InvalidArgument: "{}", Op::ERROR));
    }

    Ok(DecimalArray::new(
        checked.values,
        result_decimal_dtype,
        validity.union_nullability(result_dtype.nullability()),
    )
    .into_array())
}

fn checked_decimal_arrays<W, Op>(
    lhs: &DecimalArray,
    rhs: &DecimalArray,
    bounds: &DecimalValueBounds<W>,
    valid_rows: &Mask,
) -> CheckedValues<W>
where
    W: DecimalLane,
    Op: CheckedDecimalOp,
{
    let len = lhs.len();
    debug_assert_eq!(len, rhs.len());
    match_each_decimal_value_type!(lhs.values_type(), |L| {
        let lhs = lhs.buffer::<L>();
        match_each_decimal_value_type!(rhs.values_type(), |R| {
            let rhs = rhs.buffer::<R>();
            checked_decimal_lanes::<W, _>(len, valid_rows, |idx| {
                Op::apply(
                    <W as BigCast>::from(lhs[idx])?,
                    <W as BigCast>::from(rhs[idx])?,
                    bounds,
                )
            })
        })
    })
}

fn checked_decimal_lanes<W, F>(len: usize, valid_rows: &Mask, checked_at: F) -> CheckedValues<W>
where
    W: NativeDecimalType,
    F: FnMut(usize) -> Option<W>,
{
    match valid_rows.bit_buffer() {
        AllOr::All => checked_all_lanes(len, checked_at),
        AllOr::None => CheckedValues::<W>::zeroed(len),
        AllOr::Some(valid_bits) => checked_valid_lanes(len, valid_bits, checked_at),
    }
}

fn typed_constant<W: NativeDecimalType>(value: &DecimalValue) -> W {
    value
        .cast::<W>()
        .vortex_expect("the working width must be able to represent the constant")
}
