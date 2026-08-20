// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native execution of the arithmetic operators over decimal arrays.
//!
//! Both operands share a logical [`DecimalDType`] (equal precision and scale). Add and Sub apply
//! directly to the unscaled stored integers and are exact at that shared scale. Mul takes the raw
//! product, which the doubled result scale leaves correctly scaled, and Div rescales the dividend
//! (or the divisor, for a negative result scale) before integer division. Result precision and
//! scale follow Arrow's rules — see [`decimal_numeric_result_dtype`].
//!
//! Lanes execute in a working width wide enough that in-precision inputs cannot spuriously
//! overflow an intermediate, then narrow to the result's own storage width. Every lane is still
//! checked at that width: [`DecimalArray`] does not validate its stored values against the
//! declared precision, so an out-of-precision value can reach a kernel and must not be able to
//! overflow it. An operation that overflows the result precision on a valid lane is an error;
//! invalid lanes never error.

use std::ops::Mul;

use num_traits::CheckedAdd;
use num_traits::CheckedDiv;
use num_traits::CheckedMul;
use num_traits::CheckedSub;
use vortex_buffer::Buffer;
use vortex_compute::lane_kernels::LaneZip;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use super::checked::checked_lanes;
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
use crate::dtype::DecimalType;
use crate::dtype::NativeDecimalType;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalValue;
use crate::scalar::NumericOperator;
use crate::scalar::Scalar;
use crate::scalar::decimal_numeric_result_dtype;
use crate::scalar::decimal_numeric_work_dtype;
use crate::validity::Validity;

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

    let result_decimal_dtype = decimal_numeric_result_dtype(*decimal_dtype, op)?;
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

    let work_dtype = decimal_numeric_work_dtype(*decimal_dtype, result_decimal_dtype, op);
    match_each_decimal_value_type!(DecimalType::smallest_decimal_value_type(&work_dtype), |W| {
        let constants = DecimalOpConstants::<W>::new(result_decimal_dtype, op)?;
        macro_rules! execute_typed {
            ($Op:ty) => {
                execute_decimal_typed::<W, $Op>(
                    &lhs,
                    &rhs,
                    result_decimal_dtype,
                    &result_dtype,
                    validity,
                    &valid_rows,
                    &constants,
                )
            };
        }

        match op {
            NumericOperator::Add => execute_typed!(CheckedDecimalAdd),
            NumericOperator::Sub => execute_typed!(CheckedDecimalSub),
            NumericOperator::Mul => execute_typed!(CheckedDecimalMul),
            NumericOperator::Div => execute_typed!(CheckedDecimalDiv),
        }
    })
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

/// Per-execution bounds for checked decimal lane operations at working width `W`.
///
/// Native-width checked arithmetic only detects overflow of `W`, whose range may exceed the
/// declared decimal precision. In particular, `i256` can represent values outside precision 76,
/// so every native result must also be checked against these logical bounds.
struct DecimalValueBounds<W> {
    /// Inclusive stored-value bounds implied by the result precision.
    lower_bound: W,
    upper_bound: W,
}

impl<W: NativeDecimalType> DecimalValueBounds<W> {
    fn new(dtype: DecimalDType) -> Self {
        let precision = usize::from(dtype.precision());
        Self {
            lower_bound: W::MIN_BY_PRECISION[precision],
            upper_bound: W::MAX_BY_PRECISION[precision],
        }
    }

    /// Bounds-check a candidate result against the result precision.
    fn in_precision(&self, value: W) -> Option<W> {
        (self.lower_bound <= value && value <= self.upper_bound).then_some(value)
    }
}

/// Per-execution constants for a decimal operation at working width `W`, hoisted out of the
/// lane loop.
struct DecimalOpConstants<W> {
    bounds: DecimalValueBounds<W>,
    /// Arrow's division rescaling factors. Both are one for every other operator.
    lhs_scale_factor: W,
    rhs_scale_factor: W,
}

impl<W> DecimalOpConstants<W>
where
    W: NativeDecimalType + CheckedMul,
{
    fn new(result: DecimalDType, op: NumericOperator) -> VortexResult<Self> {
        let one = <W as BigCast>::from(1_i8).vortex_expect("one fits every decimal working width");
        let (lhs_scale_factor, rhs_scale_factor) = if op == NumericOperator::Div {
            // Arrow scales the quotient by 10^(result_scale - lhs_scale + rhs_scale). Both
            // Vortex operands share a dtype, so this simplifies to 10^result_scale. A negative
            // exponent scales the divisor instead of the dividend.
            let exponent = <u32 as From<u8>>::from(result.scale().unsigned_abs());
            if result.scale() >= 0 {
                (decimal_scale_factor::<W>(exponent)?, one)
            } else {
                (one, decimal_scale_factor::<W>(exponent)?)
            }
        } else {
            (one, one)
        };

        Ok(Self {
            bounds: DecimalValueBounds::new(result),
            lhs_scale_factor,
            rhs_scale_factor,
        })
    }
}

fn decimal_scale_factor<W>(exp: u32) -> VortexResult<W>
where
    W: NativeDecimalType + CheckedMul,
{
    let ten = <W as BigCast>::from(10_i8).vortex_expect("ten fits every decimal working width");
    let mut factor =
        <W as BigCast>::from(1_i8).vortex_expect("one fits every decimal working width");
    for _ in 0..exp {
        factor = factor.checked_mul(&ten).ok_or_else(|| {
            vortex_err!(
                InvalidArgument:
                "decimal scale factor 10^{exp} cannot be represented at the working width"
            )
        })?;
    }
    Ok(factor)
}

/// A checked decimal operation on unscaled values at working width `W`.
trait CheckedDecimalOp {
    const ERROR: &'static str;

    fn apply<W>(lhs: W, rhs: W, constants: &DecimalOpConstants<W>) -> Option<W>
    where
        W: NativeDecimalType + CheckedAdd + CheckedSub + CheckedMul + CheckedDiv + Mul<Output = W>;
}

struct CheckedDecimalAdd;

struct CheckedDecimalSub;

struct CheckedDecimalMul;

struct CheckedDecimalDiv;

impl CheckedDecimalOp for CheckedDecimalAdd {
    const ERROR: &'static str = "decimal overflow in checked add";

    fn apply<W>(lhs: W, rhs: W, constants: &DecimalOpConstants<W>) -> Option<W>
    where
        W: NativeDecimalType + CheckedAdd + CheckedSub + CheckedMul + CheckedDiv + Mul<Output = W>,
    {
        constants.bounds.in_precision(lhs.checked_add(&rhs)?)
    }
}

impl CheckedDecimalOp for CheckedDecimalSub {
    const ERROR: &'static str = "decimal overflow in checked sub";

    fn apply<W>(lhs: W, rhs: W, constants: &DecimalOpConstants<W>) -> Option<W>
    where
        W: NativeDecimalType + CheckedAdd + CheckedSub + CheckedMul + CheckedDiv + Mul<Output = W>,
    {
        constants.bounds.in_precision(lhs.checked_sub(&rhs)?)
    }
}

impl CheckedDecimalOp for CheckedDecimalMul {
    const ERROR: &'static str = "decimal overflow in checked mul";

    fn apply<W>(lhs: W, rhs: W, constants: &DecimalOpConstants<W>) -> Option<W>
    where
        W: NativeDecimalType + CheckedAdd + CheckedSub + CheckedMul + CheckedDiv + Mul<Output = W>,
    {
        constants.bounds.in_precision(lhs.checked_mul(&rhs)?)
    }
}

impl CheckedDecimalOp for CheckedDecimalDiv {
    const ERROR: &'static str = "decimal overflow or division by zero in checked div";

    fn apply<W>(lhs: W, rhs: W, constants: &DecimalOpConstants<W>) -> Option<W>
    where
        W: NativeDecimalType + CheckedAdd + CheckedSub + CheckedMul + CheckedDiv + Mul<Output = W>,
    {
        let lhs = lhs.checked_mul(&constants.lhs_scale_factor)?;
        let rhs = rhs.checked_mul(&constants.rhs_scale_factor)?;
        constants.bounds.in_precision(lhs.checked_div(&rhs)?)
    }
}

fn execute_decimal_typed<W, Op>(
    lhs: &DecimalOperand,
    rhs: &DecimalOperand,
    result_decimal_dtype: DecimalDType,
    result_dtype: &DType,
    validity: Validity,
    valid_rows: &Mask,
    constants: &DecimalOpConstants<W>,
) -> VortexResult<ArrayRef>
where
    W: NativeDecimalType + CheckedAdd + CheckedSub + CheckedMul + CheckedDiv + Mul<Output = W>,
    DecimalValue: From<W>,
    Op: CheckedDecimalOp,
{
    let len = lhs.len();

    let values = match (lhs, rhs) {
        (DecimalOperand::Array { values: lhs, .. }, DecimalOperand::Array { values: rhs, .. }) => {
            checked_decimal_arrays::<W, Op>(lhs, rhs, constants, valid_rows)
        }
        (DecimalOperand::Array { values: lhs, .. }, DecimalOperand::Constant { value, .. }) => {
            let rhs = typed_constant::<W>(value);
            match_each_decimal_value_type!(lhs.values_type(), |L| {
                let lhs = lhs.buffer::<L>();
                checked_lanes(lhs.as_slice(), valid_rows, |lhs| {
                    Op::apply(<W as BigCast>::from(lhs)?, rhs, constants)
                })
            })
        }
        (DecimalOperand::Constant { value, .. }, DecimalOperand::Array { values: rhs, .. }) => {
            let lhs = typed_constant::<W>(value);
            match_each_decimal_value_type!(rhs.values_type(), |R| {
                let rhs = rhs.buffer::<R>();
                checked_lanes(rhs.as_slice(), valid_rows, |rhs| {
                    Op::apply(lhs, <W as BigCast>::from(rhs)?, constants)
                })
            })
        }
        (
            DecimalOperand::Constant { value: lhs, .. },
            DecimalOperand::Constant { value: rhs, .. },
        ) => {
            let lhs = typed_constant::<W>(lhs);
            let rhs = typed_constant::<W>(rhs);
            let value = Op::apply(lhs, rhs, constants)
                .ok_or_else(|| vortex_err!(InvalidArgument: "{}", Op::ERROR))?;
            let value = DecimalValue::from(value)
                .normalize(result_decimal_dtype)
                .vortex_expect("bounds-checked result fits the result precision");
            return Ok(ConstantArray::new(
                Scalar::decimal(value, result_decimal_dtype, result_dtype.nullability()),
                len,
            )
            .into_array());
        }
    }
    .map_err(|_lane| vortex_err!(InvalidArgument: "{}", Op::ERROR))?;

    Ok(decimal_array_narrowed(
        values,
        result_decimal_dtype,
        validity.union_nullability(result_dtype.nullability()),
    ))
}

/// Build the result array, narrowing to the dtype's own storage width when the working width is
/// wider than it. Only division picks a working width above the result precision, and only for a
/// negative result scale, so this copies in a corner case rather than on the common path.
fn decimal_array_narrowed<W: NativeDecimalType>(
    values: Buffer<W>,
    decimal_dtype: DecimalDType,
    validity: Validity,
) -> ArrayRef {
    let target = DecimalType::smallest_decimal_value_type(&decimal_dtype);
    if target == W::DECIMAL_TYPE {
        return DecimalArray::new(values, decimal_dtype, validity).into_array();
    }

    match_each_decimal_value_type!(target, |O| {
        let narrowed: Buffer<O> = values
            .as_slice()
            .iter()
            .copied()
            .map(|value| {
                <O as BigCast>::from(value)
                    .vortex_expect("precision-checked decimal result must fit the output width")
            })
            .collect();
        DecimalArray::new(narrowed, decimal_dtype, validity).into_array()
    })
}

fn checked_decimal_arrays<W, Op>(
    lhs: &DecimalArray,
    rhs: &DecimalArray,
    constants: &DecimalOpConstants<W>,
    valid_rows: &Mask,
) -> Result<Buffer<W>, usize>
where
    W: NativeDecimalType + CheckedAdd + CheckedSub + CheckedMul + CheckedDiv + Mul<Output = W>,
    Op: CheckedDecimalOp,
{
    debug_assert_eq!(lhs.len(), rhs.len());
    match_each_decimal_value_type!(lhs.values_type(), |L| {
        let lhs = lhs.buffer::<L>();
        match_each_decimal_value_type!(rhs.values_type(), |R| {
            let rhs = rhs.buffer::<R>();
            checked_lanes(
                LaneZip::new(lhs.as_slice(), rhs.as_slice()),
                valid_rows,
                |(lhs, rhs)| {
                    Op::apply(
                        <W as BigCast>::from(lhs)?,
                        <W as BigCast>::from(rhs)?,
                        constants,
                    )
                },
            )
        })
    })
}

fn typed_constant<W: NativeDecimalType>(value: &DecimalValue) -> W {
    value
        .cast::<W>()
        .vortex_expect("the working width must be able to represent the constant")
}
