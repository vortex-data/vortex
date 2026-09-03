// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native execution of the arithmetic operators over decimal arrays.
//!
//! Add, Sub, and Div operate on decimal operands sharing one logical [`DecimalDType`]. Mul also
//! accepts decimals with different logical dtypes and signed integer operands. It multiplies their
//! stored integers directly because the result scale is the sum of the operand scales. Div rescales
//! the dividend (or the divisor, for a negative result scale) before integer division. Result
//! precision and scale follow Arrow's rules.
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
use crate::Canonical;
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
use crate::dtype::PType;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalValue;
use crate::scalar::NumericOperator;
use crate::scalar::Scalar;
use crate::scalar::decimal_numeric_work_dtype;
use crate::validity::Validity;

/// Execute a numeric operation whose result is decimal.
pub(super) fn execute_numeric_decimal(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: NumericOperator,
    result_decimal_dtype: DecimalDType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let result_dtype = DType::Decimal(
        result_decimal_dtype,
        lhs.dtype().nullability() | rhs.dtype().nullability(),
    );
    let work_dtype = if op == NumericOperator::Mul {
        DecimalDType::new(result_decimal_dtype.precision(), 0)
    } else {
        let input_dtype = lhs
            .dtype()
            .as_decimal_opt()
            .vortex_expect("non-multiplication decimal operands share a decimal dtype");
        decimal_numeric_work_dtype(*input_dtype, result_decimal_dtype, op)
    };

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

/// An integer-valued decimal operator operand, retaining the input's physical buffer.
enum DecimalOperand {
    Array {
        values: Canonical,
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
            Columnar::Constant(array) => match integer_scalar_value(array.scalar())? {
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
            Columnar::Canonical(values) => {
                let validity = match &values {
                    Canonical::Decimal(values) => values.validity()?,
                    Canonical::Primitive(values) if values.ptype().is_signed_int() => {
                        values.validity()?
                    }
                    _ => unreachable!("unsupported decimal operand dtype {}", values.dtype()),
                };
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

fn integer_scalar_value(scalar: &Scalar) -> VortexResult<Option<DecimalValue>> {
    match scalar.dtype() {
        DType::Decimal(..) => Ok(scalar.as_decimal().decimal_value()),
        DType::Primitive(PType::I8, _) => Ok(scalar
            .as_primitive()
            .try_typed_value::<i8>()?
            .map(DecimalValue::from)),
        DType::Primitive(PType::I16, _) => Ok(scalar
            .as_primitive()
            .try_typed_value::<i16>()?
            .map(DecimalValue::from)),
        DType::Primitive(PType::I32, _) => Ok(scalar
            .as_primitive()
            .try_typed_value::<i32>()?
            .map(DecimalValue::from)),
        DType::Primitive(PType::I64, _) => Ok(scalar
            .as_primitive()
            .try_typed_value::<i64>()?
            .map(DecimalValue::from)),
        dtype => unreachable!("unsupported decimal operand dtype {dtype}"),
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

macro_rules! with_decimal_operand_values {
    ($operand:expr, | $values:ident | $body:expr) => {{
        match $operand {
            DecimalOperand::Array {
                values: Canonical::Decimal(values),
                ..
            } => {
                match_each_decimal_value_type!(values.values_type(), |T| {
                    let buffer = values.buffer::<T>();
                    let $values = buffer.as_slice();
                    $body
                })
            }
            DecimalOperand::Array {
                values: Canonical::Primitive(values),
                ..
            } => match values.ptype() {
                PType::I8 => {
                    let buffer = values.to_buffer::<i8>();
                    let $values = buffer.as_slice();
                    $body
                }
                PType::I16 => {
                    let buffer = values.to_buffer::<i16>();
                    let $values = buffer.as_slice();
                    $body
                }
                PType::I32 => {
                    let buffer = values.to_buffer::<i32>();
                    let $values = buffer.as_slice();
                    $body
                }
                PType::I64 => {
                    let buffer = values.to_buffer::<i64>();
                    let $values = buffer.as_slice();
                    $body
                }
                ptype => unreachable!("unsupported decimal operand ptype {ptype}"),
            },
            DecimalOperand::Array { values, .. } => {
                unreachable!("unsupported decimal operand dtype {}", values.dtype())
            }
            DecimalOperand::Constant { .. } => unreachable!("operand is an array"),
        }
    }};
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

    if let (
        DecimalOperand::Constant { value: lhs, .. },
        DecimalOperand::Constant { value: rhs, .. },
    ) = (lhs, rhs)
    {
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

    let values = execute_decimal_lanes::<W, Op>(lhs, rhs, constants, valid_rows)
        .map_err(|_lane| vortex_err!(InvalidArgument: "{}", Op::ERROR))?;

    Ok(decimal_array_narrowed(
        values,
        result_decimal_dtype,
        validity.union_nullability(result_dtype.nullability()),
    ))
}

fn execute_decimal_lanes<W, Op>(
    lhs: &DecimalOperand,
    rhs: &DecimalOperand,
    constants: &DecimalOpConstants<W>,
    valid_rows: &Mask,
) -> Result<Buffer<W>, usize>
where
    W: NativeDecimalType + CheckedAdd + CheckedSub + CheckedMul + CheckedDiv + Mul<Output = W>,
    Op: CheckedDecimalOp,
{
    match (lhs, rhs) {
        (DecimalOperand::Constant { value, .. }, DecimalOperand::Array { .. }) => {
            execute_decimal_constant_array::<W, Op>(value, rhs, constants, valid_rows)
        }
        (DecimalOperand::Array { .. }, DecimalOperand::Constant { value, .. }) => {
            execute_decimal_array_constant::<W, Op>(lhs, value, constants, valid_rows)
        }
        (DecimalOperand::Array { .. }, DecimalOperand::Array { .. }) => {
            execute_decimal_array_array::<W, Op>(lhs, rhs, constants, valid_rows)
        }
        (DecimalOperand::Constant { .. }, DecimalOperand::Constant { .. }) => {
            unreachable!("constant operands are handled before lane execution")
        }
    }
}

fn execute_decimal_constant_array<W, Op>(
    lhs: &DecimalValue,
    rhs: &DecimalOperand,
    constants: &DecimalOpConstants<W>,
    valid_rows: &Mask,
) -> Result<Buffer<W>, usize>
where
    W: NativeDecimalType + CheckedAdd + CheckedSub + CheckedMul + CheckedDiv + Mul<Output = W>,
    Op: CheckedDecimalOp,
{
    let lhs = typed_constant::<W>(lhs);
    with_decimal_operand_values!(rhs, |rhs| {
        checked_lanes(rhs, valid_rows, |rhs| {
            Op::apply(lhs, <W as BigCast>::from(rhs)?, constants)
        })
    })
}

fn execute_decimal_array_constant<W, Op>(
    lhs: &DecimalOperand,
    rhs: &DecimalValue,
    constants: &DecimalOpConstants<W>,
    valid_rows: &Mask,
) -> Result<Buffer<W>, usize>
where
    W: NativeDecimalType + CheckedAdd + CheckedSub + CheckedMul + CheckedDiv + Mul<Output = W>,
    Op: CheckedDecimalOp,
{
    let rhs = typed_constant::<W>(rhs);
    with_decimal_operand_values!(lhs, |lhs| {
        checked_lanes(lhs, valid_rows, |lhs| {
            Op::apply(<W as BigCast>::from(lhs)?, rhs, constants)
        })
    })
}

fn execute_decimal_array_array<W, Op>(
    lhs: &DecimalOperand,
    rhs: &DecimalOperand,
    constants: &DecimalOpConstants<W>,
    valid_rows: &Mask,
) -> Result<Buffer<W>, usize>
where
    W: NativeDecimalType + CheckedAdd + CheckedSub + CheckedMul + CheckedDiv + Mul<Output = W>,
    Op: CheckedDecimalOp,
{
    with_decimal_operand_values!(lhs, |lhs| {
        execute_decimal_array_rhs::<W, Op, _>(lhs, rhs, constants, valid_rows)
    })
}

fn execute_decimal_array_rhs<W, Op, L>(
    lhs: &[L],
    rhs: &DecimalOperand,
    constants: &DecimalOpConstants<W>,
    valid_rows: &Mask,
) -> Result<Buffer<W>, usize>
where
    W: NativeDecimalType + CheckedAdd + CheckedSub + CheckedMul + CheckedDiv + Mul<Output = W>,
    Op: CheckedDecimalOp,
    L: NativeDecimalType,
{
    with_decimal_operand_values!(rhs, |rhs| {
        checked_lanes(LaneZip::new(lhs, rhs), valid_rows, |(lhs, rhs)| {
            Op::apply(
                <W as BigCast>::from(lhs)?,
                <W as BigCast>::from(rhs)?,
                constants,
            )
        })
    })
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

fn typed_constant<W: NativeDecimalType>(value: &DecimalValue) -> W {
    value
        .cast::<W>()
        .vortex_expect("the working width must be able to represent the constant")
}
