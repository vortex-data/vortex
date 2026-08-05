// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native execution of the arithmetic operators over decimal arrays.
//!
//! Both operands share a logical [`DecimalDType`] (equal precision and scale). Add and Sub apply
//! directly to the unscaled stored integers and are exact at that shared scale. The result reserves
//! one additional precision digit for a carry, capped at Vortex's maximum decimal precision.
//!
//! Lanes execute in a working width chosen so that in-precision inputs cannot spuriously
//! overflow an intermediate value. An operation that overflows the result precision on a valid
//! lane is an error; invalid lanes never error.

use num_traits::CheckedAdd;
use num_traits::CheckedSub;
use vortex_buffer::Buffer;
use vortex_compute::lane_kernels::LaneZip;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
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
use crate::dtype::NativeDecimalType;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalValue;
use crate::scalar::NumericOperator;
use crate::scalar::Scalar;
use crate::scalar::decimal_numeric_result_dtype;
use crate::validity::Validity;

/// Derive the result decimal dtype for a numeric operation over same-typed operands.
pub(crate) fn result_decimal_dtype(
    input: DecimalDType,
    op: NumericOperator,
) -> VortexResult<DecimalDType> {
    match op {
        NumericOperator::Add | NumericOperator::Sub => decimal_numeric_result_dtype(input, op),
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
        let precision = dtype.precision() as usize;
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

/// A checked fixed-point decimal operation on unscaled values at working width `W`.
trait CheckedDecimalOp {
    const ERROR: &'static str;

    fn apply<W>(lhs: W, rhs: W, bounds: &DecimalValueBounds<W>) -> Option<W>
    where
        W: NativeDecimalType + CheckedAdd + CheckedSub;
}

struct CheckedDecimalAdd;

struct CheckedDecimalSub;

impl CheckedDecimalOp for CheckedDecimalAdd {
    const ERROR: &'static str = "decimal overflow in checked add";

    fn apply<W>(lhs: W, rhs: W, bounds: &DecimalValueBounds<W>) -> Option<W>
    where
        W: NativeDecimalType + CheckedAdd + CheckedSub,
    {
        bounds.in_precision(lhs.checked_add(&rhs)?)
    }
}

impl CheckedDecimalOp for CheckedDecimalSub {
    const ERROR: &'static str = "decimal overflow in checked sub";

    fn apply<W>(lhs: W, rhs: W, bounds: &DecimalValueBounds<W>) -> Option<W>
    where
        W: NativeDecimalType + CheckedAdd + CheckedSub,
    {
        bounds.in_precision(lhs.checked_sub(&rhs)?)
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
    W: NativeDecimalType + CheckedAdd + CheckedSub,
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
    W: NativeDecimalType + CheckedAdd + CheckedSub,
    DecimalValue: From<W>,
    Op: CheckedDecimalOp,
{
    let len = lhs.len();
    let bounds = DecimalValueBounds::<W>::new(result_decimal_dtype);

    let values = match (lhs, rhs) {
        (DecimalOperand::Array { values: lhs, .. }, DecimalOperand::Array { values: rhs, .. }) => {
            checked_decimal_arrays::<W, Op>(lhs, rhs, &bounds, valid_rows)
        }
        (DecimalOperand::Array { values: lhs, .. }, DecimalOperand::Constant { value, .. }) => {
            let rhs = typed_constant::<W>(value);
            match_each_decimal_value_type!(lhs.values_type(), |L| {
                let lhs = lhs.buffer::<L>();
                checked_lanes(lhs.as_slice(), valid_rows, |lhs| {
                    Op::apply(<W as BigCast>::from(lhs)?, rhs, &bounds)
                })
            })
        }
        (DecimalOperand::Constant { value, .. }, DecimalOperand::Array { values: rhs, .. }) => {
            let lhs = typed_constant::<W>(value);
            match_each_decimal_value_type!(rhs.values_type(), |R| {
                let rhs = rhs.buffer::<R>();
                checked_lanes(rhs.as_slice(), valid_rows, |rhs| {
                    Op::apply(lhs, <W as BigCast>::from(rhs)?, &bounds)
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
    }
    .map_err(|_lane| vortex_err!(InvalidArgument: "{}", Op::ERROR))?;

    Ok(DecimalArray::new(
        values,
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
) -> Result<Buffer<W>, usize>
where
    W: NativeDecimalType + CheckedAdd + CheckedSub,
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
                        bounds,
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
