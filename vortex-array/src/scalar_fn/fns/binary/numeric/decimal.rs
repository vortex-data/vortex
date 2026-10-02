// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native execution of the arithmetic operators over decimal arrays.
//!
//! Operands are decimals of any precision and scale, or signed integers, which act as decimals of
//! scale zero. Add and Sub align both stored integers to the finer scale; with equal scales they
//! apply directly. Mul takes the raw product, which the summed result scale leaves correctly
//! scaled, and Div rescales the dividend (or the divisor, for a negative exponent) before integer
//! division. Result precision and scale follow Arrow's rules — see
//! [`decimal_binary_result_dtype`](crate::scalar::decimal_binary_result_dtype).
//!
//! Lanes execute in a working width wide enough that in-precision inputs cannot spuriously
//! overflow an intermediate, then narrow to the result's own storage width. Every lane is still
//! checked at that width: [`DecimalArray`] does not validate its stored values against the
//! declared precision, so an out-of-precision value can reach a kernel and must not be able to
//! overflow it. An operation that overflows the result precision on a valid lane is an error;
//! invalid lanes never error.
//!
//! Storage width is independent of precision, so two operands may be stored at different widths.
//! A mismatched pair is widened to the wider of the two before the lane loop, zero-copy when they
//! already match, so each working width and operator monomorphizes one array kernel per storage
//! width rather than one per pair of storage widths.

use std::marker::PhantomData;
use std::ops::Mul;

use num_traits::CheckedAdd;
use num_traits::CheckedDiv;
use num_traits::CheckedMul;
use num_traits::CheckedSub;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_compute::lane_kernels::LaneZip;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use super::checked::checked_lanes;
use super::decimal_operand_dtype;
use crate::ArrayRef;
use crate::Canonical;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::DecimalArray;
use crate::arrays::decimal::DecimalArrayExt;
use crate::arrays::decimal::widened_buffer;
use crate::dtype::BigCast;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::NativeDecimalType;
use crate::match_each_decimal_value_type;
use crate::match_each_signed_integer_ptype;
use crate::scalar::DecimalValue;
use crate::scalar::NumericOperator;
use crate::scalar::Scalar;
use crate::scalar::decimal_align_exponents;
use crate::scalar::decimal_numeric_work_dtype;
use crate::validity::Validity;

/// Execute a numeric operation whose result is `result_decimal_dtype`.
pub(super) fn execute_numeric_decimal(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: NumericOperator,
    result_decimal_dtype: DecimalDType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let lhs_dtype = decimal_operand_dtype(lhs.dtype()).vortex_expect("lhs is a decimal operand");
    let rhs_dtype = decimal_operand_dtype(rhs.dtype()).vortex_expect("rhs is a decimal operand");
    let result_dtype = DType::Decimal(
        result_decimal_dtype,
        lhs.dtype().nullability() | rhs.dtype().nullability(),
    );

    // Fast path for null constant arrays.
    if is_null_constant(lhs) || is_null_constant(rhs) {
        return Ok(null_result(&result_dtype, lhs.len()));
    }

    let Some(lhs) = DecimalOperand::try_new(lhs, lhs_dtype, ctx)? else {
        return Ok(null_result(&result_dtype, lhs.len()));
    };
    let Some(rhs) = DecimalOperand::try_new(rhs, rhs_dtype, ctx)? else {
        return Ok(null_result(&result_dtype, rhs.len()));
    };
    let len = lhs.len();
    debug_assert_eq!(len, rhs.len());

    let validity = lhs.validity().and(rhs.validity())?;
    let valid_rows = validity.execute_mask(len, ctx)?;

    let work_dtype = decimal_numeric_work_dtype(lhs_dtype, rhs_dtype, result_decimal_dtype, op);
    let (lhs_exp, rhs_exp) =
        decimal_align_exponents(lhs_dtype, rhs_dtype, result_decimal_dtype, op);
    let aligned = lhs_exp != 0 || rhs_exp != 0;
    match_each_decimal_value_type!(DecimalType::smallest_decimal_value_type(&work_dtype), |W| {
        let constants = DecimalOpConstants::<W>::new(result_decimal_dtype, lhs_exp, rhs_exp)?;
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
                    ctx.allocator(),
                )
            };
        }

        // Equal-scale Add and Sub skip the per-lane alignment multiply.
        match op {
            NumericOperator::Add if aligned => execute_typed!(Aligned<CheckedDecimalAdd>),
            NumericOperator::Add => execute_typed!(CheckedDecimalAdd),
            NumericOperator::Sub if aligned => execute_typed!(Aligned<CheckedDecimalSub>),
            NumericOperator::Sub => execute_typed!(CheckedDecimalSub),
            NumericOperator::Mul => execute_typed!(CheckedDecimalMul),
            NumericOperator::Div => execute_typed!(Aligned<CheckedDecimalDiv>),
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
///
/// Signed integer operands are viewed as decimals of scale zero over the same buffer.
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
    fn try_new(
        array: &ArrayRef,
        decimal_dtype: DecimalDType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Self>> {
        let columnar = array.clone().execute::<Columnar>(ctx)?;

        match columnar {
            Columnar::Constant(array) => match constant_decimal_value(array.scalar())? {
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
            Columnar::Canonical(Canonical::Decimal(values)) => {
                let validity = values.validity()?;
                Ok(Some(Self::Array { values, validity }))
            }
            Columnar::Canonical(Canonical::Primitive(values)) => {
                let validity = values.validity()?;
                let values = match_each_signed_integer_ptype!(values.ptype(), |T| {
                    DecimalArray::new(values.to_buffer::<T>(), decimal_dtype, validity.clone())
                });
                Ok(Some(Self::Array { values, validity }))
            }
            Columnar::Canonical(values) => {
                vortex_bail!("unsupported decimal operand dtype {}", values.dtype())
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

/// The value of a decimal or signed integer constant, or `None` if it is null.
fn constant_decimal_value(scalar: &Scalar) -> VortexResult<Option<DecimalValue>> {
    match scalar.dtype() {
        DType::Decimal(..) => Ok(scalar.as_decimal().decimal_value()),
        DType::Primitive(ptype, _) => match_each_signed_integer_ptype!(ptype, |T| {
            Ok(scalar
                .as_primitive()
                .try_typed_value::<T>()?
                .map(DecimalValue::from))
        }),
        dtype => vortex_bail!("unsupported decimal operand dtype {dtype}"),
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
    /// Powers of ten that align the operands before the operator applies. See
    /// [`decimal_align_exponents`].
    lhs_scale_factor: W,
    rhs_scale_factor: W,
}

impl<W> DecimalOpConstants<W>
where
    W: NativeDecimalType + CheckedMul,
{
    fn new(result: DecimalDType, lhs_exp: u32, rhs_exp: u32) -> VortexResult<Self> {
        Ok(Self {
            bounds: DecimalValueBounds::new(result),
            lhs_scale_factor: decimal_scale_factor::<W>(lhs_exp)?,
            rhs_scale_factor: decimal_scale_factor::<W>(rhs_exp)?,
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

/// Scales both operands by their alignment factors, then applies `Op`.
struct Aligned<Op>(PhantomData<Op>);

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
        constants.bounds.in_precision(lhs.checked_div(&rhs)?)
    }
}

impl<Op: CheckedDecimalOp> CheckedDecimalOp for Aligned<Op> {
    const ERROR: &'static str = Op::ERROR;

    fn apply<W>(lhs: W, rhs: W, constants: &DecimalOpConstants<W>) -> Option<W>
    where
        W: NativeDecimalType + CheckedAdd + CheckedSub + CheckedMul + CheckedDiv + Mul<Output = W>,
    {
        let lhs = lhs.checked_mul(&constants.lhs_scale_factor)?;
        let rhs = rhs.checked_mul(&constants.rhs_scale_factor)?;
        Op::apply(lhs, rhs, constants)
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "typed decimal execution parameters"
)]
fn execute_decimal_typed<W, Op>(
    lhs: &DecimalOperand,
    rhs: &DecimalOperand,
    result_decimal_dtype: DecimalDType,
    result_dtype: &DType,
    validity: Validity,
    valid_rows: &Mask,
    constants: &DecimalOpConstants<W>,
    allocator: &BufferAllocatorRef,
) -> VortexResult<ArrayRef>
where
    W: NativeDecimalType + CheckedAdd + CheckedSub + CheckedMul + CheckedDiv + Mul<Output = W>,
    DecimalValue: From<W>,
    Op: CheckedDecimalOp,
{
    let len = lhs.len();

    let values = match (lhs, rhs) {
        (DecimalOperand::Array { values: lhs, .. }, DecimalOperand::Array { values: rhs, .. }) => {
            checked_decimal_arrays::<W, Op>(lhs, rhs, constants, valid_rows, allocator)
        }
        (DecimalOperand::Array { values: lhs, .. }, DecimalOperand::Constant { value, .. }) => {
            let rhs = typed_constant::<W>(value);
            match_each_decimal_value_type!(lhs.values_type(), |L| {
                let lhs = lhs.buffer::<L>();
                checked_lanes(
                    lhs.as_slice(),
                    valid_rows,
                    |lhs| Op::apply(<W as BigCast>::from(lhs)?, rhs, constants),
                    allocator,
                )
            })
        }
        (DecimalOperand::Constant { value, .. }, DecimalOperand::Array { values: rhs, .. }) => {
            let lhs = typed_constant::<W>(value);
            match_each_decimal_value_type!(rhs.values_type(), |R| {
                let rhs = rhs.buffer::<R>();
                checked_lanes(
                    rhs.as_slice(),
                    valid_rows,
                    |rhs| Op::apply(lhs, <W as BigCast>::from(rhs)?, constants),
                    allocator,
                )
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
        allocator,
    ))
}

/// Build the result array, narrowing to the dtype's own storage width when the working width is
/// wider than it. Only division picks a working width above the result precision, and only for a
/// negative result scale, so this copies in a corner case rather than on the common path.
fn decimal_array_narrowed<W: NativeDecimalType>(
    values: Buffer<W>,
    decimal_dtype: DecimalDType,
    validity: Validity,
    allocator: &BufferAllocatorRef,
) -> ArrayRef {
    let target = DecimalType::smallest_decimal_value_type(&decimal_dtype);
    if target == W::DECIMAL_TYPE {
        return DecimalArray::new(values, decimal_dtype, validity).into_array();
    }

    match_each_decimal_value_type!(target, |O| {
        let mut narrowed = BufferMut::with_capacity_in(values.len(), allocator.clone());
        narrowed.extend(values.as_slice().iter().copied().map(|value| {
            <O as BigCast>::from(value)
                .vortex_expect("precision-checked decimal result must fit the output width")
        }));
        DecimalArray::new(narrowed.freeze(), decimal_dtype, validity).into_array()
    })
}

fn checked_decimal_arrays<W, Op>(
    lhs: &DecimalArray,
    rhs: &DecimalArray,
    constants: &DecimalOpConstants<W>,
    valid_rows: &Mask,
    allocator: &BufferAllocatorRef,
) -> Result<Buffer<W>, usize>
where
    W: NativeDecimalType + CheckedAdd + CheckedSub + CheckedMul + CheckedDiv + Mul<Output = W>,
    Op: CheckedDecimalOp,
{
    debug_assert_eq!(lhs.len(), rhs.len());
    // Dispatch on one storage width, not a pair: a mismatched pair is widened to the wider side
    // first, which is a copy only in that rare case.
    let storage = lhs.values_type().max(rhs.values_type());
    match_each_decimal_value_type!(storage, |L| {
        let lhs = widened_buffer::<L>(lhs);
        let rhs = widened_buffer::<L>(rhs);
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
            allocator,
        )
    })
}

fn typed_constant<W: NativeDecimalType>(value: &DecimalValue) -> W {
    value
        .cast::<W>()
        .vortex_expect("the working width must be able to represent the constant")
}
