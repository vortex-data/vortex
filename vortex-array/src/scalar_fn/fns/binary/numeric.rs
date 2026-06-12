// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::dtype::half::f16;
use crate::match_each_native_ptype;
use crate::scalar::NumericOperator;
use crate::scalar::Scalar;
use crate::validity::Validity;

struct CheckedAdd;

struct CheckedSub;

struct CheckedMul;

struct CheckedDiv;

trait CheckedPrimitiveOp {
    const ERROR: &'static str;
}

trait CheckedPrimitiveBinary<T: NativePType>: CheckedPrimitiveOp {
    fn checked(lhs: T, rhs: T) -> Option<T>;
}

trait CheckedPrimitiveBinaryAll:
    CheckedPrimitiveBinary<u8>
    + CheckedPrimitiveBinary<u16>
    + CheckedPrimitiveBinary<u32>
    + CheckedPrimitiveBinary<u64>
    + CheckedPrimitiveBinary<i8>
    + CheckedPrimitiveBinary<i16>
    + CheckedPrimitiveBinary<i32>
    + CheckedPrimitiveBinary<i64>
    + CheckedPrimitiveBinary<f16>
    + CheckedPrimitiveBinary<f32>
    + CheckedPrimitiveBinary<f64>
{
}

impl<Op> CheckedPrimitiveBinaryAll for Op where
    Op: CheckedPrimitiveBinary<u8>
        + CheckedPrimitiveBinary<u16>
        + CheckedPrimitiveBinary<u32>
        + CheckedPrimitiveBinary<u64>
        + CheckedPrimitiveBinary<i8>
        + CheckedPrimitiveBinary<i16>
        + CheckedPrimitiveBinary<i32>
        + CheckedPrimitiveBinary<i64>
        + CheckedPrimitiveBinary<f16>
        + CheckedPrimitiveBinary<f32>
        + CheckedPrimitiveBinary<f64>
{
}

impl CheckedPrimitiveOp for CheckedAdd {
    const ERROR: &'static str = "integer overflow in checked add";
}

impl CheckedPrimitiveOp for CheckedSub {
    const ERROR: &'static str = "integer overflow in checked sub";
}

impl CheckedPrimitiveOp for CheckedMul {
    const ERROR: &'static str = "integer overflow in checked mul";
}

impl CheckedPrimitiveOp for CheckedDiv {
    const ERROR: &'static str = "integer division by zero or overflow in checked div";
}

impl<T: CheckedArithmetic> CheckedPrimitiveBinary<T> for CheckedAdd {
    #[inline(always)]
    fn checked(lhs: T, rhs: T) -> Option<T> {
        lhs.checked_add(rhs)
    }
}

impl<T: CheckedArithmetic> CheckedPrimitiveBinary<T> for CheckedSub {
    #[inline(always)]
    fn checked(lhs: T, rhs: T) -> Option<T> {
        lhs.checked_sub(rhs)
    }
}

impl<T: CheckedArithmetic> CheckedPrimitiveBinary<T> for CheckedMul {
    #[inline(always)]
    fn checked(lhs: T, rhs: T) -> Option<T> {
        lhs.checked_mul(rhs)
    }
}

impl<T: CheckedArithmetic> CheckedPrimitiveBinary<T> for CheckedDiv {
    #[inline(always)]
    fn checked(lhs: T, rhs: T) -> Option<T> {
        lhs.checked_div(rhs)
    }
}

/// Execute a numeric operation between two arrays.
pub(crate) fn execute_numeric(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: NumericOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    match op {
        NumericOperator::Add => execute_checked_numeric::<CheckedAdd>(lhs, rhs, ctx),
        NumericOperator::Sub => execute_checked_numeric::<CheckedSub>(lhs, rhs, ctx),
        NumericOperator::Mul => execute_checked_numeric::<CheckedMul>(lhs, rhs, ctx),
        NumericOperator::Div => execute_checked_numeric::<CheckedDiv>(lhs, rhs, ctx),
    }
}

fn execute_checked_numeric<Op>(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    Op: CheckedPrimitiveBinaryAll,
{
    let ptype = PType::try_from(lhs.dtype())?;
    if !lhs.dtype().eq_ignore_nullability(rhs.dtype()) {
        vortex_bail!(
            "numeric operator requires matching primitive types, got {} and {}",
            lhs.dtype(),
            rhs.dtype()
        );
    }

    match_each_native_ptype!(ptype, |T| { execute_checked_typed::<T, Op>(lhs, rhs, ctx) })
}

fn execute_checked_typed<T, Op>(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: NativePType,
    Op: CheckedPrimitiveBinary<T>,
    Scalar: From<T>,
    Scalar: From<Option<T>>,
{
    let result_dtype = lhs
        .dtype()
        .with_nullability(lhs.dtype().nullability() | rhs.dtype().nullability());
    let lhs = PrimitiveOperand::<T>::try_new(lhs, ctx)?;
    let rhs = PrimitiveOperand::<T>::try_new(rhs, ctx)?;
    let len = lhs.len();
    if len != rhs.len() {
        vortex_bail!(
            "numeric operator requires equal lengths, got {} and {}",
            len,
            rhs.len()
        );
    }

    let validity = lhs.validity().and(rhs.validity())?;
    let valid_rows = ValidRows::from_validity(&validity, len, ctx)?;
    if valid_rows.is_none() {
        return primitive_result_array::<T>(Buffer::<T>::zeroed(len), validity, &result_dtype);
    }

    let values = match (&lhs, &rhs) {
        (PrimitiveOperand::Array(lhs), PrimitiveOperand::Array(rhs)) => {
            checked_array_array::<T, Op>(lhs.values(), rhs.values(), &valid_rows)?
        }
        (PrimitiveOperand::Array(lhs), PrimitiveOperand::Constant { value: rhs, .. }) => {
            checked_array_constant::<T, Op>(lhs.values(), *rhs, &valid_rows)?
        }
        (PrimitiveOperand::Constant { value: lhs, .. }, PrimitiveOperand::Array(rhs)) => {
            checked_constant_array::<T, Op>(*lhs, rhs.values(), &valid_rows)?
        }
        (
            PrimitiveOperand::Constant { value: lhs, .. },
            PrimitiveOperand::Constant { value: rhs, .. },
        ) => {
            let value = Op::checked(*lhs, *rhs).ok_or_else(|| numeric_error::<Op>())?;
            return Ok(constant_result_array(value, len, &result_dtype));
        }
        (PrimitiveOperand::Null(_), _) | (_, PrimitiveOperand::Null(_)) => Buffer::<T>::zeroed(len),
    };

    primitive_result_array::<T>(values, validity, &result_dtype)
}

fn primitive_result_array<T: NativePType>(
    values: Buffer<T>,
    validity: Validity,
    dtype: &DType,
) -> VortexResult<ArrayRef> {
    let array = PrimitiveArray::new(values, validity).into_array();
    if array.dtype() == dtype {
        return Ok(array);
    }
    array.cast(dtype.clone())
}

fn constant_result_array<T>(value: T, len: usize, dtype: &DType) -> ArrayRef
where
    T: NativePType,
    Scalar: From<T> + From<Option<T>>,
{
    if dtype.is_nullable() {
        ConstantArray::new(Some(value), len).into_array()
    } else {
        ConstantArray::new(value, len).into_array()
    }
}

enum PrimitiveOperand<T: NativePType> {
    Array(TypedPrimitive<T>),
    Constant {
        value: T,
        len: usize,
        validity: Validity,
    },
    Null(usize),
}

impl<T: NativePType> PrimitiveOperand<T> {
    fn try_new(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        if let Some(constant) = array.as_opt::<Constant>() {
            return Ok(
                match constant.scalar().as_primitive().try_typed_value::<T>()? {
                    Some(value) => Self::Constant {
                        value,
                        len: array.len(),
                        validity: if constant.scalar().dtype().is_nullable() {
                            Validity::AllValid
                        } else {
                            Validity::NonNullable
                        },
                    },
                    None => Self::Null(array.len()),
                },
            );
        }

        Ok(Self::Array(TypedPrimitive::new(
            array.clone().execute::<PrimitiveArray>(ctx)?,
        )?))
    }

    fn len(&self) -> usize {
        match self {
            Self::Array(array) => array.values().len(),
            Self::Constant { len, .. } | Self::Null(len) => *len,
        }
    }

    fn validity(&self) -> Validity {
        match self {
            Self::Array(array) => array.validity(),
            Self::Constant { validity, .. } => validity.clone(),
            Self::Null(_) => Validity::AllInvalid,
        }
    }
}

struct TypedPrimitive<T: NativePType> {
    values: Buffer<T>,
    validity: Validity,
}

impl<T: NativePType> TypedPrimitive<T> {
    fn new(array: PrimitiveArray) -> VortexResult<Self> {
        let validity = array.validity()?;
        let values = array.into_buffer::<T>();
        Ok(Self { values, validity })
    }

    fn values(&self) -> &[T] {
        &self.values
    }

    fn validity(&self) -> Validity {
        self.validity.clone()
    }
}

enum ValidRows {
    All,
    Some(Mask),
    None,
}

impl ValidRows {
    fn from_validity(
        validity: &Validity,
        len: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let mask = validity.execute_mask(len, ctx)?;
        Ok(if mask.all_true() {
            Self::All
        } else if mask.all_false() {
            Self::None
        } else {
            Self::Some(mask)
        })
    }

    fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }
}

fn checked_array_array<T, Op>(
    lhs: &[T],
    rhs: &[T],
    valid_rows: &ValidRows,
) -> VortexResult<Buffer<T>>
where
    T: NativePType,
    Op: CheckedPrimitiveBinary<T>,
{
    debug_assert_eq!(lhs.len(), rhs.len());

    match valid_rows {
        ValidRows::All => checked_array_array_all_valid::<T, Op>(lhs, rhs),
        ValidRows::Some(mask) => checked_array_array_masked::<T, Op>(lhs, rhs, mask),
        ValidRows::None => Ok(Buffer::<T>::zeroed(lhs.len())),
    }
}

fn checked_array_constant<T, Op>(
    lhs: &[T],
    rhs: T,
    valid_rows: &ValidRows,
) -> VortexResult<Buffer<T>>
where
    T: NativePType,
    Op: CheckedPrimitiveBinary<T>,
{
    match valid_rows {
        ValidRows::All => checked_array_constant_all_valid::<T, Op>(lhs, rhs),
        ValidRows::Some(mask) => checked_array_constant_masked::<T, Op>(lhs, rhs, mask),
        ValidRows::None => Ok(Buffer::<T>::zeroed(lhs.len())),
    }
}

fn checked_constant_array<T, Op>(
    lhs: T,
    rhs: &[T],
    valid_rows: &ValidRows,
) -> VortexResult<Buffer<T>>
where
    T: NativePType,
    Op: CheckedPrimitiveBinary<T>,
{
    match valid_rows {
        ValidRows::All => checked_constant_array_all_valid::<T, Op>(lhs, rhs),
        ValidRows::Some(mask) => checked_constant_array_masked::<T, Op>(lhs, rhs, mask),
        ValidRows::None => Ok(Buffer::<T>::zeroed(rhs.len())),
    }
}

fn checked_array_array_all_valid<T, Op>(lhs: &[T], rhs: &[T]) -> VortexResult<Buffer<T>>
where
    T: NativePType,
    Op: CheckedPrimitiveBinary<T>,
{
    let mut failed = false;
    let mut values = BufferMut::<T>::zeroed(lhs.len());
    for ((dst, &lhs), &rhs) in values.iter_mut().zip(lhs).zip(rhs) {
        let checked = Op::checked(lhs, rhs);
        let invalid = checked.is_none();
        *dst = checked.unwrap_or_default();
        failed |= invalid;
    }
    check_numeric_error::<Op>(failed)?;
    Ok(values.freeze())
}

fn checked_array_array_masked<T, Op>(
    lhs: &[T],
    rhs: &[T],
    valid_rows: &Mask,
) -> VortexResult<Buffer<T>>
where
    T: NativePType,
    Op: CheckedPrimitiveBinary<T>,
{
    let mut failed = false;
    let mut values = BufferMut::<T>::zeroed(lhs.len());
    for (((dst, &lhs), &rhs), valid) in values.iter_mut().zip(lhs).zip(rhs).zip(valid_rows.iter()) {
        let checked = Op::checked(lhs, rhs);
        let invalid = checked.is_none();
        *dst = checked.unwrap_or_default();
        failed |= invalid & valid;
    }
    check_numeric_error::<Op>(failed)?;
    Ok(values.freeze())
}

fn checked_array_constant_all_valid<T, Op>(lhs: &[T], rhs: T) -> VortexResult<Buffer<T>>
where
    T: NativePType,
    Op: CheckedPrimitiveBinary<T>,
{
    let mut failed = false;
    let mut values = BufferMut::<T>::zeroed(lhs.len());
    for (dst, &lhs) in values.iter_mut().zip(lhs) {
        let checked = Op::checked(lhs, rhs);
        let invalid = checked.is_none();
        *dst = checked.unwrap_or_default();
        failed |= invalid;
    }
    check_numeric_error::<Op>(failed)?;
    Ok(values.freeze())
}

fn checked_array_constant_masked<T, Op>(
    lhs: &[T],
    rhs: T,
    valid_rows: &Mask,
) -> VortexResult<Buffer<T>>
where
    T: NativePType,
    Op: CheckedPrimitiveBinary<T>,
{
    let mut failed = false;
    let mut values = BufferMut::<T>::zeroed(lhs.len());
    for ((dst, &lhs), valid) in values.iter_mut().zip(lhs).zip(valid_rows.iter()) {
        let checked = Op::checked(lhs, rhs);
        let invalid = checked.is_none();
        *dst = checked.unwrap_or_default();
        failed |= invalid & valid;
    }
    check_numeric_error::<Op>(failed)?;
    Ok(values.freeze())
}

fn checked_constant_array_all_valid<T, Op>(lhs: T, rhs: &[T]) -> VortexResult<Buffer<T>>
where
    T: NativePType,
    Op: CheckedPrimitiveBinary<T>,
{
    let mut failed = false;
    let mut values = BufferMut::<T>::zeroed(rhs.len());
    for (dst, &rhs) in values.iter_mut().zip(rhs) {
        let checked = Op::checked(lhs, rhs);
        let invalid = checked.is_none();
        *dst = checked.unwrap_or_default();
        failed |= invalid;
    }
    check_numeric_error::<Op>(failed)?;
    Ok(values.freeze())
}

fn checked_constant_array_masked<T, Op>(
    lhs: T,
    rhs: &[T],
    valid_rows: &Mask,
) -> VortexResult<Buffer<T>>
where
    T: NativePType,
    Op: CheckedPrimitiveBinary<T>,
{
    let mut failed = false;
    let mut values = BufferMut::<T>::zeroed(rhs.len());
    for ((dst, &rhs), valid) in values.iter_mut().zip(rhs).zip(valid_rows.iter()) {
        let checked = Op::checked(lhs, rhs);
        let invalid = checked.is_none();
        *dst = checked.unwrap_or_default();
        failed |= invalid & valid;
    }
    check_numeric_error::<Op>(failed)?;
    Ok(values.freeze())
}

trait CheckedArithmetic: NativePType {
    fn checked_add(self, rhs: Self) -> Option<Self>;
    fn checked_sub(self, rhs: Self) -> Option<Self>;
    fn checked_mul(self, rhs: Self) -> Option<Self>;
    fn checked_div(self, rhs: Self) -> Option<Self>;
}

impl CheckedArithmetic for u8 {
    #[inline(always)]
    fn checked_add(self, rhs: Self) -> Option<Self> {
        let (value, overflow) = self.overflowing_add(rhs);
        (!overflow).then_some(value)
    }

    #[inline(always)]
    fn checked_sub(self, rhs: Self) -> Option<Self> {
        let (value, overflow) = self.overflowing_sub(rhs);
        (!overflow).then_some(value)
    }

    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    fn checked_mul(self, rhs: Self) -> Option<Self> {
        let product = (self as u16) * (rhs as u16);
        (product <= u8::MAX as u16).then_some(product as Self)
    }

    #[inline(always)]
    fn checked_div(self, rhs: Self) -> Option<Self> {
        let invalid = rhs == 0;
        let divisor = if invalid { 1 } else { rhs };
        (!invalid).then_some(self.wrapping_div(divisor))
    }
}

impl CheckedArithmetic for u16 {
    #[inline(always)]
    fn checked_add(self, rhs: Self) -> Option<Self> {
        let (value, overflow) = self.overflowing_add(rhs);
        (!overflow).then_some(value)
    }

    #[inline(always)]
    fn checked_sub(self, rhs: Self) -> Option<Self> {
        let (value, overflow) = self.overflowing_sub(rhs);
        (!overflow).then_some(value)
    }

    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    fn checked_mul(self, rhs: Self) -> Option<Self> {
        let product = (self as u32) * (rhs as u32);
        (product <= u16::MAX as u32).then_some(product as Self)
    }

    #[inline(always)]
    fn checked_div(self, rhs: Self) -> Option<Self> {
        let invalid = rhs == 0;
        let divisor = if invalid { 1 } else { rhs };
        (!invalid).then_some(self.wrapping_div(divisor))
    }
}

impl CheckedArithmetic for u32 {
    #[inline(always)]
    fn checked_add(self, rhs: Self) -> Option<Self> {
        let (value, overflow) = self.overflowing_add(rhs);
        (!overflow).then_some(value)
    }

    #[inline(always)]
    fn checked_sub(self, rhs: Self) -> Option<Self> {
        let (value, overflow) = self.overflowing_sub(rhs);
        (!overflow).then_some(value)
    }

    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    fn checked_mul(self, rhs: Self) -> Option<Self> {
        let product = (self as u64) * (rhs as u64);
        (product <= u32::MAX as u64).then_some(product as Self)
    }

    #[inline(always)]
    fn checked_div(self, rhs: Self) -> Option<Self> {
        let invalid = rhs == 0;
        let divisor = if invalid { 1 } else { rhs };
        (!invalid).then_some(self.wrapping_div(divisor))
    }
}

impl CheckedArithmetic for u64 {
    #[inline(always)]
    fn checked_add(self, rhs: Self) -> Option<Self> {
        let (value, overflow) = self.overflowing_add(rhs);
        (!overflow).then_some(value)
    }

    #[inline(always)]
    fn checked_sub(self, rhs: Self) -> Option<Self> {
        let (value, overflow) = self.overflowing_sub(rhs);
        (!overflow).then_some(value)
    }

    #[inline(always)]
    fn checked_mul(self, rhs: Self) -> Option<Self> {
        let (value, overflow) = self.overflowing_mul(rhs);
        (!overflow).then_some(value)
    }

    #[inline(always)]
    fn checked_div(self, rhs: Self) -> Option<Self> {
        let invalid = rhs == 0;
        let divisor = if invalid { 1 } else { rhs };
        (!invalid).then_some(self.wrapping_div(divisor))
    }
}

impl CheckedArithmetic for i8 {
    #[inline(always)]
    fn checked_add(self, rhs: Self) -> Option<Self> {
        let value = self.wrapping_add(rhs);
        let overflow = ((self ^ value) & (rhs ^ value)) < 0;
        (!overflow).then_some(value)
    }

    #[inline(always)]
    fn checked_sub(self, rhs: Self) -> Option<Self> {
        let value = self.wrapping_sub(rhs);
        let overflow = ((self ^ rhs) & (self ^ value)) < 0;
        (!overflow).then_some(value)
    }

    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    fn checked_mul(self, rhs: Self) -> Option<Self> {
        let product = (self as i16) * (rhs as i16);
        (product >= i8::MIN as i16 && product <= i8::MAX as i16).then_some(product as Self)
    }

    #[inline(always)]
    fn checked_div(self, rhs: Self) -> Option<Self> {
        let div_by_zero = rhs == 0;
        let overflow = self == i8::MIN && rhs == -1;
        let divisor = if div_by_zero { 1 } else { rhs };
        (!(div_by_zero | overflow)).then_some(self.wrapping_div(divisor))
    }
}

impl CheckedArithmetic for i16 {
    #[inline(always)]
    fn checked_add(self, rhs: Self) -> Option<Self> {
        let value = self.wrapping_add(rhs);
        let overflow = ((self ^ value) & (rhs ^ value)) < 0;
        (!overflow).then_some(value)
    }

    #[inline(always)]
    fn checked_sub(self, rhs: Self) -> Option<Self> {
        let value = self.wrapping_sub(rhs);
        let overflow = ((self ^ rhs) & (self ^ value)) < 0;
        (!overflow).then_some(value)
    }

    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    fn checked_mul(self, rhs: Self) -> Option<Self> {
        let product = (self as i32) * (rhs as i32);
        (product >= i16::MIN as i32 && product <= i16::MAX as i32).then_some(product as Self)
    }

    #[inline(always)]
    fn checked_div(self, rhs: Self) -> Option<Self> {
        let div_by_zero = rhs == 0;
        let overflow = self == i16::MIN && rhs == -1;
        let divisor = if div_by_zero { 1 } else { rhs };
        (!(div_by_zero | overflow)).then_some(self.wrapping_div(divisor))
    }
}

impl CheckedArithmetic for i32 {
    #[inline(always)]
    fn checked_add(self, rhs: Self) -> Option<Self> {
        let value = self.wrapping_add(rhs);
        let overflow = ((self ^ value) & (rhs ^ value)) < 0;
        (!overflow).then_some(value)
    }

    #[inline(always)]
    fn checked_sub(self, rhs: Self) -> Option<Self> {
        let value = self.wrapping_sub(rhs);
        let overflow = ((self ^ rhs) & (self ^ value)) < 0;
        (!overflow).then_some(value)
    }

    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    fn checked_mul(self, rhs: Self) -> Option<Self> {
        let product = (self as i64) * (rhs as i64);
        (product >= i32::MIN as i64 && product <= i32::MAX as i64).then_some(product as Self)
    }

    #[inline(always)]
    fn checked_div(self, rhs: Self) -> Option<Self> {
        let div_by_zero = rhs == 0;
        let overflow = self == i32::MIN && rhs == -1;
        let divisor = if div_by_zero { 1 } else { rhs };
        (!(div_by_zero | overflow)).then_some(self.wrapping_div(divisor))
    }
}

impl CheckedArithmetic for i64 {
    #[inline(always)]
    fn checked_add(self, rhs: Self) -> Option<Self> {
        let value = self.wrapping_add(rhs);
        let overflow = ((self ^ value) & (rhs ^ value)) < 0;
        (!overflow).then_some(value)
    }

    #[inline(always)]
    fn checked_sub(self, rhs: Self) -> Option<Self> {
        let value = self.wrapping_sub(rhs);
        let overflow = ((self ^ rhs) & (self ^ value)) < 0;
        (!overflow).then_some(value)
    }

    #[inline(always)]
    fn checked_mul(self, rhs: Self) -> Option<Self> {
        let (value, overflow) = self.overflowing_mul(rhs);
        (!overflow).then_some(value)
    }

    #[inline(always)]
    fn checked_div(self, rhs: Self) -> Option<Self> {
        let div_by_zero = rhs == 0;
        let overflow = self == i64::MIN && rhs == -1;
        let divisor = if div_by_zero { 1 } else { rhs };
        (!(div_by_zero | overflow)).then_some(self.wrapping_div(divisor))
    }
}

impl CheckedArithmetic for f16 {
    #[inline(always)]
    fn checked_add(self, rhs: Self) -> Option<Self> {
        Some(self + rhs)
    }

    #[inline(always)]
    fn checked_sub(self, rhs: Self) -> Option<Self> {
        Some(self - rhs)
    }

    #[inline(always)]
    fn checked_mul(self, rhs: Self) -> Option<Self> {
        Some(self * rhs)
    }

    #[inline(always)]
    fn checked_div(self, rhs: Self) -> Option<Self> {
        Some(self / rhs)
    }
}

impl CheckedArithmetic for f32 {
    #[inline(always)]
    fn checked_add(self, rhs: Self) -> Option<Self> {
        Some(self + rhs)
    }

    #[inline(always)]
    fn checked_sub(self, rhs: Self) -> Option<Self> {
        Some(self - rhs)
    }

    #[inline(always)]
    fn checked_mul(self, rhs: Self) -> Option<Self> {
        Some(self * rhs)
    }

    #[inline(always)]
    fn checked_div(self, rhs: Self) -> Option<Self> {
        Some(self / rhs)
    }
}

impl CheckedArithmetic for f64 {
    #[inline(always)]
    fn checked_add(self, rhs: Self) -> Option<Self> {
        Some(self + rhs)
    }

    #[inline(always)]
    fn checked_sub(self, rhs: Self) -> Option<Self> {
        Some(self - rhs)
    }

    #[inline(always)]
    fn checked_mul(self, rhs: Self) -> Option<Self> {
        Some(self * rhs)
    }

    #[inline(always)]
    fn checked_div(self, rhs: Self) -> Option<Self> {
        Some(self / rhs)
    }
}

fn check_numeric_error<Op: CheckedPrimitiveOp>(failed: bool) -> VortexResult<()> {
    if failed {
        return Err(numeric_error::<Op>());
    }
    Ok(())
}

fn numeric_error<Op: CheckedPrimitiveOp>() -> VortexError {
    vortex_err!(InvalidArgument: "{}", Op::ERROR)
}

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::RecursiveCanonical;
    use crate::VortexSessionExecute;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::operators::Operator;
    use crate::validity::Validity;

    fn sub_scalar(array: &ArrayRef, scalar: impl Into<Scalar>) -> VortexResult<ArrayRef> {
        array
            .binary(
                ConstantArray::new(scalar, array.len()).into_array(),
                Operator::Sub,
            )
            .and_then(|a| {
                a.execute::<RecursiveCanonical>(&mut LEGACY_SESSION.create_execution_ctx())
            })
            .map(|a| a.0.into_array())
    }

    #[test]
    fn test_scalar_subtract_unsigned() {
        let values = buffer![1u16, 2, 3].into_array();
        let result = sub_scalar(&values, 1u16).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([0u16, 1, 2]));
    }

    #[test]
    fn test_scalar_subtract_signed() {
        let values = buffer![1i64, 2, 3].into_array();
        let result = sub_scalar(&values, -1i64).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([2i64, 3, 4]));
    }

    #[test]
    fn test_scalar_subtract_nullable() {
        let values = PrimitiveArray::from_option_iter([Some(1u16), Some(2), None, Some(3)]);
        let result = sub_scalar(&values.into_array(), Some(1u16)).unwrap();
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(0u16), Some(1), None, Some(2)])
        );
    }

    #[test]
    fn test_scalar_subtract_float() {
        let values = buffer![1.0f64, 2.0, 3.0].into_array();
        let result = sub_scalar(&values, -1f64).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([2.0f64, 3.0, 4.0]));
    }

    #[test]
    fn test_scalar_subtract_float_underflow_is_ok() {
        let values = buffer![f32::MIN, 2.0, 3.0].into_array();
        let _results = sub_scalar(&values, 1.0f32).unwrap();
        let _results = sub_scalar(&values, f32::MAX).unwrap();
    }

    #[test]
    fn test_integer_overflow_errors() {
        let values = buffer![u8::MAX].into_array();
        let result = values
            .binary(
                ConstantArray::new(1u8, values.len()).into_array(),
                Operator::Add,
            )
            .and_then(|a| a.execute::<PrimitiveArray>(&mut LEGACY_SESSION.create_execution_ctx()));

        assert!(result.is_err());
    }

    #[test]
    fn test_integer_divide_by_zero_errors() {
        let values = buffer![1i32].into_array();
        let result = values
            .binary(
                ConstantArray::new(0i32, values.len()).into_array(),
                Operator::Div,
            )
            .and_then(|a| a.execute::<PrimitiveArray>(&mut LEGACY_SESSION.create_execution_ctx()));

        assert!(result.is_err());
    }

    #[test]
    fn test_integer_errors_ignore_null_lanes() {
        let values = PrimitiveArray::new(buffer![u8::MAX, 1], Validity::from_iter([false, true]))
            .into_array();
        let result = values
            .binary(
                ConstantArray::new(1u8, values.len()).into_array(),
                Operator::Add,
            )
            .and_then(|a| {
                a.execute::<RecursiveCanonical>(&mut LEGACY_SESSION.create_execution_ctx())
            })
            .map(|a| a.0.into_array())
            .unwrap();

        assert_arrays_eq!(result, PrimitiveArray::from_option_iter([None, Some(2u8)]));
    }

    #[test]
    fn test_present_nullable_constant_preserves_nullable_output() {
        let values = buffer![1u8, 2].into_array();
        let result = values
            .binary(
                ConstantArray::new(Some(1u8), values.len()).into_array(),
                Operator::Add,
            )
            .and_then(|a| a.execute::<PrimitiveArray>(&mut LEGACY_SESSION.create_execution_ctx()))
            .unwrap();

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(2u8), Some(3)])
        );
    }
}
