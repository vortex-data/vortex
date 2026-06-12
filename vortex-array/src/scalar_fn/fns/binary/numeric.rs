// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
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
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::dtype::half::f16;
use crate::match_each_native_ptype;
use crate::scalar::NumericOperator;
use crate::validity::Validity;

/// Execute a numeric operation between two arrays.
///
/// This is the entry point for numeric operations from the binary expression. The implementation
/// keeps constants scalar, canonicalizes non-constant inputs to primitive buffers, and accumulates
/// integer arithmetic failures before returning a single operation-level error.
pub(crate) fn execute_numeric(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: NumericOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if let Some(result) = constant_numeric(lhs, rhs, op)? {
        return Ok(result);
    }

    native_numeric(lhs, rhs, op, ctx)
}

fn native_numeric(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: NumericOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let ptype = PType::try_from(lhs.dtype())?;
    if !lhs.dtype().eq_ignore_nullability(rhs.dtype()) {
        vortex_bail!(
            "numeric operator {} requires matching primitive types, got {} and {}",
            op,
            lhs.dtype(),
            rhs.dtype()
        );
    }

    match_each_native_ptype!(ptype, |T| { execute_numeric_typed::<T>(lhs, rhs, op, ctx) })
}

fn execute_numeric_typed<T: NativeNumeric>(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: NumericOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let lhs = NumericOperand::<T>::try_new(lhs, ctx)?;
    let rhs = NumericOperand::<T>::try_new(rhs, ctx)?;
    let len = lhs.len();
    if len != rhs.len() {
        vortex_bail!(
            "numeric operator {} requires equal lengths, got {} and {}",
            op,
            len,
            rhs.len()
        );
    }

    let validity = lhs.validity().and(rhs.validity())?;
    let valid_rows = ValidRows::from_validity(&validity, len, ctx)?;
    if valid_rows.is_none() {
        return Ok(PrimitiveArray::new(Buffer::<T>::zeroed(len), validity).into_array());
    }

    let values = match (&lhs, &rhs) {
        (NumericOperand::Array(lhs), NumericOperand::Array(rhs)) => {
            T::apply_array_array(lhs.values(), rhs.values(), op, &valid_rows)?
        }
        (NumericOperand::Array(lhs), NumericOperand::Constant { value: rhs, .. }) => {
            T::apply_array_constant(lhs.values(), *rhs, op, &valid_rows)?
        }
        (NumericOperand::Constant { value: lhs, .. }, NumericOperand::Array(rhs)) => {
            T::apply_constant_array(*lhs, rhs.values(), op, &valid_rows)?
        }
        (
            NumericOperand::Constant { value: lhs, .. },
            NumericOperand::Constant { value: rhs, .. },
        ) => BufferMut::full(T::apply_scalar(*lhs, *rhs, op)?, len).freeze(),
        (NumericOperand::Null(_), _) | (_, NumericOperand::Null(_)) => Buffer::<T>::zeroed(len),
    };

    Ok(PrimitiveArray::new(values, validity).into_array())
}

fn constant_numeric(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: NumericOperator,
) -> VortexResult<Option<ArrayRef>> {
    let (Some(lhs), Some(rhs)) = (lhs.as_opt::<Constant>(), rhs.as_opt::<Constant>()) else {
        return Ok(None);
    };

    let result = lhs
        .scalar()
        .as_primitive()
        .checked_binary_numeric(&rhs.scalar().as_primitive(), op)
        .ok_or_else(|| numeric_error(op))?;

    Ok(Some(ConstantArray::new(result, lhs.len()).into_array()))
}

enum NumericOperand<T: NativePType> {
    Array(TypedPrimitive<T>),
    Constant {
        value: T,
        len: usize,
        validity: Validity,
    },
    Null(usize),
}

impl<T: NativePType> NumericOperand<T> {
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

trait NativeNumeric: NativePType + Sized {
    fn apply_array_array(
        lhs: &[Self],
        rhs: &[Self],
        op: NumericOperator,
        valid_rows: &ValidRows,
    ) -> VortexResult<Buffer<Self>>;

    fn apply_array_constant(
        lhs: &[Self],
        rhs: Self,
        op: NumericOperator,
        valid_rows: &ValidRows,
    ) -> VortexResult<Buffer<Self>>;

    fn apply_constant_array(
        lhs: Self,
        rhs: &[Self],
        op: NumericOperator,
        valid_rows: &ValidRows,
    ) -> VortexResult<Buffer<Self>>;

    fn apply_scalar(lhs: Self, rhs: Self, op: NumericOperator) -> VortexResult<Self>;
}

trait OverflowingInteger: NativePType {
    fn overflowing_add(self, rhs: Self) -> (Self, bool);
    fn overflowing_sub(self, rhs: Self) -> (Self, bool);
    fn overflowing_mul(self, rhs: Self) -> (Self, bool);
    fn overflowing_div(self, rhs: Self) -> (Self, bool);
}

trait IntegerOp<T: OverflowingInteger> {
    fn apply(lhs: T, rhs: T) -> (T, bool);
}

struct AddOp;
struct SubOp;
struct MulOp;
struct DivOp;

impl<T: OverflowingInteger> IntegerOp<T> for AddOp {
    #[inline(always)]
    fn apply(lhs: T, rhs: T) -> (T, bool) {
        lhs.overflowing_add(rhs)
    }
}

impl<T: OverflowingInteger> IntegerOp<T> for SubOp {
    #[inline(always)]
    fn apply(lhs: T, rhs: T) -> (T, bool) {
        lhs.overflowing_sub(rhs)
    }
}

impl<T: OverflowingInteger> IntegerOp<T> for MulOp {
    #[inline(always)]
    fn apply(lhs: T, rhs: T) -> (T, bool) {
        lhs.overflowing_mul(rhs)
    }
}

impl<T: OverflowingInteger> IntegerOp<T> for DivOp {
    #[inline(always)]
    fn apply(lhs: T, rhs: T) -> (T, bool) {
        lhs.overflowing_div(rhs)
    }
}

trait FloatingOp<T: NativePType> {
    fn apply(lhs: T, rhs: T) -> T;
}

impl<T> FloatingOp<T> for AddOp
where
    T: NativePType + std::ops::Add<Output = T>,
{
    #[inline(always)]
    fn apply(lhs: T, rhs: T) -> T {
        lhs + rhs
    }
}

impl<T> FloatingOp<T> for SubOp
where
    T: NativePType + std::ops::Sub<Output = T>,
{
    #[inline(always)]
    fn apply(lhs: T, rhs: T) -> T {
        lhs - rhs
    }
}

impl<T> FloatingOp<T> for MulOp
where
    T: NativePType + std::ops::Mul<Output = T>,
{
    #[inline(always)]
    fn apply(lhs: T, rhs: T) -> T {
        lhs * rhs
    }
}

impl<T> FloatingOp<T> for DivOp
where
    T: NativePType + std::ops::Div<Output = T>,
{
    #[inline(always)]
    fn apply(lhs: T, rhs: T) -> T {
        lhs / rhs
    }
}

macro_rules! impl_integer_numeric {
    ($($ty:ty),* $(,)?) => {
        $(
            impl NativeNumeric for $ty {
                fn apply_array_array(
                    lhs: &[Self],
                    rhs: &[Self],
                    op: NumericOperator,
                    valid_rows: &ValidRows,
                ) -> VortexResult<Buffer<Self>> {
                    integer_array_array(lhs, rhs, op, valid_rows)
                }

                fn apply_array_constant(
                    lhs: &[Self],
                    rhs: Self,
                    op: NumericOperator,
                    valid_rows: &ValidRows,
                ) -> VortexResult<Buffer<Self>> {
                    integer_array_constant(lhs, rhs, op, valid_rows)
                }

                fn apply_constant_array(
                    lhs: Self,
                    rhs: &[Self],
                    op: NumericOperator,
                    valid_rows: &ValidRows,
                ) -> VortexResult<Buffer<Self>> {
                    integer_constant_array(lhs, rhs, op, valid_rows)
                }

                fn apply_scalar(lhs: Self, rhs: Self, op: NumericOperator) -> VortexResult<Self> {
                    integer_scalar(lhs, rhs, op)
                }
            }
        )*
    };
}

macro_rules! impl_signed_integer_div {
    ($($ty:ty),* $(,)?) => {
        $(
            impl OverflowingInteger for $ty {
                #[inline(always)]
                fn overflowing_add(self, rhs: Self) -> (Self, bool) {
                    let result = self.wrapping_add(rhs);
                    let overflow = ((self ^ result) & (rhs ^ result)) < 0;
                    (result, overflow)
                }

                #[inline(always)]
                fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
                    let result = self.wrapping_sub(rhs);
                    let overflow = ((self ^ rhs) & (self ^ result)) < 0;
                    (result, overflow)
                }

                #[inline(always)]
                fn overflowing_mul(self, rhs: Self) -> (Self, bool) {
                    self.overflowing_mul(rhs)
                }

                #[inline(always)]
                fn overflowing_div(self, rhs: Self) -> (Self, bool) {
                    let div_by_zero = rhs == 0;
                    let overflow = self == <$ty>::MIN && rhs == -1;
                    let divisor = if div_by_zero { 1 } else { rhs };
                    (self.wrapping_div(divisor), div_by_zero | overflow)
                }
            }
        )*
    };
}

macro_rules! impl_signed_widening_integer_div {
    ($($ty:ty => $wide:ty),* $(,)?) => {
        $(
            impl OverflowingInteger for $ty {
                #[inline(always)]
                fn overflowing_add(self, rhs: Self) -> (Self, bool) {
                    let result = self.wrapping_add(rhs);
                    let overflow = ((self ^ result) & (rhs ^ result)) < 0;
                    (result, overflow)
                }

                #[inline(always)]
                fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
                    let result = self.wrapping_sub(rhs);
                    let overflow = ((self ^ rhs) & (self ^ result)) < 0;
                    (result, overflow)
                }

                #[inline(always)]
                #[allow(clippy::cast_possible_truncation)]
                fn overflowing_mul(self, rhs: Self) -> (Self, bool) {
                    let product = (self as $wide) * (rhs as $wide);
                    (
                        product as Self,
                        product < <$ty>::MIN as $wide || product > <$ty>::MAX as $wide,
                    )
                }

                #[inline(always)]
                fn overflowing_div(self, rhs: Self) -> (Self, bool) {
                    let div_by_zero = rhs == 0;
                    let overflow = self == <$ty>::MIN && rhs == -1;
                    let divisor = if div_by_zero { 1 } else { rhs };
                    (self.wrapping_div(divisor), div_by_zero | overflow)
                }
            }
        )*
    };
}

macro_rules! impl_unsigned_integer_div {
    ($($ty:ty),* $(,)?) => {
        $(
            impl OverflowingInteger for $ty {
                #[inline(always)]
                fn overflowing_add(self, rhs: Self) -> (Self, bool) {
                    self.overflowing_add(rhs)
                }

                #[inline(always)]
                fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
                    self.overflowing_sub(rhs)
                }

                #[inline(always)]
                fn overflowing_mul(self, rhs: Self) -> (Self, bool) {
                    self.overflowing_mul(rhs)
                }

                #[inline(always)]
                fn overflowing_div(self, rhs: Self) -> (Self, bool) {
                    let div_by_zero = rhs == 0;
                    let divisor = if div_by_zero { 1 } else { rhs };
                    (self.wrapping_div(divisor), div_by_zero)
                }
            }
        )*
    };
}

macro_rules! impl_unsigned_widening_integer_div {
    ($($ty:ty => $wide:ty),* $(,)?) => {
        $(
            impl OverflowingInteger for $ty {
                #[inline(always)]
                fn overflowing_add(self, rhs: Self) -> (Self, bool) {
                    self.overflowing_add(rhs)
                }

                #[inline(always)]
                fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
                    self.overflowing_sub(rhs)
                }

                #[inline(always)]
                #[allow(clippy::cast_possible_truncation)]
                fn overflowing_mul(self, rhs: Self) -> (Self, bool) {
                    let product = (self as $wide) * (rhs as $wide);
                    (product as Self, product > <$ty>::MAX as $wide)
                }

                #[inline(always)]
                fn overflowing_div(self, rhs: Self) -> (Self, bool) {
                    let div_by_zero = rhs == 0;
                    let divisor = if div_by_zero { 1 } else { rhs };
                    (self.wrapping_div(divisor), div_by_zero)
                }
            }
        )*
    };
}

macro_rules! impl_floating_numeric {
    ($($ty:ty),* $(,)?) => {
        $(
            impl NativeNumeric for $ty {
                fn apply_array_array(
                    lhs: &[Self],
                    rhs: &[Self],
                    op: NumericOperator,
                    _valid_rows: &ValidRows,
                ) -> VortexResult<Buffer<Self>> {
                    Ok(floating_array_array(lhs, rhs, op))
                }

                fn apply_array_constant(
                    lhs: &[Self],
                    rhs: Self,
                    op: NumericOperator,
                    _valid_rows: &ValidRows,
                ) -> VortexResult<Buffer<Self>> {
                    Ok(floating_array_constant(lhs, rhs, op))
                }

                fn apply_constant_array(
                    lhs: Self,
                    rhs: &[Self],
                    op: NumericOperator,
                    _valid_rows: &ValidRows,
                ) -> VortexResult<Buffer<Self>> {
                    Ok(floating_constant_array(lhs, rhs, op))
                }

                fn apply_scalar(lhs: Self, rhs: Self, op: NumericOperator) -> VortexResult<Self> {
                    Ok(floating_scalar(lhs, rhs, op))
                }
            }
        )*
    };
}

impl_unsigned_widening_integer_div!(u8 => u16, u16 => u32, u32 => u64);
impl_unsigned_integer_div!(u64);
impl_signed_widening_integer_div!(i8 => i16, i16 => i32, i32 => i64);
impl_signed_integer_div!(i64);
impl_integer_numeric!(u8, u16, u32, u64, i8, i16, i32, i64);
impl_floating_numeric!(f16, f32, f64);

fn integer_array_array<T: OverflowingInteger>(
    lhs: &[T],
    rhs: &[T],
    op: NumericOperator,
    valid_rows: &ValidRows,
) -> VortexResult<Buffer<T>> {
    match op {
        NumericOperator::Add => integer_array_array_op::<T, AddOp>(lhs, rhs, op, valid_rows),
        NumericOperator::Sub => integer_array_array_op::<T, SubOp>(lhs, rhs, op, valid_rows),
        NumericOperator::Mul => integer_array_array_op::<T, MulOp>(lhs, rhs, op, valid_rows),
        NumericOperator::Div => integer_array_array_op::<T, DivOp>(lhs, rhs, op, valid_rows),
    }
}

fn integer_array_constant<T: OverflowingInteger>(
    lhs: &[T],
    rhs: T,
    op: NumericOperator,
    valid_rows: &ValidRows,
) -> VortexResult<Buffer<T>> {
    match op {
        NumericOperator::Add => integer_array_constant_op::<T, AddOp>(lhs, rhs, op, valid_rows),
        NumericOperator::Sub => integer_array_constant_op::<T, SubOp>(lhs, rhs, op, valid_rows),
        NumericOperator::Mul => integer_array_constant_op::<T, MulOp>(lhs, rhs, op, valid_rows),
        NumericOperator::Div => integer_array_constant_op::<T, DivOp>(lhs, rhs, op, valid_rows),
    }
}

fn integer_constant_array<T: OverflowingInteger>(
    lhs: T,
    rhs: &[T],
    op: NumericOperator,
    valid_rows: &ValidRows,
) -> VortexResult<Buffer<T>> {
    match op {
        NumericOperator::Add => integer_constant_array_op::<T, AddOp>(lhs, rhs, op, valid_rows),
        NumericOperator::Sub => integer_constant_array_op::<T, SubOp>(lhs, rhs, op, valid_rows),
        NumericOperator::Mul => integer_constant_array_op::<T, MulOp>(lhs, rhs, op, valid_rows),
        NumericOperator::Div => integer_constant_array_op::<T, DivOp>(lhs, rhs, op, valid_rows),
    }
}

fn integer_scalar<T: OverflowingInteger>(lhs: T, rhs: T, op: NumericOperator) -> VortexResult<T> {
    match op {
        NumericOperator::Add => integer_scalar_op::<T, AddOp>(lhs, rhs, op),
        NumericOperator::Sub => integer_scalar_op::<T, SubOp>(lhs, rhs, op),
        NumericOperator::Mul => integer_scalar_op::<T, MulOp>(lhs, rhs, op),
        NumericOperator::Div => integer_scalar_op::<T, DivOp>(lhs, rhs, op),
    }
}

fn integer_array_array_op<T, Op>(
    lhs: &[T],
    rhs: &[T],
    op: NumericOperator,
    valid_rows: &ValidRows,
) -> VortexResult<Buffer<T>>
where
    T: OverflowingInteger,
    Op: IntegerOp<T>,
{
    debug_assert_eq!(lhs.len(), rhs.len());

    match valid_rows {
        ValidRows::All => integer_array_array_all_valid::<T, Op>(lhs, rhs, op),
        ValidRows::Some(mask) => integer_array_array_masked::<T, Op>(lhs, rhs, op, mask),
        ValidRows::None => Ok(Buffer::<T>::zeroed(lhs.len())),
    }
}

fn integer_array_constant_op<T, Op>(
    lhs: &[T],
    rhs: T,
    op: NumericOperator,
    valid_rows: &ValidRows,
) -> VortexResult<Buffer<T>>
where
    T: OverflowingInteger,
    Op: IntegerOp<T>,
{
    match valid_rows {
        ValidRows::All => integer_array_constant_all_valid::<T, Op>(lhs, rhs, op),
        ValidRows::Some(mask) => integer_array_constant_masked::<T, Op>(lhs, rhs, op, mask),
        ValidRows::None => Ok(Buffer::<T>::zeroed(lhs.len())),
    }
}

fn integer_constant_array_op<T, Op>(
    lhs: T,
    rhs: &[T],
    op: NumericOperator,
    valid_rows: &ValidRows,
) -> VortexResult<Buffer<T>>
where
    T: OverflowingInteger,
    Op: IntegerOp<T>,
{
    match valid_rows {
        ValidRows::All => integer_constant_array_all_valid::<T, Op>(lhs, rhs, op),
        ValidRows::Some(mask) => integer_constant_array_masked::<T, Op>(lhs, rhs, op, mask),
        ValidRows::None => Ok(Buffer::<T>::zeroed(rhs.len())),
    }
}

fn integer_scalar_op<T, Op>(lhs: T, rhs: T, op: NumericOperator) -> VortexResult<T>
where
    T: OverflowingInteger,
    Op: IntegerOp<T>,
{
    let (value, failed) = Op::apply(lhs, rhs);
    check_numeric_error(op, failed)?;
    Ok(value)
}

fn integer_array_array_all_valid<T, Op>(
    lhs: &[T],
    rhs: &[T],
    op: NumericOperator,
) -> VortexResult<Buffer<T>>
where
    T: OverflowingInteger,
    Op: IntegerOp<T>,
{
    let mut failed = false;
    let mut values = BufferMut::<T>::zeroed(lhs.len());
    for ((dst, &lhs), &rhs) in values.iter_mut().zip(lhs).zip(rhs) {
        let (value, error) = Op::apply(lhs, rhs);
        *dst = value;
        failed |= error;
    }
    check_numeric_error(op, failed)?;
    Ok(values.freeze())
}

fn integer_array_array_masked<T, Op>(
    lhs: &[T],
    rhs: &[T],
    op: NumericOperator,
    valid_rows: &Mask,
) -> VortexResult<Buffer<T>>
where
    T: OverflowingInteger,
    Op: IntegerOp<T>,
{
    let mut failed = false;
    let mut values = BufferMut::<T>::zeroed(lhs.len());
    for (((dst, &lhs), &rhs), valid) in values.iter_mut().zip(lhs).zip(rhs).zip(valid_rows.iter()) {
        let (value, error) = Op::apply(lhs, rhs);
        *dst = value;
        failed |= error & valid;
    }
    check_numeric_error(op, failed)?;
    Ok(values.freeze())
}

fn integer_array_constant_all_valid<T, Op>(
    lhs: &[T],
    rhs: T,
    op: NumericOperator,
) -> VortexResult<Buffer<T>>
where
    T: OverflowingInteger,
    Op: IntegerOp<T>,
{
    let mut failed = false;
    let mut values = BufferMut::<T>::zeroed(lhs.len());
    for (dst, &lhs) in values.iter_mut().zip(lhs) {
        let (value, error) = Op::apply(lhs, rhs);
        *dst = value;
        failed |= error;
    }
    check_numeric_error(op, failed)?;
    Ok(values.freeze())
}

fn integer_array_constant_masked<T, Op>(
    lhs: &[T],
    rhs: T,
    op: NumericOperator,
    valid_rows: &Mask,
) -> VortexResult<Buffer<T>>
where
    T: OverflowingInteger,
    Op: IntegerOp<T>,
{
    let mut failed = false;
    let mut values = BufferMut::<T>::zeroed(lhs.len());
    for ((dst, &lhs), valid) in values.iter_mut().zip(lhs).zip(valid_rows.iter()) {
        let (value, error) = Op::apply(lhs, rhs);
        *dst = value;
        failed |= error & valid;
    }
    check_numeric_error(op, failed)?;
    Ok(values.freeze())
}

fn integer_constant_array_all_valid<T, Op>(
    lhs: T,
    rhs: &[T],
    op: NumericOperator,
) -> VortexResult<Buffer<T>>
where
    T: OverflowingInteger,
    Op: IntegerOp<T>,
{
    let mut failed = false;
    let mut values = BufferMut::<T>::zeroed(rhs.len());
    for (dst, &rhs) in values.iter_mut().zip(rhs) {
        let (value, error) = Op::apply(lhs, rhs);
        *dst = value;
        failed |= error;
    }
    check_numeric_error(op, failed)?;
    Ok(values.freeze())
}

fn integer_constant_array_masked<T, Op>(
    lhs: T,
    rhs: &[T],
    op: NumericOperator,
    valid_rows: &Mask,
) -> VortexResult<Buffer<T>>
where
    T: OverflowingInteger,
    Op: IntegerOp<T>,
{
    let mut failed = false;
    let mut values = BufferMut::<T>::zeroed(rhs.len());
    for ((dst, &rhs), valid) in values.iter_mut().zip(rhs).zip(valid_rows.iter()) {
        let (value, error) = Op::apply(lhs, rhs);
        *dst = value;
        failed |= error & valid;
    }
    check_numeric_error(op, failed)?;
    Ok(values.freeze())
}

fn floating_array_array<T>(lhs: &[T], rhs: &[T], op: NumericOperator) -> Buffer<T>
where
    T: NativePType
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    match op {
        NumericOperator::Add => floating_array_array_op::<T, AddOp>(lhs, rhs),
        NumericOperator::Sub => floating_array_array_op::<T, SubOp>(lhs, rhs),
        NumericOperator::Mul => floating_array_array_op::<T, MulOp>(lhs, rhs),
        NumericOperator::Div => floating_array_array_op::<T, DivOp>(lhs, rhs),
    }
}

fn floating_array_constant<T>(lhs: &[T], rhs: T, op: NumericOperator) -> Buffer<T>
where
    T: NativePType
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    match op {
        NumericOperator::Add => floating_array_constant_op::<T, AddOp>(lhs, rhs),
        NumericOperator::Sub => floating_array_constant_op::<T, SubOp>(lhs, rhs),
        NumericOperator::Mul => floating_array_constant_op::<T, MulOp>(lhs, rhs),
        NumericOperator::Div => floating_array_constant_op::<T, DivOp>(lhs, rhs),
    }
}

fn floating_constant_array<T>(lhs: T, rhs: &[T], op: NumericOperator) -> Buffer<T>
where
    T: NativePType
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    match op {
        NumericOperator::Add => floating_constant_array_op::<T, AddOp>(lhs, rhs),
        NumericOperator::Sub => floating_constant_array_op::<T, SubOp>(lhs, rhs),
        NumericOperator::Mul => floating_constant_array_op::<T, MulOp>(lhs, rhs),
        NumericOperator::Div => floating_constant_array_op::<T, DivOp>(lhs, rhs),
    }
}

fn floating_scalar<T>(lhs: T, rhs: T, op: NumericOperator) -> T
where
    T: NativePType
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    match op {
        NumericOperator::Add => <AddOp as FloatingOp<T>>::apply(lhs, rhs),
        NumericOperator::Sub => <SubOp as FloatingOp<T>>::apply(lhs, rhs),
        NumericOperator::Mul => <MulOp as FloatingOp<T>>::apply(lhs, rhs),
        NumericOperator::Div => <DivOp as FloatingOp<T>>::apply(lhs, rhs),
    }
}

fn floating_array_array_op<T, Op>(lhs: &[T], rhs: &[T]) -> Buffer<T>
where
    T: NativePType,
    Op: FloatingOp<T>,
{
    debug_assert_eq!(lhs.len(), rhs.len());

    let mut values = BufferMut::<T>::zeroed(lhs.len());
    for ((dst, &lhs), &rhs) in values.iter_mut().zip(lhs).zip(rhs) {
        *dst = Op::apply(lhs, rhs);
    }
    values.freeze()
}

fn floating_array_constant_op<T, Op>(lhs: &[T], rhs: T) -> Buffer<T>
where
    T: NativePType,
    Op: FloatingOp<T>,
{
    let mut values = BufferMut::<T>::zeroed(lhs.len());
    for (dst, &lhs) in values.iter_mut().zip(lhs) {
        *dst = Op::apply(lhs, rhs);
    }
    values.freeze()
}

fn floating_constant_array_op<T, Op>(lhs: T, rhs: &[T]) -> Buffer<T>
where
    T: NativePType,
    Op: FloatingOp<T>,
{
    let mut values = BufferMut::<T>::zeroed(rhs.len());
    for (dst, &rhs) in values.iter_mut().zip(rhs) {
        *dst = Op::apply(lhs, rhs);
    }
    values.freeze()
}

fn check_numeric_error(op: NumericOperator, failed: bool) -> VortexResult<()> {
    if failed {
        return Err(numeric_error(op));
    }
    Ok(())
}

fn numeric_error(op: NumericOperator) -> vortex_error::VortexError {
    match op {
        NumericOperator::Add | NumericOperator::Sub | NumericOperator::Mul => {
            vortex_err!(InvalidArgument: "integer overflow in numeric {} operation", op)
        }
        NumericOperator::Div => {
            vortex_err!(InvalidArgument: "integer division by zero or overflow in numeric / operation")
        }
    }
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
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::binary::numeric::ConstantArray;
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
