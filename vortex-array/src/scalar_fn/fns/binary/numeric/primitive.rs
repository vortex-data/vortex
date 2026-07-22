// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::checked::CheckedValues;
use super::checked::any_valid_error;
use super::checked::checked_all_lanes;
use super::checked::checked_valid_lanes;
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

trait CheckedPrimitiveOp<T: NativePType>: Sized {
    const ERROR: &'static str;
    const CHECKED_VALUE_LOOP: bool = false;

    // The vectorizable kernels need an overflowing-style result, not
    // `Option<T>`: every lane can write a value unconditionally while reducing
    // the error flag separately. `checked` is still available for scalar and
    // one-pass paths where early exit is the right shape.
    fn apply(lhs: T, rhs: T) -> (T, bool);

    #[inline(always)]
    fn checked(lhs: T, rhs: T) -> Option<T> {
        let (value, failed) = Self::apply(lhs, rhs);
        if failed { None } else { Some(value) }
    }
}

impl<T: CheckedArithmetic> CheckedPrimitiveOp<T> for CheckedAdd {
    const ERROR: &'static str = "integer overflow in checked add";

    #[inline(always)]
    fn apply(lhs: T, rhs: T) -> (T, bool) {
        (lhs.add_value(rhs), lhs.add_error(rhs))
    }
}

impl<T: CheckedArithmetic> CheckedPrimitiveOp<T> for CheckedSub {
    const ERROR: &'static str = "integer overflow in checked sub";

    #[inline(always)]
    fn apply(lhs: T, rhs: T) -> (T, bool) {
        (lhs.sub_value(rhs), lhs.sub_error(rhs))
    }
}

impl<T: CheckedArithmetic> CheckedPrimitiveOp<T> for CheckedMul {
    const ERROR: &'static str = "integer overflow in checked mul";

    #[inline(always)]
    fn apply(lhs: T, rhs: T) -> (T, bool) {
        (lhs.mul_value(rhs), lhs.mul_error(rhs))
    }
}

impl<T: CheckedArithmetic> CheckedPrimitiveOp<T> for CheckedDiv {
    const ERROR: &'static str = "integer division by zero or overflow in checked div";
    // Integer division still lowers to scalar divides, so the split
    // value/error-scan loop used to auto-vectorize add/sub/mul only adds a
    // second full scan. Use the generic branchy checked value loop for integer
    // division, matching Arrow/Velox. Float division has no checked errors and
    // stays on the split/vectorizable default path.
    const CHECKED_VALUE_LOOP: bool = T::DIV_CHECKS_IN_VALUE_LOOP;

    #[inline(always)]
    fn apply(lhs: T, rhs: T) -> (T, bool) {
        let failed = lhs.div_error(rhs);
        let value = if failed {
            T::default()
        } else {
            lhs.div_value(rhs)
        };
        (value, failed)
    }

    #[inline(always)]
    fn checked(lhs: T, rhs: T) -> Option<T> {
        lhs.div_checked(rhs)
    }
}

/// Execute a numeric operation between two primitive-typed arrays.
pub(super) fn execute_numeric_primitive(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: NumericOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let ptype = PType::try_from(lhs.dtype())?;

    match_each_native_ptype!(ptype, |T| {
        match op {
            NumericOperator::Add => execute_checked_typed::<T, CheckedAdd>(lhs, rhs, ctx),
            NumericOperator::Sub => execute_checked_typed::<T, CheckedSub>(lhs, rhs, ctx),
            NumericOperator::Mul => execute_checked_typed::<T, CheckedMul>(lhs, rhs, ctx),
            NumericOperator::Div => execute_checked_typed::<T, CheckedDiv>(lhs, rhs, ctx),
        }
    })
}

fn execute_checked_typed<T, Op>(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: NativePType,
    Op: CheckedPrimitiveOp<T>,
    Scalar: From<T>,
    Scalar: From<Option<T>>,
{
    let result_dtype = lhs
        .dtype()
        .with_nullability(lhs.dtype().nullability() | rhs.dtype().nullability());
    let lhs = PrimitiveOperand::<T>::try_new(lhs, ctx)?;
    let rhs = PrimitiveOperand::<T>::try_new(rhs, ctx)?;
    let len = lhs.len();
    debug_assert_eq!(len, rhs.len());

    let validity = lhs.validity().and(rhs.validity())?;
    let valid_rows = validity.execute_mask(len, ctx)?;

    let checked = match (&lhs, &rhs) {
        (
            PrimitiveOperand::Array { values: lhs, .. },
            PrimitiveOperand::Array { values: rhs, .. },
        ) => checked_array_array::<T, Op>(lhs, rhs, &valid_rows),
        (
            PrimitiveOperand::Array { values: lhs, .. },
            PrimitiveOperand::Constant { value: rhs, .. },
        ) => checked_array_constant::<T, Op>(lhs, *rhs, &valid_rows),
        (
            PrimitiveOperand::Constant { value: lhs, .. },
            PrimitiveOperand::Array { values: rhs, .. },
        ) => checked_constant_array::<T, Op>(*lhs, rhs, &valid_rows),
        (
            PrimitiveOperand::Constant { value: lhs, .. },
            PrimitiveOperand::Constant { value: rhs, .. },
        ) => {
            let value = Op::checked(*lhs, *rhs)
                .ok_or_else(|| vortex_err!(InvalidArgument: "{}", Op::ERROR))?;
            return Ok(constant_result_array(value, len, &result_dtype));
        }
        (PrimitiveOperand::Null(_), _) | (_, PrimitiveOperand::Null(_)) => {
            CheckedValues::zeroed(len)
        }
    };

    if checked.failed {
        return Err(vortex_err!(InvalidArgument: "{}", Op::ERROR));
    }

    primitive_result_array::<T>(checked.values, validity, &result_dtype)
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

/// A primitive binary-operator operand: a materialized buffer, a non-null constant, or an
/// all-null constant.
pub(crate) enum PrimitiveOperand<T: NativePType> {
    Array {
        values: Buffer<T>,
        validity: Validity,
    },
    Constant {
        value: T,
        len: usize,
        validity: Validity,
    },
    Null(usize),
}

impl<T: NativePType> PrimitiveOperand<T> {
    pub(crate) fn try_new(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
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

        let array = array.clone().execute::<PrimitiveArray>(ctx)?;
        let validity = array.validity()?;
        let values = array.into_buffer::<T>();
        Ok(Self::Array { values, validity })
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Array { values, .. } => values.len(),
            Self::Constant { len, .. } | Self::Null(len) => *len,
        }
    }

    pub(crate) fn validity(&self) -> Validity {
        match self {
            Self::Array { validity, .. } => validity.clone(),
            Self::Constant { validity, .. } => validity.clone(),
            Self::Null(_) => Validity::AllInvalid,
        }
    }
}

fn checked_array_array<T, Op>(lhs: &[T], rhs: &[T], valid_rows: &Mask) -> CheckedValues<T>
where
    T: NativePType,
    Op: CheckedPrimitiveOp<T>,
{
    debug_assert_eq!(lhs.len(), rhs.len());

    match valid_rows.bit_buffer() {
        AllOr::All if Op::CHECKED_VALUE_LOOP => checked_array_array_one_pass::<T, Op>(lhs, rhs),
        AllOr::All => checked_array_array_all_lanes::<T, Op>(lhs, rhs),
        AllOr::None => CheckedValues::zeroed(lhs.len()),
        AllOr::Some(valid_bits) if Op::CHECKED_VALUE_LOOP => {
            checked_array_array_valid_lanes_one_pass::<T, Op>(lhs, rhs, valid_bits)
        }
        AllOr::Some(valid_bits) => checked_array_array_valid_lanes::<T, Op>(lhs, rhs, valid_bits),
    }
}

fn checked_array_constant<T, Op>(lhs: &[T], rhs: T, valid_rows: &Mask) -> CheckedValues<T>
where
    T: NativePType,
    Op: CheckedPrimitiveOp<T>,
{
    match valid_rows.bit_buffer() {
        AllOr::All if Op::CHECKED_VALUE_LOOP => checked_array_constant_one_pass::<T, Op>(lhs, rhs),
        AllOr::All => checked_array_constant_all_lanes::<T, Op>(lhs, rhs),
        AllOr::None => CheckedValues::zeroed(lhs.len()),
        AllOr::Some(valid_bits) if Op::CHECKED_VALUE_LOOP => {
            checked_array_constant_valid_lanes_one_pass::<T, Op>(lhs, rhs, valid_bits)
        }
        AllOr::Some(valid_bits) => {
            checked_array_constant_valid_lanes::<T, Op>(lhs, rhs, valid_bits)
        }
    }
}

fn checked_constant_array<T, Op>(lhs: T, rhs: &[T], valid_rows: &Mask) -> CheckedValues<T>
where
    T: NativePType,
    Op: CheckedPrimitiveOp<T>,
{
    match valid_rows.bit_buffer() {
        AllOr::All if Op::CHECKED_VALUE_LOOP => checked_constant_array_one_pass::<T, Op>(lhs, rhs),
        AllOr::All => checked_constant_array_all_lanes::<T, Op>(lhs, rhs),
        AllOr::None => CheckedValues::zeroed(rhs.len()),
        AllOr::Some(valid_bits) if Op::CHECKED_VALUE_LOOP => {
            checked_constant_array_valid_lanes_one_pass::<T, Op>(lhs, rhs, valid_bits)
        }
        AllOr::Some(valid_bits) => {
            checked_constant_array_valid_lanes::<T, Op>(lhs, rhs, valid_bits)
        }
    }
}

fn checked_array_array_all_lanes<T, Op>(lhs: &[T], rhs: &[T]) -> CheckedValues<T>
where
    T: NativePType,
    Op: CheckedPrimitiveOp<T>,
{
    collect_all_lanes(lhs.len(), |idx| Op::apply(lhs[idx], rhs[idx]))
}

fn checked_array_array_valid_lanes<T, Op>(
    lhs: &[T],
    rhs: &[T],
    valid_bits: &BitBuffer,
) -> CheckedValues<T>
where
    T: NativePType,
    Op: CheckedPrimitiveOp<T>,
{
    let mut checked = collect_all_lanes(lhs.len(), |idx| Op::apply(lhs[idx], rhs[idx]));

    checked.failed = checked.failed
        && any_valid_error(lhs.len(), valid_bits, |idx| Op::apply(lhs[idx], rhs[idx]).1);
    checked
}

fn checked_array_constant_all_lanes<T, Op>(lhs: &[T], rhs: T) -> CheckedValues<T>
where
    T: NativePType,
    Op: CheckedPrimitiveOp<T>,
{
    collect_all_lanes(lhs.len(), |idx| Op::apply(lhs[idx], rhs))
}

fn checked_array_constant_valid_lanes<T, Op>(
    lhs: &[T],
    rhs: T,
    valid_bits: &BitBuffer,
) -> CheckedValues<T>
where
    T: NativePType,
    Op: CheckedPrimitiveOp<T>,
{
    let mut checked = collect_all_lanes(lhs.len(), |idx| Op::apply(lhs[idx], rhs));

    checked.failed =
        checked.failed && any_valid_error(lhs.len(), valid_bits, |idx| Op::apply(lhs[idx], rhs).1);
    checked
}

fn checked_constant_array_all_lanes<T, Op>(lhs: T, rhs: &[T]) -> CheckedValues<T>
where
    T: NativePType,
    Op: CheckedPrimitiveOp<T>,
{
    collect_all_lanes(rhs.len(), |idx| Op::apply(lhs, rhs[idx]))
}

fn checked_constant_array_valid_lanes<T, Op>(
    lhs: T,
    rhs: &[T],
    valid_bits: &BitBuffer,
) -> CheckedValues<T>
where
    T: NativePType,
    Op: CheckedPrimitiveOp<T>,
{
    let mut checked = collect_all_lanes(rhs.len(), |idx| Op::apply(lhs, rhs[idx]));

    checked.failed =
        checked.failed && any_valid_error(rhs.len(), valid_bits, |idx| Op::apply(lhs, rhs[idx]).1);
    checked
}

fn collect_all_lanes<T, F>(len: usize, mut value_and_error_at: F) -> CheckedValues<T>
where
    T: NativePType,
    F: FnMut(usize) -> (T, bool),
{
    let mut values = BufferMut::<T>::with_capacity(len);
    let slots = values.spare_capacity_mut().as_mut_ptr();
    let mut failed = false;
    for idx in 0..len {
        let (value, is_error) = value_and_error_at(idx);
        failed |= is_error;
        // SAFETY: `idx < len <= capacity`, and each slot is written once.
        unsafe { slots.add(idx).write(std::mem::MaybeUninit::new(value)) };
    }
    // SAFETY: every slot in `0..len` was initialized above.
    unsafe { values.set_len(len) };
    CheckedValues {
        values: values.freeze(),
        failed,
    }
}

fn checked_array_array_one_pass<T, Op>(lhs: &[T], rhs: &[T]) -> CheckedValues<T>
where
    T: NativePType,
    Op: CheckedPrimitiveOp<T>,
{
    checked_all_lanes(lhs.len(), |idx| Op::checked(lhs[idx], rhs[idx]))
}

fn checked_array_array_valid_lanes_one_pass<T, Op>(
    lhs: &[T],
    rhs: &[T],
    valid_bits: &BitBuffer,
) -> CheckedValues<T>
where
    T: NativePType,
    Op: CheckedPrimitiveOp<T>,
{
    checked_valid_lanes(lhs.len(), valid_bits, |idx| Op::checked(lhs[idx], rhs[idx]))
}

fn checked_array_constant_one_pass<T, Op>(lhs: &[T], rhs: T) -> CheckedValues<T>
where
    T: NativePType,
    Op: CheckedPrimitiveOp<T>,
{
    checked_all_lanes(lhs.len(), |idx| Op::checked(lhs[idx], rhs))
}

fn checked_array_constant_valid_lanes_one_pass<T, Op>(
    lhs: &[T],
    rhs: T,
    valid_bits: &BitBuffer,
) -> CheckedValues<T>
where
    T: NativePType,
    Op: CheckedPrimitiveOp<T>,
{
    checked_valid_lanes(lhs.len(), valid_bits, |idx| Op::checked(lhs[idx], rhs))
}

fn checked_constant_array_one_pass<T, Op>(lhs: T, rhs: &[T]) -> CheckedValues<T>
where
    T: NativePType,
    Op: CheckedPrimitiveOp<T>,
{
    checked_all_lanes(rhs.len(), |idx| Op::checked(lhs, rhs[idx]))
}

fn checked_constant_array_valid_lanes_one_pass<T, Op>(
    lhs: T,
    rhs: &[T],
    valid_bits: &BitBuffer,
) -> CheckedValues<T>
where
    T: NativePType,
    Op: CheckedPrimitiveOp<T>,
{
    checked_valid_lanes(rhs.len(), valid_bits, |idx| Op::checked(lhs, rhs[idx]))
}

trait CheckedArithmetic: NativePType {
    const DIV_CHECKS_IN_VALUE_LOOP: bool;

    fn add_value(self, rhs: Self) -> Self;
    fn add_error(self, rhs: Self) -> bool;
    fn sub_value(self, rhs: Self) -> Self;
    fn sub_error(self, rhs: Self) -> bool;
    fn mul_value(self, rhs: Self) -> Self;
    fn mul_error(self, rhs: Self) -> bool;
    fn div_value(self, rhs: Self) -> Self;
    fn div_error(self, rhs: Self) -> bool;
    fn div_checked(self, rhs: Self) -> Option<Self>;
}

macro_rules! impl_checked_unsigned {
    ($ty:ty,widening_mul: $wide:ty) => {
        impl CheckedArithmetic for $ty {
            const DIV_CHECKS_IN_VALUE_LOOP: bool = true;

            #[inline(always)]
            fn add_value(self, rhs: Self) -> Self {
                self.wrapping_add(rhs)
            }

            #[inline(always)]
            fn add_error(self, rhs: Self) -> bool {
                self > <$ty>::MAX - rhs
            }

            #[inline(always)]
            fn sub_value(self, rhs: Self) -> Self {
                self.wrapping_sub(rhs)
            }

            #[inline(always)]
            fn sub_error(self, rhs: Self) -> bool {
                self < rhs
            }

            #[inline(always)]
            fn mul_value(self, rhs: Self) -> Self {
                self.wrapping_mul(rhs)
            }

            #[inline(always)]
            fn mul_error(self, rhs: Self) -> bool {
                (self as $wide) * (rhs as $wide) > <$ty>::MAX as $wide
            }

            #[inline(always)]
            fn div_value(self, rhs: Self) -> Self {
                self / rhs
            }

            #[inline(always)]
            fn div_error(self, rhs: Self) -> bool {
                rhs == 0
            }

            #[inline(always)]
            fn div_checked(self, rhs: Self) -> Option<Self> {
                self.checked_div(rhs)
            }
        }
    };
    ($ty:ty,overflowing_mul) => {
        impl CheckedArithmetic for $ty {
            const DIV_CHECKS_IN_VALUE_LOOP: bool = true;

            #[inline(always)]
            fn add_value(self, rhs: Self) -> Self {
                self.wrapping_add(rhs)
            }

            #[inline(always)]
            fn add_error(self, rhs: Self) -> bool {
                self > <$ty>::MAX - rhs
            }

            #[inline(always)]
            fn sub_value(self, rhs: Self) -> Self {
                self.wrapping_sub(rhs)
            }

            #[inline(always)]
            fn sub_error(self, rhs: Self) -> bool {
                self < rhs
            }

            #[inline(always)]
            fn mul_value(self, rhs: Self) -> Self {
                self.wrapping_mul(rhs)
            }

            #[inline(always)]
            fn mul_error(self, rhs: Self) -> bool {
                self.overflowing_mul(rhs).1
            }

            #[inline(always)]
            fn div_value(self, rhs: Self) -> Self {
                self / rhs
            }

            #[inline(always)]
            fn div_error(self, rhs: Self) -> bool {
                rhs == 0
            }

            #[inline(always)]
            fn div_checked(self, rhs: Self) -> Option<Self> {
                self.checked_div(rhs)
            }
        }
    };
}

macro_rules! impl_checked_signed {
    ($ty:ty,widening_mul: $wide:ty) => {
        impl CheckedArithmetic for $ty {
            const DIV_CHECKS_IN_VALUE_LOOP: bool = true;

            #[inline(always)]
            fn add_value(self, rhs: Self) -> Self {
                self.wrapping_add(rhs)
            }

            #[inline(always)]
            fn add_error(self, rhs: Self) -> bool {
                let value = self.wrapping_add(rhs);
                ((self ^ value) & (rhs ^ value)) < 0
            }

            #[inline(always)]
            fn sub_value(self, rhs: Self) -> Self {
                self.wrapping_sub(rhs)
            }

            #[inline(always)]
            fn sub_error(self, rhs: Self) -> bool {
                let value = self.wrapping_sub(rhs);
                ((self ^ rhs) & (self ^ value)) < 0
            }

            #[inline(always)]
            fn mul_value(self, rhs: Self) -> Self {
                self.wrapping_mul(rhs)
            }

            #[inline(always)]
            fn mul_error(self, rhs: Self) -> bool {
                let product = (self as $wide) * (rhs as $wide);
                product < <$ty>::MIN as $wide || product > <$ty>::MAX as $wide
            }

            #[inline(always)]
            fn div_value(self, rhs: Self) -> Self {
                self / rhs
            }

            #[inline(always)]
            fn div_error(self, rhs: Self) -> bool {
                rhs == 0 || (self == <$ty>::MIN && rhs == -1)
            }

            #[inline(always)]
            fn div_checked(self, rhs: Self) -> Option<Self> {
                self.checked_div(rhs)
            }
        }
    };
    ($ty:ty,overflowing_mul) => {
        impl CheckedArithmetic for $ty {
            const DIV_CHECKS_IN_VALUE_LOOP: bool = true;

            #[inline(always)]
            fn add_value(self, rhs: Self) -> Self {
                self.wrapping_add(rhs)
            }

            #[inline(always)]
            fn add_error(self, rhs: Self) -> bool {
                let value = self.wrapping_add(rhs);
                ((self ^ value) & (rhs ^ value)) < 0
            }

            #[inline(always)]
            fn sub_value(self, rhs: Self) -> Self {
                self.wrapping_sub(rhs)
            }

            #[inline(always)]
            fn sub_error(self, rhs: Self) -> bool {
                let value = self.wrapping_sub(rhs);
                ((self ^ rhs) & (self ^ value)) < 0
            }

            #[inline(always)]
            fn mul_value(self, rhs: Self) -> Self {
                self.wrapping_mul(rhs)
            }

            #[inline(always)]
            fn mul_error(self, rhs: Self) -> bool {
                self.overflowing_mul(rhs).1
            }

            #[inline(always)]
            fn div_value(self, rhs: Self) -> Self {
                self / rhs
            }

            #[inline(always)]
            fn div_error(self, rhs: Self) -> bool {
                rhs == 0 || (self == <$ty>::MIN && rhs == -1)
            }

            #[inline(always)]
            fn div_checked(self, rhs: Self) -> Option<Self> {
                self.checked_div(rhs)
            }
        }
    };
}

macro_rules! impl_checked_float {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl CheckedArithmetic for $ty {
                const DIV_CHECKS_IN_VALUE_LOOP: bool = false;

                #[inline(always)]
                fn add_value(self, rhs: Self) -> Self {
                    self + rhs
                }

                #[inline(always)]
                fn add_error(self, _rhs: Self) -> bool {
                    false
                }

                #[inline(always)]
                fn sub_value(self, rhs: Self) -> Self {
                    self - rhs
                }

                #[inline(always)]
                fn sub_error(self, _rhs: Self) -> bool {
                    false
                }

                #[inline(always)]
                fn mul_value(self, rhs: Self) -> Self {
                    self * rhs
                }

                #[inline(always)]
                fn mul_error(self, _rhs: Self) -> bool {
                    false
                }

                #[inline(always)]
                fn div_value(self, rhs: Self) -> Self {
                    self / rhs
                }

                #[inline(always)]
                fn div_error(self, _rhs: Self) -> bool {
                    false
                }

                #[inline(always)]
                fn div_checked(self, rhs: Self) -> Option<Self> {
                    Some(self / rhs)
                }
            }
        )+
    };
}

impl_checked_unsigned!(u8, widening_mul: u16);
impl_checked_unsigned!(u16, widening_mul: u32);
impl_checked_unsigned!(u32, widening_mul: u64);
impl_checked_unsigned!(u64, overflowing_mul);
impl_checked_signed!(i8, widening_mul: i16);
impl_checked_signed!(i16, widening_mul: i32);
impl_checked_signed!(i32, widening_mul: i64);
impl_checked_signed!(i64, overflowing_mul);
impl_checked_float!(f16, f32, f64);
