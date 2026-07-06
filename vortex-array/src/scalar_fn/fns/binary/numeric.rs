// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
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

/// Execute a numeric operation between two arrays.
pub(crate) fn execute_numeric(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: NumericOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let ptype = PType::try_from(lhs.dtype())?;
    if !lhs.dtype().eq_ignore_nullability(rhs.dtype()) {
        vortex_bail!(
            "numeric operator requires matching primitive types, got {} and {}",
            lhs.dtype(),
            rhs.dtype()
        );
    }

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
    if len != rhs.len() {
        vortex_bail!(
            "numeric operator requires equal lengths, got {} and {}",
            len,
            rhs.len()
        );
    }

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
    check_numeric_errors(checked.failed, Op::ERROR)?;

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

enum PrimitiveOperand<T: NativePType> {
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

        let array = array.clone().execute::<PrimitiveArray>(ctx)?;
        let validity = array.validity()?;
        let values = array.into_buffer::<T>();
        Ok(Self::Array { values, validity })
    }

    fn len(&self) -> usize {
        match self {
            Self::Array { values, .. } => values.len(),
            Self::Constant { len, .. } | Self::Null(len) => *len,
        }
    }

    fn validity(&self) -> Validity {
        match self {
            Self::Array { validity, .. } => validity.clone(),
            Self::Constant { validity, .. } => validity.clone(),
            Self::Null(_) => Validity::AllInvalid,
        }
    }
}

struct CheckedValues<T: NativePType> {
    values: Buffer<T>,
    failed: bool,
}

impl<T: NativePType> CheckedValues<T> {
    fn zeroed(len: usize) -> Self {
        Self {
            values: Buffer::<T>::zeroed(len),
            failed: false,
        }
    }

    fn failed(len: usize) -> Self {
        Self {
            values: Buffer::<T>::zeroed(len),
            failed: true,
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

// Checked one-pass ops delay early exit until the end of a small block. This
// keeps the loop generic while avoiding a branch-driven exit decision on every
// lane; it is deliberately independent of mask density or input length.
const CHECKED_BLOCK_LANES: usize = 16;

fn checked_all_lanes<T, F>(len: usize, mut checked_at: F) -> CheckedValues<T>
where
    T: NativePType,
    F: FnMut(usize) -> Option<T>,
{
    let mut values = BufferMut::<T>::with_capacity(len);
    let mut base = 0;

    while base + CHECKED_BLOCK_LANES <= len {
        let mut block_failed = false;
        for idx in base..base + CHECKED_BLOCK_LANES {
            match checked_at(idx) {
                Some(value) => {
                    // SAFETY: the buffer is allocated with capacity `len`, and
                    // this loop pushes at most one value for each `idx`.
                    unsafe { values.push_unchecked(value) };
                }
                None => {
                    block_failed = true;
                    // SAFETY: the buffer is allocated with capacity `len`, and
                    // this loop pushes at most one value for each `idx`.
                    unsafe { values.push_unchecked(T::default()) };
                }
            }
        }

        if block_failed {
            return CheckedValues::failed(len);
        }
        base += CHECKED_BLOCK_LANES;
    }

    for idx in base..len {
        let Some(value) = checked_at(idx) else {
            return CheckedValues::failed(len);
        };
        // SAFETY: the buffer is allocated with capacity `len`, and this loop
        // pushes at most one value for each `idx`.
        unsafe { values.push_unchecked(value) };
    }

    CheckedValues {
        values: values.freeze(),
        failed: false,
    }
}

fn checked_valid_lanes<T, F>(
    len: usize,
    valid_bits: &BitBuffer,
    mut checked_at: F,
) -> CheckedValues<T>
where
    T: NativePType,
    F: FnMut(usize) -> Option<T>,
{
    let mut values = BufferMut::<T>::zeroed(len);
    let mut failed = false;
    {
        let values = values.as_mut_slice();
        for_each_valid_idx(len, valid_bits, |idx| {
            let Some(value) = checked_at(idx) else {
                failed = true;
                return false;
            };
            values[idx] = value;
            true
        });
    }

    CheckedValues {
        values: values.freeze(),
        failed,
    }
}

fn any_valid_error<F>(len: usize, valid_bits: &BitBuffer, is_error: F) -> bool
where
    F: Fn(usize) -> bool,
{
    !for_each_valid_idx(len, valid_bits, |idx| !is_error(idx))
}

fn for_each_valid_idx<F>(len: usize, valid_bits: &BitBuffer, mut f: F) -> bool
where
    F: FnMut(usize) -> bool,
{
    debug_assert_eq!(len, valid_bits.len());

    for (word_idx, valid_word) in valid_bits.chunks().iter_padded().enumerate() {
        if valid_word == 0 {
            continue;
        }

        let offset = word_idx * 64;
        let lanes = len.saturating_sub(offset).min(64);
        if lanes == 64 && valid_word == u64::MAX {
            for bit_idx in 0..64 {
                if !f(offset + bit_idx) {
                    return false;
                }
            }
            continue;
        }

        let mut valid_word = if lanes == 64 {
            valid_word
        } else {
            valid_word & ((1u64 << lanes) - 1)
        };
        while valid_word != 0 {
            let bit_idx = valid_word.trailing_zeros() as usize;
            if !f(offset + bit_idx) {
                return false;
            }
            valid_word &= valid_word - 1;
        }
    }

    true
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

fn check_numeric_errors(failed: bool, error: &'static str) -> VortexResult<()> {
    if failed {
        return Err(vortex_err!(InvalidArgument: "{}", error));
    }

    Ok(())
}
#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::RecursiveCanonical;
    use crate::VortexSessionExecute;
    use crate::array_session;
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
                a.execute::<RecursiveCanonical>(&mut array_session().create_execution_ctx())
            })
            .map(|a| a.0.into_array())
    }

    #[test]
    fn test_scalar_subtract_unsigned() {
        let mut ctx = array_session().create_execution_ctx();
        let values = buffer![1u16, 2, 3].into_array();
        let result = sub_scalar(&values, 1u16).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([0u16, 1, 2]), &mut ctx);
    }

    #[test]
    fn test_scalar_subtract_signed() {
        let mut ctx = array_session().create_execution_ctx();
        let values = buffer![1i64, 2, 3].into_array();
        let result = sub_scalar(&values, -1i64).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([2i64, 3, 4]), &mut ctx);
    }

    #[test]
    fn test_scalar_subtract_nullable() {
        let mut ctx = array_session().create_execution_ctx();
        let values = PrimitiveArray::from_option_iter([Some(1u16), Some(2), None, Some(3)]);
        let result = sub_scalar(&values.into_array(), Some(1u16)).unwrap();
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(0u16), Some(1), None, Some(2)]),
            &mut ctx
        );
    }

    #[test]
    fn test_scalar_subtract_float() {
        let mut ctx = array_session().create_execution_ctx();
        let values = buffer![1.0f64, 2.0, 3.0].into_array();
        let result = sub_scalar(&values, -1f64).unwrap();
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_iter([2.0f64, 3.0, 4.0]),
            &mut ctx
        );
    }

    #[test]
    fn test_scalar_subtract_float_underflow_is_ok() {
        let values = buffer![f32::MIN, 2.0, 3.0].into_array();
        let _results = sub_scalar(&values, 1.0f32).unwrap();
        let _results = sub_scalar(&values, f32::MAX).unwrap();
    }

    #[test]
    fn test_float_divide_by_zero_is_ok() {
        let mut ctx = array_session().create_execution_ctx();
        let values = buffer![1.0f64, -1.0].into_array();
        let result = values
            .binary(
                ConstantArray::new(0.0f64, values.len()).into_array(),
                Operator::Div,
            )
            .and_then(|a| a.execute::<PrimitiveArray>(&mut array_session().create_execution_ctx()))
            .unwrap();

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_iter([f64::INFINITY, f64::NEG_INFINITY]),
            &mut ctx
        );
    }

    #[test]
    fn test_integer_overflow_errors() {
        let values = buffer![u8::MAX].into_array();
        let result = values
            .binary(
                ConstantArray::new(1u8, values.len()).into_array(),
                Operator::Add,
            )
            .and_then(|a| a.execute::<PrimitiveArray>(&mut array_session().create_execution_ctx()));

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
            .and_then(|a| a.execute::<PrimitiveArray>(&mut array_session().create_execution_ctx()));

        assert!(result.is_err());
    }

    #[test]
    fn test_integer_divide_overflow_errors() {
        let values = buffer![i64::MIN].into_array();
        let result = values
            .binary(
                ConstantArray::new(-1i64, values.len()).into_array(),
                Operator::Div,
            )
            .and_then(|a| a.execute::<PrimitiveArray>(&mut array_session().create_execution_ctx()));

        assert!(result.is_err());
    }

    #[test]
    fn test_integer_divide_errors_ignore_null_lanes() {
        let mut ctx = array_session().create_execution_ctx();
        let lhs = PrimitiveArray::new(buffer![10i32, 10], Validity::from_iter([false, true]))
            .into_array();
        let rhs = buffer![0i32, 2].into_array();
        let result = lhs
            .binary(rhs, Operator::Div)
            .and_then(|a| {
                a.execute::<RecursiveCanonical>(&mut array_session().create_execution_ctx())
            })
            .map(|a| a.0.into_array())
            .unwrap();

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([None, Some(5i32)]),
            &mut ctx
        );
    }

    #[test]
    fn test_integer_errors_ignore_null_lanes() {
        let mut ctx = array_session().create_execution_ctx();
        let values = PrimitiveArray::new(buffer![u8::MAX, 1], Validity::from_iter([false, true]))
            .into_array();
        let result = values
            .binary(
                ConstantArray::new(1u8, values.len()).into_array(),
                Operator::Add,
            )
            .and_then(|a| {
                a.execute::<RecursiveCanonical>(&mut array_session().create_execution_ctx())
            })
            .map(|a| a.0.into_array())
            .unwrap();

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([None, Some(2u8)]),
            &mut ctx
        );
    }

    #[test]
    fn test_integer_array_array_errors_on_valid_lanes() {
        let lhs = PrimitiveArray::new(
            buffer![u8::MAX, 1, u8::MAX],
            Validity::from_iter([false, true, true]),
        )
        .into_array();
        let rhs = buffer![1u8, 1, 1].into_array();
        let result = lhs
            .binary(rhs, Operator::Add)
            .and_then(|a| a.execute::<PrimitiveArray>(&mut array_session().create_execution_ctx()));

        assert!(result.is_err());
    }

    #[test]
    fn test_present_nullable_constant_preserves_nullable_output() {
        let mut ctx = array_session().create_execution_ctx();
        let values = buffer![1u8, 2].into_array();
        let result = values
            .binary(
                ConstantArray::new(Some(1u8), values.len()).into_array(),
                Operator::Add,
            )
            .and_then(|a| a.execute::<PrimitiveArray>(&mut array_session().create_execution_ctx()))
            .unwrap();

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(2u8), Some(3)]),
            &mut ctx
        );
    }
}
