// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_compute::lane_kernels::IndexedSource;
use vortex_compute::lane_kernels::LaneZip;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use super::checked::Failure;
use super::checked::checked_apply_lanes;
use super::checked::checked_lanes;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
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
use crate::scalar_fn::fns::binary::primitive_operand::PrimitiveOperand;
use crate::validity::Validity;

struct CheckedAdd;

struct CheckedSub;

struct CheckedMul;

struct CheckedDiv;

trait CheckedPrimitiveOp<T: NativePType>: Sized {
    /// Message for the error raised when this operation fails on a valid lane.
    const ERROR: &'static str;

    /// Whether to check inside the value loop and exit early, rather than reducing evidence over
    /// the whole batch and re-running only once some lane has flagged.
    const CHECKED_VALUE_LOOP: bool = false;

    /// How this operation reports a failing lane. See [`Failure`].
    type Failure: Failure;

    /// Compute a lane's value and its failure evidence together.
    ///
    /// Overflowing-style rather than `Option<T>`, so the vectorizable kernels can write every
    /// lane's value unconditionally and reduce the evidence separately. [`Self::checked`] is the
    /// shape for the scalar and early-exit paths.
    fn apply(lhs: T, rhs: T) -> (T, Self::Failure);

    /// [`Self::apply`] folded into the `Option` that the early-exit kernels take.
    #[inline(always)]
    fn checked(lhs: T, rhs: T) -> Option<T> {
        let (value, failed) = Self::apply(lhs, rhs);

        (failed == Self::Failure::default()).then_some(value)
    }
}

impl<T: CheckedArithmetic> CheckedPrimitiveOp<T> for CheckedAdd {
    const ERROR: &'static str = "integer overflow in checked add";

    type Failure = bool;

    #[inline(always)]
    fn apply(lhs: T, rhs: T) -> (T, bool) {
        (lhs.add_value(rhs), lhs.add_error(rhs))
    }
}

impl<T: CheckedArithmetic> CheckedPrimitiveOp<T> for CheckedSub {
    const ERROR: &'static str = "integer overflow in checked sub";

    type Failure = bool;

    #[inline(always)]
    fn apply(lhs: T, rhs: T) -> (T, bool) {
        (lhs.sub_value(rhs), lhs.sub_error(rhs))
    }
}

impl<T: CheckedArithmetic> CheckedPrimitiveOp<T> for CheckedMul {
    const ERROR: &'static str = "integer overflow in checked mul";

    type Failure = T::MulFailure;

    #[inline(always)]
    fn apply(lhs: T, rhs: T) -> (T, T::MulFailure) {
        (lhs.mul_value(rhs), lhs.mul_failure(rhs))
    }
}

impl<T: CheckedArithmetic> CheckedPrimitiveOp<T> for CheckedDiv {
    const ERROR: &'static str = "integer division by zero or overflow in checked div";
    // Integer division still lowers to scalar divides, so the split
    // value/error-scan loop used to auto-vectorize add/sub/mul only adds a
    // second full scan. Use the one-pass early-exit checked kernel for integer
    // division, matching Arrow/Velox. Float division has no checked errors and
    // stays on the split/vectorizable default path.
    const CHECKED_VALUE_LOOP: bool = T::DIV_CHECKS_IN_VALUE_LOOP;

    type Failure = bool;

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

    let values = match (&lhs, &rhs) {
        (
            PrimitiveOperand::Array { values: lhs, .. },
            PrimitiveOperand::Array { values: rhs, .. },
        ) => checked_op_lanes::<_, T, Op>(
            LaneZip::new(lhs.as_slice(), rhs.as_slice()),
            &valid_rows,
            |(lhs, rhs)| (lhs, rhs),
        ),
        (
            PrimitiveOperand::Array { values: lhs, .. },
            PrimitiveOperand::Constant { value: rhs, .. },
        ) => {
            // Capture the constant by value so it stays hoisted out of the lane loop.
            let rhs = *rhs;
            checked_op_lanes::<_, T, Op>(lhs.as_slice(), &valid_rows, move |lhs| (lhs, rhs))
        }
        (
            PrimitiveOperand::Constant { value: lhs, .. },
            PrimitiveOperand::Array { values: rhs, .. },
        ) => {
            let lhs = *lhs;
            checked_op_lanes::<_, T, Op>(rhs.as_slice(), &valid_rows, move |rhs| (lhs, rhs))
        }
        (
            PrimitiveOperand::Constant { value: lhs, .. },
            PrimitiveOperand::Constant { value: rhs, .. },
        ) => {
            let value = Op::checked(*lhs, *rhs)
                .ok_or_else(|| vortex_err!(InvalidArgument: "{}", Op::ERROR))?;
            return Ok(constant_result_array(value, len, &result_dtype));
        }
        (PrimitiveOperand::Null(_), _) | (_, PrimitiveOperand::Null(_)) => Ok(Buffer::zeroed(len)),
    }
    .map_err(|_lane| vortex_err!(InvalidArgument: "{}", Op::ERROR))?;

    primitive_result_array::<T>(values, validity, &result_dtype)
}

/// Run `Op` over the lanes of `source` in the loop shape it declares: the one-pass early-exit
/// kernel when `Op::CHECKED_VALUE_LOOP` is set, and the split value/failure kernel otherwise.
///
/// `#[inline]`: see [`checked_lanes`].
#[inline]
fn checked_op_lanes<S, T, Op>(
    source: S,
    valid_rows: &Mask,
    to_operands: impl Fn(S::Item) -> (T, T),
) -> Result<Buffer<T>, usize>
where
    S: IndexedSource + Copy,
    T: NativePType,
    Op: CheckedPrimitiveOp<T>,
{
    if Op::CHECKED_VALUE_LOOP {
        checked_lanes(source, valid_rows, |item| {
            let (lhs, rhs) = to_operands(item);
            Op::checked(lhs, rhs)
        })
    } else {
        checked_apply_lanes(source, valid_rows, |item| {
            let (lhs, rhs) = to_operands(item);
            Op::apply(lhs, rhs)
        })
    }
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

trait CheckedArithmetic: NativePType {
    const DIV_CHECKS_IN_VALUE_LOOP: bool;

    /// How multiplication reports a failing lane, which is width-dependent in a way the other
    /// operations are not. `impl_checked_unsigned!` and `impl_checked_signed!` say why each family
    /// reports what it reports.
    type MulFailure: Failure;

    fn add_value(self, rhs: Self) -> Self;
    fn add_error(self, rhs: Self) -> bool;
    fn sub_value(self, rhs: Self) -> Self;
    fn sub_error(self, rhs: Self) -> bool;
    fn mul_value(self, rhs: Self) -> Self;
    fn mul_failure(self, rhs: Self) -> Self::MulFailure;
    fn div_value(self, rhs: Self) -> Self;
    fn div_error(self, rhs: Self) -> bool;
    fn div_checked(self, rhs: Self) -> Option<Self>;
}

/// The integer arithmetic every width shares, given the two things that actually differ between
/// them: how multiplication reports a failing lane, and how add/sub/div detect one.
///
/// Only `mul_failure` genuinely varies per width, so it is the macro's parameter and everything
/// else is written once. The `$mul_failure_ty` and body are supplied by the caller because the
/// choice between a widened high half and a `bool` is exactly the vectorization decision described
/// on [`Failure`].
macro_rules! impl_checked_integer {
    (
        $ty:ty,
        add_error: |$add_lhs:ident, $add_rhs:ident| $add_error:expr,
        sub_error: |$sub_lhs:ident, $sub_rhs:ident| $sub_error:expr,
        div_error: |$div_lhs:ident, $div_rhs:ident| $div_error:expr,
        mul_failure: $(#[$mul_failure_attr:meta])* $mul_failure_ty:ty
            = |$mf_lhs:ident, $mf_rhs:ident| $mul_failure:expr,
    ) => {
        impl CheckedArithmetic for $ty {
            const DIV_CHECKS_IN_VALUE_LOOP: bool = true;

            type MulFailure = $mul_failure_ty;

            #[inline(always)]
            fn add_value(self, rhs: Self) -> Self {
                self.wrapping_add(rhs)
            }

            #[inline(always)]
            fn add_error(self, rhs: Self) -> bool {
                let ($add_lhs, $add_rhs) = (self, rhs);
                $add_error
            }

            #[inline(always)]
            fn sub_value(self, rhs: Self) -> Self {
                self.wrapping_sub(rhs)
            }

            #[inline(always)]
            fn sub_error(self, rhs: Self) -> bool {
                let ($sub_lhs, $sub_rhs) = (self, rhs);
                $sub_error
            }

            #[inline(always)]
            fn mul_value(self, rhs: Self) -> Self {
                self.wrapping_mul(rhs)
            }

            #[inline(always)]
            $(#[$mul_failure_attr])*
            fn mul_failure(self, rhs: Self) -> $mul_failure_ty {
                let ($mf_lhs, $mf_rhs) = (self, rhs);
                $mul_failure
            }

            #[inline(always)]
            fn div_value(self, rhs: Self) -> Self {
                self / rhs
            }

            #[inline(always)]
            fn div_error(self, rhs: Self) -> bool {
                let ($div_lhs, $div_rhs) = (self, rhs);
                $div_error
            }

            #[inline(always)]
            fn div_checked(self, rhs: Self) -> Option<Self> {
                self.checked_div(rhs)
            }
        }
    };
}

/// The unsigned widths. `add`, `sub` and `div` are written once here, and `widening_mul` covers
/// every width including the 64-bit one, which widens into `u128`: the discarded high half of the
/// widened product is the failure evidence, and costs none of the comparison LLVM folds into
/// `umul.with.overflow`.
macro_rules! impl_checked_unsigned {
    ($ty:ty, widening_mul: $wide:ty) => {
        impl_checked_integer!(
            $ty,
            add_error: |lhs, rhs| lhs > <$ty>::MAX - rhs,
            sub_error: |lhs, rhs| lhs < rhs,
            div_error: |_lhs, rhs| rhs == 0,
            mul_failure: $ty = |lhs, rhs| (((lhs as $wide) * (rhs as $wide)) >> <$ty>::BITS) as $ty,
        );
    };
}

/// The signed widths, on the same principle. `widening_mul` is the shorthand for the narrow widths,
/// whose two-sided range check over a wider product is not the shape LLVM folds into an overflow
/// intrinsic, so they vectorize while reporting a plain `bool`. The 64-bit width cannot: deriving a
/// `bool` there costs the comparison that scalarizes the loop, so `high_half_mul` hands back the
/// discarded high half as a word. Both arms derive every shift and bound from `$ty`, so
/// instantiating one at a new width cannot silently keep another width's constants.
macro_rules! impl_checked_signed {
    ($ty:ty, widening_mul: $wide:ty) => {
        impl_checked_signed!($ty, mul_failure: bool = |lhs, rhs| {
            let product = (lhs as $wide) * (rhs as $wide);
            product < <$ty>::MIN as $wide || product > <$ty>::MAX as $wide
        });
    };
    // Zero exactly when the product fits: a signed multiply overflows iff the high half of the true
    // product differs from the sign extension of the half that was kept, so the two XOR to zero on
    // the lanes that fit. `tests::test_multiply_overflow_boundaries` pins the boundaries.
    ($ty:ty, high_half_mul: $wide:ty => $failure:ty) => {
        impl_checked_signed!($ty, mul_failure: #[expect(
            clippy::cast_possible_truncation,
            reason = "the truncated half is the result, and the discarded half is the evidence"
        )] $failure = |lhs, rhs| {
            let wide = (lhs as $wide) * (rhs as $wide);
            let kept = wide as $ty;
            let discarded = (wide >> <$ty>::BITS) as $ty;

            (discarded ^ (kept >> (<$ty>::BITS - 1))) as $failure
        });
    };
    (
        $ty:ty,
        mul_failure: $(#[$mul_failure_attr:meta])* $mul_failure_ty:ty
            = |$l:ident, $r:ident| $mul_failure:expr
    ) => {
        impl_checked_integer!(
            $ty,
            add_error: |lhs, rhs| {
                let value = lhs.wrapping_add(rhs);
                ((lhs ^ value) & (rhs ^ value)) < 0
            },
            sub_error: |lhs, rhs| {
                let value = lhs.wrapping_sub(rhs);
                ((lhs ^ rhs) & (lhs ^ value)) < 0
            },
            div_error: |lhs, rhs| rhs == 0 || (lhs == <$ty>::MIN && rhs == -1),
            mul_failure: $(#[$mul_failure_attr])* $mul_failure_ty = |$l, $r| $mul_failure,
        );
    };
}

macro_rules! impl_checked_float {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl CheckedArithmetic for $ty {
                const DIV_CHECKS_IN_VALUE_LOOP: bool = false;

                type MulFailure = bool;

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
                fn mul_failure(self, _rhs: Self) -> bool {
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
impl_checked_unsigned!(u64, widening_mul: u128);
impl_checked_signed!(i8, widening_mul: i16);
impl_checked_signed!(i16, widening_mul: i32);
impl_checked_signed!(i32, widening_mul: i64);
impl_checked_signed!(i64, high_half_mul: i128 => u64);
impl_checked_float!(f16, f32, f64);

#[cfg(test)]
mod tests {
    use super::CheckedArithmetic;

    /// Values whose pairwise products are worth probing: the saturating boundaries, the sign-change
    /// pivots, and a spread of magnitudes that straddles the 64-bit split.
    const PROBES: &[i64] = &[
        0,            //
        1,            //
        -1,           //
        2,            //
        -2,           //
        3,            //
        i64::MIN,     //
        i64::MIN + 1, //
        i64::MAX,     //
        i64::MAX - 1, //
        1 << 31,      //
        1 << 32,      //
        1 << 62,      //
        -(1 << 62),   //
        0x7FFF_FFFF,  //
        -0x8000_0000, //
    ];

    /// Every `mul_failure` impl is either a bit trick or a two-sided range check, so hold each
    /// against the obvious reference: `reference` is the width's own `checked_mul`, whose `None`
    /// _is_ the definition of overflow.
    #[track_caller]
    fn assert_agrees_with_checked_mul<T: CheckedArithmetic>(lhs: T, rhs: T, reference: Option<T>) {
        let failed = lhs.mul_failure(rhs) != <T::MulFailure as Default>::default();

        assert_eq!(failed, reference.is_none(), "{lhs:?} * {rhs:?}");
    }

    #[test]
    fn mul_failure_agrees_with_checked_mul_at_64_bits() {
        for &lhs in PROBES {
            for &rhs in PROBES {
                assert_agrees_with_checked_mul(lhs, rhs, lhs.checked_mul(rhs));

                let (lhs, rhs) = (lhs as u64, rhs as u64);

                assert_agrees_with_checked_mul(lhs, rhs, lhs.checked_mul(rhs));
            }
        }
    }

    /// The 8-bit widths are cheap enough to check exhaustively, which pins the shift of the
    /// unsigned formula and the range check of the signed one against every product that exists.
    #[test]
    fn mul_failure_agrees_with_checked_mul_exhaustively_at_8_bits() {
        for lhs in u8::MIN..=u8::MAX {
            for rhs in u8::MIN..=u8::MAX {
                assert_agrees_with_checked_mul(lhs, rhs, lhs.checked_mul(rhs));
            }
        }

        for lhs in i8::MIN..=i8::MAX {
            for rhs in i8::MIN..=i8::MAX {
                assert_agrees_with_checked_mul(lhs, rhs, lhs.checked_mul(rhs));
            }
        }
    }
}
