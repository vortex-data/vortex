// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Checked arithmetic for one primitive row.

use std::ops::BitOrAssign;

use crate::dtype::NativePType;
use crate::dtype::half::f16;

/// Checked addition, failing on integer overflow.
pub(super) struct CheckedAdd;

/// Checked subtraction, failing on integer overflow.
pub(super) struct CheckedSub;

/// Checked multiplication, failing on integer overflow.
pub(super) struct CheckedMul;

/// Checked division, failing on integer division by zero and on `MIN / -1`.
pub(super) struct CheckedDiv;

/// OR-reducible evidence that a row failed, with [`Default`] meaning success.
pub(super) trait Failure: Copy + Default + PartialEq + BitOrAssign {}

impl Failure for bool {}
impl Failure for u8 {}
impl Failure for u16 {}
impl Failure for u32 {}
impl Failure for u64 {}

/// One arithmetic operator at one width, split into its value and failure evidence.
pub(super) trait CheckedPrimitiveOp<T: NativePType>: 'static + Sized {
    /// The error reported for a batch in which some valid row failed.
    const ERROR: &'static str;

    /// How this operation reports a failing row. See [`Failure`].
    type Fail: Failure;

    /// The result of this operation, paired with evidence of whether the row failed.
    fn apply(lhs: T, rhs: T) -> (T, Self::Fail);
}

impl<T: CheckedArithmetic> CheckedPrimitiveOp<T> for CheckedAdd {
    const ERROR: &'static str = "integer overflow in checked add";

    type Fail = bool;

    #[inline]
    fn apply(lhs: T, rhs: T) -> (T, bool) {
        (lhs.add_value(rhs), lhs.add_error(rhs))
    }
}

impl<T: CheckedArithmetic> CheckedPrimitiveOp<T> for CheckedSub {
    const ERROR: &'static str = "integer overflow in checked sub";

    type Fail = bool;

    #[inline]
    fn apply(lhs: T, rhs: T) -> (T, bool) {
        (lhs.sub_value(rhs), lhs.sub_error(rhs))
    }
}

impl<T: CheckedArithmetic> CheckedPrimitiveOp<T> for CheckedMul {
    const ERROR: &'static str = "integer overflow in checked mul";

    type Fail = T::MulFailure;

    #[inline]
    fn apply(lhs: T, rhs: T) -> (T, T::MulFailure) {
        (lhs.mul_value(rhs), lhs.mul_failure(rhs))
    }
}

impl<T: CheckedArithmetic> CheckedPrimitiveOp<T> for CheckedDiv {
    const ERROR: &'static str = "integer division by zero or overflow in checked div";

    type Fail = bool;

    #[inline]
    fn apply(lhs: T, rhs: T) -> (T, bool) {
        let failed = lhs.div_error(rhs);
        let value = if failed {
            T::default()
        } else {
            lhs.div_value(rhs)
        };
        (value, failed)
    }
}

/// Per-width arithmetic used to compute values and failure evidence.
///
/// The add, subtract, and multiply value methods **must** be total over every stored lane value.
/// [`Self::div_value`] may assume that [`Self::div_error`] returned `false` for the same operands.
pub(super) trait CheckedArithmetic: NativePType {
    /// How multiplication reports a failing row.
    ///
    /// This may be a word rather than `bool` when narrowing evidence would block vectorization.
    type MulFailure: Failure;

    fn add_value(self, rhs: Self) -> Self;
    fn add_error(self, rhs: Self) -> bool;
    fn sub_value(self, rhs: Self) -> Self;
    fn sub_error(self, rhs: Self) -> bool;
    fn mul_value(self, rhs: Self) -> Self;
    fn mul_failure(self, rhs: Self) -> Self::MulFailure;

    /// Divide operands that [`Self::div_error`] accepted.
    fn div_value(self, rhs: Self) -> Self;

    /// Return whether [`Self::div_value`] would trap for these operands.
    fn div_error(self, rhs: Self) -> bool;
}

/// Generate the shared integer operations from their failure predicates.
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
            type MulFailure = $mul_failure_ty;

            #[inline]
            fn add_value(self, rhs: Self) -> Self {
                self.wrapping_add(rhs)
            }

            #[inline]
            fn add_error(self, rhs: Self) -> bool {
                let ($add_lhs, $add_rhs) = (self, rhs);
                $add_error
            }

            #[inline]
            fn sub_value(self, rhs: Self) -> Self {
                self.wrapping_sub(rhs)
            }

            #[inline]
            fn sub_error(self, rhs: Self) -> bool {
                let ($sub_lhs, $sub_rhs) = (self, rhs);
                $sub_error
            }

            #[inline]
            fn mul_value(self, rhs: Self) -> Self {
                self.wrapping_mul(rhs)
            }

            #[inline]
            $(#[$mul_failure_attr])*
            fn mul_failure(self, rhs: Self) -> $mul_failure_ty {
                let ($mf_lhs, $mf_rhs) = (self, rhs);
                $mul_failure
            }

            #[inline]
            fn div_value(self, rhs: Self) -> Self {
                self / rhs
            }

            #[inline]
            fn div_error(self, rhs: Self) -> bool {
                let ($div_lhs, $div_rhs) = (self, rhs);
                $div_error
            }
        }
    };
}

/// Unsigned multiplication reports its discarded high half as failure evidence.
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

/// Signed widths use a range check or discarded high-half evidence.
macro_rules! impl_checked_signed {
    ($ty:ty, widening_mul: $wide:ty) => {
        impl_checked_signed!($ty, mul_failure: bool = |lhs, rhs| {
            let product = (lhs as $wide) * (rhs as $wide);
            product < <$ty>::MIN as $wide || product > <$ty>::MAX as $wide
        });
    };
    ($ty:ty, high_half_mul: $wide:ty => $failure:ty) => {
        impl_checked_signed!($ty, mul_failure: #[expect(
            clippy::cast_possible_truncation,
            reason = "the truncated half is the result, and the discarded half is the evidence"
        )] $failure = |lhs, rhs| {
            let wide = (lhs as $wide) * (rhs as $wide);
            let kept = wide as $ty;
            let discarded = (wide >> <$ty>::BITS) as $ty;

            // A product fits exactly when its discarded half is the sign extension of the kept
            // half. XOR reduces that comparison to zero evidence for success and nonzero evidence
            // for overflow without converting the wide product to a branch.

            (discarded ^ (kept >> (<$ty>::BITS - 1))) as $failure
        });
    };
    (
        $ty:ty,
        mul_failure: $(#[$mul_failure_attr:meta])* $mul_failure_ty:ty
            = |$lhs:ident, $rhs:ident| $mul_failure:expr
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
            mul_failure: $(#[$mul_failure_attr])* $mul_failure_ty = |$lhs, $rhs| $mul_failure,
        );
    };
}

macro_rules! impl_checked_float {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl CheckedArithmetic for $ty {
                type MulFailure = bool;

                #[inline]
                fn add_value(self, rhs: Self) -> Self {
                    self + rhs
                }

                #[inline]
                fn add_error(self, _rhs: Self) -> bool {
                    false
                }

                #[inline]
                fn sub_value(self, rhs: Self) -> Self {
                    self - rhs
                }

                #[inline]
                fn sub_error(self, _rhs: Self) -> bool {
                    false
                }

                #[inline]
                fn mul_value(self, rhs: Self) -> Self {
                    self * rhs
                }

                #[inline]
                fn mul_failure(self, _rhs: Self) -> bool {
                    false
                }

                #[inline]
                fn div_value(self, rhs: Self) -> Self {
                    self / rhs
                }

                #[inline]
                fn div_error(self, _rhs: Self) -> bool {
                    false
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

    /// Values around zero, signed extrema, and 32- and 64-bit boundaries where the discarded
    /// multiplication half or its sign extension changes.
    const PROBES: &[i64] = &[
        0,            // Additive identity.
        1,            // Smallest positive value.
        -1,           // All sign bits set.
        2,            // Small positive power of two.
        -2,           // Small negative power of two.
        3,            // Small non-power of two.
        i64::MIN,     // Minimum signed value.
        i64::MIN + 1, // Minimum signed value's neighbor.
        i64::MAX,     // Maximum signed value.
        i64::MAX - 1, // Maximum signed value's neighbor.
        1 << 31,      // First positive value outside i32.
        1 << 32,      // First value with bit 32 set.
        1 << 62,      // Largest positive power of two in i64.
        -(1 << 62),   // Negative counterpart of the largest power of two.
        0x7FFF_FFFF,  // Maximum i32 represented as i64.
        -0x8000_0000, // Minimum i32 represented as i64.
    ];

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
