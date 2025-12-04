// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arithmetic operations on buffers and vectors.

use vortex_dtype::half::f16;

mod buffer;
mod buffer_checked;
mod datum;
mod primitive_scalar;
mod primitive_vector;
mod pscalar;
mod pvector;
mod pvector_checked;

/// Trait for arithmetic operations.
pub trait Arithmetic<Op, Rhs = Self> {
    /// The result type after performing the operation.
    type Output;

    /// Perform the operation.
    fn eval(self, rhs: Rhs) -> Self::Output;
}

/// Trait for checked arithmetic operators.
pub trait Operator<T> {
    /// Apply the operator to the two operands.
    fn apply(a: &T, b: &T) -> T;
}

/// Trait for checked arithmetic operations.
pub trait CheckedArithmetic<Op, Rhs = Self> {
    /// The result type after performing the operation.
    type Output;

    /// Perform the operation, returning None on overflow/underflow or division by zero.
    /// See the `Op` marker detailed semantics on the checked behavior.
    fn checked_eval(self, rhs: Rhs) -> Option<Self::Output>;
}

/// Trait for checked arithmetic operators.
pub trait CheckedOperator<T> {
    /// Apply the operator to the two operands, returning None on overflow/underflow.
    fn apply(a: &T, b: &T) -> Option<T>;
}

/// Marker type for arithmetic addition.
pub struct Add;
/// Marker type for arithmetic subtraction.
pub struct Sub;
/// Marker type for arithmetic multiplication.
pub struct Mul;
/// Marker type for arithmetic division.
pub struct Div;

/// Marker type for arithmetic addition that wraps on overflow.
pub struct WrappingAdd;
/// Marker type for arithmetic subtraction that wraps on overflow.
pub struct WrappingSub;
/// Marker type for arithmetic multiplication that wraps on overflow.
pub struct WrappingMul;

/// Marker type for arithmetic addition that saturates on overflow.
pub struct SaturatingAdd;
/// Marker type for arithmetic subtraction that saturates on overflow.
pub struct SaturatingSub;
/// Marker type for arithmetic multiplication that saturates on overflow.
pub struct SaturatingMul;

impl<T: num_traits::CheckedAdd> CheckedOperator<T> for Add {
    #[inline(always)]
    fn apply(a: &T, b: &T) -> Option<T> {
        num_traits::CheckedAdd::checked_add(a, b)
    }
}
impl<T: num_traits::CheckedSub> CheckedOperator<T> for Sub {
    #[inline(always)]
    fn apply(a: &T, b: &T) -> Option<T> {
        num_traits::CheckedSub::checked_sub(a, b)
    }
}
impl<T: num_traits::CheckedMul> CheckedOperator<T> for Mul {
    #[inline(always)]
    fn apply(a: &T, b: &T) -> Option<T> {
        num_traits::CheckedMul::checked_mul(a, b)
    }
}
impl<T: num_traits::CheckedDiv> CheckedOperator<T> for Div {
    #[inline(always)]
    fn apply(a: &T, b: &T) -> Option<T> {
        num_traits::CheckedDiv::checked_div(a, b)
    }
}

impl<T: num_traits::WrappingAdd> Operator<T> for WrappingAdd {
    #[inline(always)]
    fn apply(a: &T, b: &T) -> T {
        num_traits::WrappingAdd::wrapping_add(a, b)
    }
}
impl<T: num_traits::WrappingSub> Operator<T> for WrappingSub {
    #[inline(always)]
    fn apply(a: &T, b: &T) -> T {
        num_traits::WrappingSub::wrapping_sub(a, b)
    }
}
impl<T: num_traits::WrappingMul> Operator<T> for WrappingMul {
    #[inline(always)]
    fn apply(a: &T, b: &T) -> T {
        num_traits::WrappingMul::wrapping_mul(a, b)
    }
}

impl<T: num_traits::SaturatingAdd> Operator<T> for SaturatingAdd {
    #[inline(always)]
    fn apply(a: &T, b: &T) -> T {
        num_traits::SaturatingAdd::saturating_add(a, b)
    }
}
impl<T: num_traits::SaturatingSub> Operator<T> for SaturatingSub {
    #[inline(always)]
    fn apply(a: &T, b: &T) -> T {
        num_traits::SaturatingSub::saturating_sub(a, b)
    }
}
impl<T: num_traits::SaturatingMul> Operator<T> for SaturatingMul {
    #[inline(always)]
    fn apply(a: &T, b: &T) -> T {
        num_traits::SaturatingMul::saturating_mul(a, b)
    }
}

/// Macro to implement arithmetic operators for floating-point types.
///
/// These are not deferred to the `std::ops::Add` since those implementations will panic on
/// overflow in some cases (e.g., debug builds).
macro_rules! impl_float {
    ($T:ty) => {
        impl Operator<$T> for Add {
            #[inline(always)]
            fn apply(a: &$T, b: &$T) -> $T {
                a + b
            }
        }
        impl Operator<$T> for Sub {
            #[inline(always)]
            fn apply(a: &$T, b: &$T) -> $T {
                a - b
            }
        }
        impl Operator<$T> for Mul {
            #[inline(always)]
            fn apply(a: &$T, b: &$T) -> $T {
                a * b
            }
        }
        impl Operator<$T> for Div {
            #[inline(always)]
            fn apply(a: &$T, b: &$T) -> $T {
                a / b
            }
        }
    };
}

impl_float!(f16);
impl_float!(f32);
impl_float!(f64);
