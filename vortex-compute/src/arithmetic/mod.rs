// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arithmetic operations on buffers and vectors.

use vortex_dtype::half::f16;

mod buffer;
mod pvector;

/// Trait for checked arithmetic operations.
///
/// It may be easier to use the specific traits like [`CheckedAdd`], [`CheckedSub`], etc.
pub trait Checked<Op, Rhs = Self> {
    /// The result type after performing the operation.
    type Output;

    /// Perform the operation.
    fn checked_op(self, rhs: Rhs) -> Option<Self::Output>;
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

macro_rules! impl_integers {
    ($T:ty) => {
        impl CheckedOperator<$T> for Add {
            fn apply(lhs: &$T, rhs: &$T) -> Option<$T> {
                num_traits::CheckedAdd::checked_add(lhs, rhs)
            }
        }
        impl CheckedOperator<$T> for Sub {
            fn apply(lhs: &$T, rhs: &$T) -> Option<$T> {
                num_traits::CheckedSub::checked_sub(lhs, rhs)
            }
        }
        impl CheckedOperator<$T> for Mul {
            fn apply(lhs: &$T, rhs: &$T) -> Option<$T> {
                num_traits::CheckedMul::checked_mul(lhs, rhs)
            }
        }
        impl CheckedOperator<$T> for Div {
            fn apply(lhs: &$T, rhs: &$T) -> Option<$T> {
                num_traits::CheckedDiv::checked_div(lhs, rhs)
            }
        }
    };
}

impl_integers!(i8);
impl_integers!(i16);
impl_integers!(i32);
impl_integers!(i64);
impl_integers!(i128);
impl_integers!(isize);
impl_integers!(u8);
impl_integers!(u16);
impl_integers!(u32);
impl_integers!(u64);
impl_integers!(u128);
impl_integers!(usize);

macro_rules! impl_floats {
    ($T:ty) => {
        impl CheckedOperator<$T> for Add {
            fn apply(lhs: &$T, rhs: &$T) -> Option<$T> {
                Some(lhs + rhs)
            }
        }
        impl CheckedOperator<$T> for Sub {
            fn apply(lhs: &$T, rhs: &$T) -> Option<$T> {
                Some(lhs - rhs)
            }
        }
        impl CheckedOperator<$T> for Mul {
            fn apply(lhs: &$T, rhs: &$T) -> Option<$T> {
                Some(lhs * rhs)
            }
        }
        impl CheckedOperator<$T> for Div {
            fn apply(lhs: &$T, rhs: &$T) -> Option<$T> {
                Some(lhs / rhs)
            }
        }
    };
}

impl_floats!(f16);
impl_floats!(f32);
impl_floats!(f64);
