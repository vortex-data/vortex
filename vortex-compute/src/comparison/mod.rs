// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Comparison operations for Vortex vectors.

use vortex_dtype::half::f16;

mod bool;
mod collection;
mod primitive_scalar;
mod primitive_vector;
mod pvector;

/// Trait for comparison operations.
pub trait Compare<Op, Rhs = Self> {
    /// The result type after performing the operation.
    type Output;

    /// Perform the comparison operation.
    fn compare(self, rhs: Rhs) -> Self::Output;
}

/// Trait for comparison operators.
pub trait ComparisonOperator<T> {
    /// Apply the operator to the two operands.
    fn apply(a: &T, b: &T) -> bool;
}

/// A marker type for equality comparison operations.
pub struct Equal;
/// A marker type for inequality comparison operations.
pub struct NotEqual;
/// A marker type for less-than comparison operations.
pub struct LessThan;
/// A marker type for less-than-or-equal comparison operations.
pub struct LessThanOrEqual;
/// A marker type for greater-than comparison operations.
pub struct GreaterThan;
/// A marker type for greater-than-or-equal comparison operations.
pub struct GreaterThanOrEqual;

/// Marker trait for comparable items.
pub trait ComparableItem {
    /// Check if two items are equal.
    fn is_equal(lhs: &Self, rhs: &Self) -> bool;

    /// Check if the `lhs` item is less than the `rhs` item.
    fn is_less_than(lhs: &Self, rhs: &Self) -> bool;
}

impl<T: ComparableItem> ComparisonOperator<T> for Equal {
    fn apply(a: &T, b: &T) -> bool {
        T::is_equal(a, b)
    }
}

impl<T: ComparableItem> ComparisonOperator<T> for NotEqual {
    fn apply(a: &T, b: &T) -> bool {
        !T::is_equal(a, b)
    }
}

impl<T: ComparableItem> ComparisonOperator<T> for LessThan {
    fn apply(a: &T, b: &T) -> bool {
        T::is_less_than(a, b)
    }
}

impl<T: ComparableItem> ComparisonOperator<T> for GreaterThanOrEqual {
    fn apply(a: &T, b: &T) -> bool {
        !T::is_less_than(a, b)
    }
}

impl<T: ComparableItem> ComparisonOperator<T> for GreaterThan {
    fn apply(a: &T, b: &T) -> bool {
        T::is_less_than(b, a)
    }
}

impl<T: ComparableItem> ComparisonOperator<T> for LessThanOrEqual {
    fn apply(a: &T, b: &T) -> bool {
        !T::is_less_than(b, a)
    }
}

macro_rules! impl_integer {
    ($T:ty) => {
        impl ComparableItem for $T {
            #[inline(always)]
            fn is_equal(lhs: &Self, rhs: &Self) -> bool {
                lhs == rhs
            }

            #[inline(always)]
            fn is_less_than(lhs: &Self, rhs: &Self) -> bool {
                lhs < rhs
            }
        }
    };
}

impl_integer!(i8);
impl_integer!(i16);
impl_integer!(i32);
impl_integer!(i64);
impl_integer!(i128);
impl_integer!(u8);
impl_integer!(u16);
impl_integer!(u32);
impl_integer!(u64);
impl_integer!(u128);

macro_rules! impl_float {
    ($T:ty) => {
        impl ComparableItem for $T {
            #[inline(always)]
            fn is_equal(lhs: &Self, rhs: &Self) -> bool {
                lhs.to_bits().eq(&rhs.to_bits())
            }

            #[inline(always)]
            fn is_less_than(lhs: &Self, rhs: &Self) -> bool {
                lhs.total_cmp(rhs).is_lt()
            }
        }
    };
}

impl_float!(f16);
impl_float!(f32);
impl_float!(f64);

/// Dispatches a comparison operation based on a runtime operator value.
///
/// This macro allows you to call `Compare::<Op>::compare(lhs, rhs)` where `Op` is determined
/// at runtime from an operator enum variant.
///
/// # Arguments
///
/// * `$op` - An expression that evaluates to an operator enum (e.g., `Operator::Eq`)
/// * `$lhs` - The left-hand side operand
/// * `$rhs` - The right-hand side operand
/// * `$Eq`, `$NotEq`, etc. - The enum variants to match against
///
/// # Example
///
/// ```ignore
/// use vortex_compute::compare_op;
/// use vortex_compute::comparison::Compare;
///
/// let result = compare_op!(
///     op,
///     lhs,
///     rhs,
///     Operator::Eq,
///     Operator::NotEq,
///     Operator::Lt,
///     Operator::Lte,
///     Operator::Gt,
///     Operator::Gte
/// );
/// ```
#[macro_export]
macro_rules! compare_op {
    ($op:expr, $lhs:expr, $rhs:expr, $Eq:pat, $NotEq:pat, $Lt:pat, $Lte:pat, $Gt:pat, $Gte:pat) => {
        match $op {
            $Eq => $crate::comparison::Compare::<$crate::comparison::Equal>::compare($lhs, $rhs),
            $NotEq => {
                $crate::comparison::Compare::<$crate::comparison::NotEqual>::compare($lhs, $rhs)
            }
            $Lt => $crate::comparison::Compare::<$crate::comparison::LessThan>::compare($lhs, $rhs),
            $Lte => $crate::comparison::Compare::<$crate::comparison::LessThanOrEqual>::compare(
                $lhs, $rhs,
            ),
            $Gt => {
                $crate::comparison::Compare::<$crate::comparison::GreaterThan>::compare($lhs, $rhs)
            }
            $Gte => $crate::comparison::Compare::<$crate::comparison::GreaterThanOrEqual>::compare(
                $lhs, $rhs,
            ),
            _ => unreachable!("Not a comparison operator"),
        }
    };
}
