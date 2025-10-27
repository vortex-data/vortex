// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arithmetic::buffer::{
    buffer_op, buffer_op_mut, buffer_op_mut_scalar, buffer_op_scalar,
};
use crate::arithmetic::{CheckedAdd, CheckedDiv, CheckedMul, CheckedSub};
use std::ops::BitAnd;
use vortex_dtype::NativePType;
use vortex_vector::{PVector, PVectorMut, VectorMutOps, VectorOps};

macro_rules! checked_op {
    ($Trait:ident, $op:tt) => {
        impl<T: NativePType + num_traits::$Trait> $Trait<&PVector<T>> for &PVector<T> {
            type Output = PVector<T>;

            fn $op(self, other: &PVector<T>) -> Option<Self::Output> {
                pvector_op(self, other, num_traits::$Trait::$op)
            }
        }

        impl<T: NativePType + num_traits::$Trait> $Trait<&T> for &PVector<T> {
            type Output = PVector<T>;

            fn $op(self, other: &T) -> Option<Self::Output> {
                pvector_op_scalar(self, other, num_traits::$Trait::$op)
            }
        }

        impl<T: NativePType + num_traits::$Trait> $Trait<&PVector<T>> for PVector<T> {
            type Output = PVector<T>;

            fn $op(self, other: &PVector<T>) -> Option<Self::Output> {
                pvector_op_inplace(self, other, num_traits::$Trait::$op)
            }
        }

        impl<T: NativePType + num_traits::$Trait> $Trait<&T> for PVector<T> {
            type Output = PVector<T>;

            fn $op(self, other: &T) -> Option<Self::Output> {
                pvector_op_inplace_scalar(self, other, num_traits::$Trait::$op)
            }
        }

        impl<T: NativePType + num_traits::$Trait> $Trait<&PVector<T>> for PVectorMut<T> {
            type Output = PVector<T>;

            fn $op(self, other: &PVector<T>) -> Option<Self::Output> {
                pvector_op_mut(self, other, num_traits::$Trait::$op)
            }
        }

        impl<T: NativePType + num_traits::$Trait> $Trait<&T> for PVectorMut<T> {
            type Output = PVector<T>;

            fn $op(self, other: &T) -> Option<Self::Output> {
                pvector_op_mut_scalar(self, other, num_traits::$Trait::$op)
            }
        }
    };
}

checked_op!(CheckedAdd, checked_add);
checked_op!(CheckedSub, checked_sub);
checked_op!(CheckedMul, checked_mul);
checked_op!(CheckedDiv, checked_div);

fn pvector_op_inplace<O, T>(lhs: PVector<T>, rhs: &PVector<T>, op: O) -> Option<PVector<T>>
where
    O: Fn(&T, &T) -> Option<T>,
    T: NativePType,
{
    match lhs.try_into_mut() {
        Ok(lhs) => pvector_op_mut(lhs, rhs, op),
        Err(lhs) => pvector_op(&lhs, rhs, op),
    }
}

fn pvector_op_mut<O, T>(lhs: PVectorMut<T>, rhs: &PVector<T>, op: O) -> Option<PVector<T>>
where
    O: Fn(&T, &T) -> Option<T>,
    T: NativePType,
{
    assert_eq!(lhs.len(), rhs.len());

    let (lhs_buffer, lhs_validity) = lhs.into_parts();

    // TODO(ngates): based on the true count of the validity, we may wish to short-circuit here
    //  or choose a different implementation.
    let validity = lhs_validity.freeze().bitand(rhs.validity());
    let elements = buffer_op_mut(lhs_buffer, rhs.elements(), op)?;

    Some(PVector::new(elements, validity))
}

fn pvector_op<O, T>(lhs: &PVector<T>, rhs: &PVector<T>, op: O) -> Option<PVector<T>>
where
    O: Fn(&T, &T) -> Option<T>,
    T: NativePType,
{
    assert_eq!(lhs.len(), rhs.len());

    // TODO(ngates): based on the true count of the validity, we may wish to short-circuit here
    //  or choose a different implementation.
    let validity = lhs.validity().bitand(rhs.validity());

    let elements = buffer_op(lhs.elements(), rhs.elements(), op)?;
    Some(PVector::new(elements, validity))
}

fn pvector_op_inplace_scalar<O, T>(lhs: PVector<T>, rhs: &T, op: O) -> Option<PVector<T>>
where
    O: Fn(&T, &T) -> Option<T>,
    T: NativePType,
{
    match lhs.try_into_mut() {
        Ok(lhs) => pvector_op_mut_scalar(lhs, rhs, op),
        Err(lhs) => pvector_op_scalar(&lhs, rhs, op),
    }
}

fn pvector_op_mut_scalar<O, T>(lhs: PVectorMut<T>, rhs: &T, op: O) -> Option<PVector<T>>
where
    O: Fn(&T, &T) -> Option<T>,
    T: NativePType,
{
    let (lhs_buffer, lhs_validity) = lhs.into_parts();
    let validity = lhs_validity.freeze();

    let elements = buffer_op_mut_scalar(lhs_buffer, rhs, op)?;

    Some(PVector::new(elements, validity))
}

fn pvector_op_scalar<O, T>(lhs: &PVector<T>, rhs: &T, op: O) -> Option<PVector<T>>
where
    O: Fn(&T, &T) -> Option<T>,
    T: NativePType,
{
    let buffer = buffer_op_scalar(lhs.elements(), rhs, op)?;
    Some(PVector::new(buffer, lhs.validity().clone()))
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_mask::Mask;
    use vortex_vector::PVector;

    use super::*;

    #[test]
    fn test_add_pvectors() {
        let left = PVector::new(buffer![1u32, 2, 3, 4], Mask::new_true(4));
        let right = PVector::new(buffer![10u32, 20, 30, 40], Mask::new_true(4));

        let result = left.checked_add(&right).unwrap();
        assert_eq!(result.elements(), &buffer![11u32, 22, 33, 44]);
    }

    #[test]
    fn test_add_scalar() {
        let vec = PVector::new(buffer![1u32, 2, 3, 4], Mask::new_true(4));
        let result = vec.checked_add(&10).unwrap();
        assert_eq!(result.elements(), &buffer![11u32, 12, 13, 14]);
    }

    #[test]
    fn test_add_with_nulls() {
        let left = PVector::new(buffer![1u32, 2, 3], Mask::from_iter([true, false, true]));
        let right = PVector::new(buffer![10u32, 20, 30], Mask::new_true(3));

        let result = left.checked_add(&right).unwrap();
        // Validity is AND'd, so if either side is null, result is null
        assert_eq!(result.validity(), &Mask::from_iter([true, false, true]));
        assert_eq!(result.elements(), &buffer![11u32, 22, 33]);
    }

    #[test]
    fn test_sub_pvectors() {
        let left = PVector::new(buffer![10u32, 20, 30, 40], Mask::new_true(4));
        let right = PVector::new(buffer![1u32, 2, 3, 4], Mask::new_true(4));

        let result = left.checked_sub(&right).unwrap();
        assert_eq!(result.elements(), &buffer![9u32, 18, 27, 36]);
    }

    #[test]
    fn test_sub_scalar() {
        let vec = PVector::new(buffer![10u32, 20, 30, 40], Mask::new_true(4));
        let result = vec.checked_sub(&5).unwrap();
        assert_eq!(result.elements(), &buffer![5u32, 15, 25, 35]);
    }

    #[test]
    fn test_mul_pvectors() {
        let left = PVector::new(buffer![2u32, 3, 4, 5], Mask::new_true(4));
        let right = PVector::new(buffer![10u32, 20, 30, 40], Mask::new_true(4));

        let result = left.checked_mul(&right).unwrap();
        assert_eq!(result.elements(), &buffer![20u32, 60, 120, 200]);
    }

    #[test]
    fn test_mul_scalar() {
        let vec = PVector::new(buffer![1u32, 2, 3, 4], Mask::new_true(4));
        let result = vec.checked_mul(&10).unwrap();
        assert_eq!(result.elements(), &buffer![10u32, 20, 30, 40]);
    }

    #[test]
    fn test_div_pvectors() {
        let left = PVector::new(buffer![100u32, 200, 300, 400], Mask::new_true(4));
        let right = PVector::new(buffer![10u32, 20, 30, 40], Mask::new_true(4));

        let result = left.checked_div(&right).unwrap();
        assert_eq!(result.elements(), &buffer![10u32, 10, 10, 10]);
    }

    #[test]
    fn test_div_scalar() {
        let vec = PVector::new(buffer![100u32, 200, 300, 400], Mask::new_true(4));
        let result = vec.checked_div(&10).unwrap();
        assert_eq!(result.elements(), &buffer![10u32, 20, 30, 40]);
    }

    #[test]
    fn test_overflow_returns_none() {
        let left = PVector::new(buffer![u8::MAX, 100], Mask::new_true(2));
        let right = PVector::new(buffer![1u8, 50], Mask::new_true(2));

        let result = left.checked_add(&right);
        assert!(result.is_none());
    }

    #[test]
    fn test_div_by_zero_returns_none() {
        let left = PVector::new(buffer![10u32, 20, 30], Mask::new_true(3));
        let right = PVector::new(buffer![2u32, 0, 3], Mask::new_true(3));

        let result = left.checked_div(&right);
        assert!(result.is_none());
    }

    #[test]
    fn test_scalar_preserves_validity() {
        let vec = PVector::new(buffer![1u32, 2, 3], Mask::from_iter([true, false, true]));
        let result = vec.checked_add(&10).unwrap();

        assert_eq!(result.validity(), &Mask::from_iter([true, false, true]));
        assert_eq!(result.elements(), &buffer![11u32, 12, 13]);
    }
}
