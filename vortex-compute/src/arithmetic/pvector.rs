// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;

use vortex_buffer::{Buffer, BufferMut};
use vortex_dtype::NativePType;
use vortex_vector::{PVector, PVectorMut, VectorMutOps, VectorOps};

use crate::arithmetic::{Checked, CheckedOperator};

/// Implementation that attempts to downcast to a mutable vector and operates in-place.
impl<Op, T> Checked<Op, &PVector<T>> for PVector<T>
where
    T: NativePType,
    Op: CheckedOperator<T>,
{
    type Output = PVector<T>;

    fn checked_op(self, rhs: &PVector<T>) -> Option<Self::Output> {
        match self.try_into_mut() {
            Ok(lhs) => Checked::<Op, _>::checked_op(lhs, rhs),
            Err(lhs) => Checked::<Op, _>::checked_op(&lhs, rhs),
        }
    }
}

/// Implementation that operates in-place over a mutable vector.
impl<Op, T> Checked<Op, &PVector<T>> for PVectorMut<T>
where
    T: NativePType,
    Op: CheckedOperator<T>,
    BufferMut<T>: for<'a> Checked<Op, &'a Buffer<T>, Output = Buffer<T>>,
{
    type Output = PVector<T>;

    fn checked_op(self, other: &PVector<T>) -> Option<Self::Output> {
        assert_eq!(self.len(), other.len());

        let (lhs_buffer, lhs_validity) = self.into_parts();

        // TODO(ngates): based on the true count of the validity, we may wish to short-circuit here
        //  or choose a different implementation.
        let validity = lhs_validity.freeze().bitand(other.validity());
        let elements = Checked::<Op, _>::checked_op(lhs_buffer, other.elements())?;

        Some(PVector::new(elements, validity))
    }
}

/// Implementation that allocates a new output vector.
impl<Op, T> Checked<Op, &PVector<T>> for &PVector<T>
where
    T: NativePType,
    Op: CheckedOperator<T>,
    for<'a> &'a Buffer<T>: Checked<Op, &'a Buffer<T>, Output = Buffer<T>>,
{
    type Output = PVector<T>;

    fn checked_op(self, rhs: &PVector<T>) -> Option<Self::Output> {
        assert_eq!(self.len(), rhs.len());

        // TODO(ngates): based on the true count of the validity, we may wish to short-circuit here
        //  or choose a different implementation.
        let validity = self.validity().bitand(rhs.validity());

        let elements = Checked::<Op, _>::checked_op(self.elements(), rhs.elements())?;
        Some(PVector::new(elements, validity))
    }
}

/// Implementation that attempts to downcast to a mutable vector and operates in-place against
/// a scalar RHS value.
impl<Op, T> Checked<Op, &T> for PVector<T>
where
    T: NativePType,
    Op: CheckedOperator<T>,
    PVectorMut<T>: for<'a> Checked<Op, &'a T, Output = PVector<T>>,
{
    type Output = PVector<T>;

    fn checked_op(self, rhs: &T) -> Option<Self::Output> {
        match self.try_into_mut() {
            Ok(lhs) => Checked::<Op, _>::checked_op(lhs, rhs),
            Err(lhs) => Checked::<Op, _>::checked_op(&lhs, rhs),
        }
    }
}

/// Implementation that operates in-place over a mutable vector against a scalar RHS value.
impl<Op, T> Checked<Op, &T> for PVectorMut<T>
where
    T: NativePType,
    Op: CheckedOperator<T>,
    BufferMut<T>: for<'a> Checked<Op, &'a T, Output = Buffer<T>>,
{
    type Output = PVector<T>;

    fn checked_op(self, rhs: &T) -> Option<Self::Output> {
        let (lhs_buffer, lhs_validity) = self.into_parts();
        let validity = lhs_validity.freeze();

        let elements = Checked::<Op, _>::checked_op(lhs_buffer, rhs)?;

        Some(PVector::new(elements, validity))
    }
}

/// Implementation that allocates a new output vector against a scalar RHS value.
impl<Op, T> Checked<Op, &T> for &PVector<T>
where
    T: NativePType,
    Op: CheckedOperator<T>,
    for<'a> &'a Buffer<T>: Checked<Op, &'a T, Output = Buffer<T>>,
{
    type Output = PVector<T>;

    fn checked_op(self, rhs: &T) -> Option<Self::Output> {
        let buffer = Checked::<Op, _>::checked_op(self.elements(), rhs)?;
        Some(PVector::new(buffer, self.validity().clone()))
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_mask::Mask;
    use vortex_vector::{PVector, VectorOps};

    use crate::arithmetic::{CheckedAdd, CheckedDiv, CheckedMul, CheckedSub};

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
