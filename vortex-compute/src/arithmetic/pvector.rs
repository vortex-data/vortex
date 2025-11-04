// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;

use vortex_buffer::{Buffer, BufferMut};
use vortex_dtype::NativePType;
use vortex_vector::primitive::{PVector, PVectorMut};
use vortex_vector::{VectorMutOps, VectorOps};

use crate::arithmetic::{Arithmetic, Operator};

/// Implementation that attempts to downcast to a mutable vector and operates in-place.
impl<Op, T> Arithmetic<Op, &PVector<T>> for PVector<T>
where
    T: NativePType,
    Op: Operator<T>,
{
    type Output = PVector<T>;

    fn eval(self, rhs: &PVector<T>) -> Self::Output {
        match self.try_into_mut() {
            Ok(lhs) => Arithmetic::<Op, _>::eval(lhs, rhs),
            Err(lhs) => Arithmetic::<Op, _>::eval(&lhs, rhs),
        }
    }
}

/// Implementation that operates in-place over a mutable vector.
impl<Op, T> Arithmetic<Op, &PVector<T>> for PVectorMut<T>
where
    T: NativePType,
    Op: Operator<T>,
    BufferMut<T>: for<'a> Arithmetic<Op, &'a Buffer<T>, Output = Buffer<T>>,
{
    type Output = PVector<T>;

    fn eval(self, other: &PVector<T>) -> Self::Output {
        assert_eq!(self.len(), other.len());

        let (lhs_buffer, lhs_validity) = self.into_parts();

        // TODO(ngates): based on the true count of the validity, we may wish to short-circuit here
        //  or choose a different implementation.
        let validity = lhs_validity.freeze().bitand(other.validity());
        let elements = Arithmetic::<Op, _>::eval(lhs_buffer, other.elements());

        PVector::new(elements, validity)
    }
}

/// Implementation that allocates a new output vector.
impl<Op, T> Arithmetic<Op, &PVector<T>> for &PVector<T>
where
    T: NativePType,
    Op: Operator<T>,
    for<'a> &'a Buffer<T>: Arithmetic<Op, &'a Buffer<T>, Output = Buffer<T>>,
{
    type Output = PVector<T>;

    fn eval(self, rhs: &PVector<T>) -> Self::Output {
        assert_eq!(self.len(), rhs.len());

        // TODO(ngates): based on the true count of the validity, we may wish to short-circuit here
        //  or choose a different implementation.
        let validity = self.validity().bitand(rhs.validity());

        let elements = Arithmetic::<Op, _>::eval(self.elements(), rhs.elements());
        PVector::new(elements, validity)
    }
}

/// Implementation that attempts to downcast to a mutable vector and operates in-place against
/// a scalar RHS value.
impl<Op, T> Arithmetic<Op, &T> for PVector<T>
where
    T: NativePType,
    Op: Operator<T>,
    PVectorMut<T>: for<'a> Arithmetic<Op, &'a T, Output = PVector<T>>,
{
    type Output = PVector<T>;

    fn eval(self, rhs: &T) -> Self::Output {
        match self.try_into_mut() {
            Ok(lhs) => Arithmetic::<Op, _>::eval(lhs, rhs),
            Err(lhs) => Arithmetic::<Op, _>::eval(&lhs, rhs),
        }
    }
}

/// Implementation that operates in-place over a mutable vector against a scalar RHS value.
impl<Op, T> Arithmetic<Op, &T> for PVectorMut<T>
where
    T: NativePType,
    Op: Operator<T>,
    BufferMut<T>: for<'a> Arithmetic<Op, &'a T, Output = Buffer<T>>,
{
    type Output = PVector<T>;

    fn eval(self, rhs: &T) -> Self::Output {
        let (lhs_buffer, lhs_validity) = self.into_parts();
        let validity = lhs_validity.freeze();

        let elements = Arithmetic::<Op, _>::eval(lhs_buffer, rhs);

        PVector::new(elements, validity)
    }
}

/// Implementation that allocates a new output vector against a scalar RHS value.
impl<Op, T> Arithmetic<Op, &T> for &PVector<T>
where
    T: NativePType,
    Op: Operator<T>,
    for<'a> &'a Buffer<T>: Arithmetic<Op, &'a T, Output = Buffer<T>>,
{
    type Output = PVector<T>;

    fn eval(self, rhs: &T) -> Self::Output {
        let buffer = Arithmetic::<Op, _>::eval(self.elements(), rhs);
        PVector::new(buffer, self.validity().clone())
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_mask::Mask;
    use vortex_vector::VectorOps;
    use vortex_vector::primitive::PVector;

    use crate::arithmetic::{Arithmetic, WrappingAdd, WrappingMul, WrappingSub};

    #[test]
    fn test_add_pvectors() {
        let left = PVector::new(buffer![1u32, 2, 3, 4], Mask::new_true(4));
        let right = PVector::new(buffer![10u32, 20, 30, 40], Mask::new_true(4));

        let result = Arithmetic::<WrappingAdd, _>::eval(left, &right);
        assert_eq!(result.elements(), &buffer![11u32, 22, 33, 44]);
    }

    #[test]
    fn test_add_scalar() {
        let vec = PVector::new(buffer![1u32, 2, 3, 4], Mask::new_true(4));
        let result = Arithmetic::<WrappingAdd, _>::eval(vec, &10);
        assert_eq!(result.elements(), &buffer![11u32, 12, 13, 14]);
    }

    #[test]
    fn test_add_with_nulls() {
        let left = PVector::new(buffer![1u32, 2, 3], Mask::from_iter([true, false, true]));
        let right = PVector::new(buffer![10u32, 20, 30], Mask::new_true(3));

        let result = Arithmetic::<WrappingAdd, _>::eval(left, &right);
        // Validity is AND'd, so if either side is null, result is null
        assert_eq!(result.validity(), &Mask::from_iter([true, false, true]));
        assert_eq!(result.elements(), &buffer![11u32, 22, 33]);
    }

    #[test]
    fn test_sub_pvectors() {
        let left = PVector::new(buffer![10u32, 20, 30, 40], Mask::new_true(4));
        let right = PVector::new(buffer![1u32, 2, 3, 4], Mask::new_true(4));

        let result = Arithmetic::<WrappingSub, _>::eval(left, &right);
        assert_eq!(result.elements(), &buffer![9u32, 18, 27, 36]);
    }

    #[test]
    fn test_sub_scalar() {
        let vec = PVector::new(buffer![10u32, 20, 30, 40], Mask::new_true(4));
        let result = Arithmetic::<WrappingSub, _>::eval(vec, &5);
        assert_eq!(result.elements(), &buffer![5u32, 15, 25, 35]);
    }

    #[test]
    fn test_mul_pvectors() {
        let left = PVector::new(buffer![2u32, 3, 4, 5], Mask::new_true(4));
        let right = PVector::new(buffer![10u32, 20, 30, 40], Mask::new_true(4));

        let result = Arithmetic::<WrappingMul, _>::eval(left, &right);
        assert_eq!(result.elements(), &buffer![20u32, 60, 120, 200]);
    }

    #[test]
    fn test_mul_scalar() {
        let vec = PVector::new(buffer![1u32, 2, 3, 4], Mask::new_true(4));
        let result = Arithmetic::<WrappingMul, _>::eval(vec, &10);
        assert_eq!(result.elements(), &buffer![10u32, 20, 30, 40]);
    }

    #[test]
    fn test_scalar_preserves_validity() {
        let vec = PVector::new(buffer![1u32, 2, 3], Mask::from_iter([true, false, true]));
        let result = Arithmetic::<WrappingAdd, _>::eval(vec, &10);

        assert_eq!(result.validity(), &Mask::from_iter([true, false, true]));
        assert_eq!(result.elements(), &buffer![11u32, 12, 13]);
    }
}
