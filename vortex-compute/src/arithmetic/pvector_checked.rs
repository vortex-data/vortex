// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;

use vortex_buffer::{Buffer, BufferMut};
use vortex_dtype::NativePType;
use vortex_vector::primitive::{PVector, PVectorMut};
use vortex_vector::{VectorMutOps, VectorOps};

use crate::arithmetic::CheckedArithmetic;

/// Implementation that attempts to downcast to a mutable vector and operates in-place.
impl<Op, T> CheckedArithmetic<Op, &PVector<T>> for PVector<T>
where
    T: NativePType,
    PVectorMut<T>: for<'a> CheckedArithmetic<Op, &'a PVector<T>, Output = PVector<T>>,
    for<'a> &'a PVector<T>: CheckedArithmetic<Op, &'a PVector<T>, Output = PVector<T>>,
{
    type Output = PVector<T>;

    fn checked_eval(self, rhs: &PVector<T>) -> Option<Self::Output> {
        match self.try_into_mut() {
            Ok(lhs) => CheckedArithmetic::<Op, _>::checked_eval(lhs, rhs),
            Err(lhs) => CheckedArithmetic::<Op, _>::checked_eval(&lhs, rhs),
        }
    }
}

/// Implementation that operates in-place over a mutable vector.
impl<Op, T> CheckedArithmetic<Op, &PVector<T>> for PVectorMut<T>
where
    T: NativePType,
    BufferMut<T>: for<'a> CheckedArithmetic<Op, &'a Buffer<T>, Output = Buffer<T>>,
{
    type Output = PVector<T>;

    fn checked_eval(self, other: &PVector<T>) -> Option<Self::Output> {
        assert_eq!(self.len(), other.len());

        let (lhs_buffer, lhs_validity) = self.into_parts();

        // TODO(ngates): based on the true count of the validity, we may wish to short-circuit here
        //  or choose a different implementation.
        let validity = lhs_validity.freeze().bitand(other.validity());
        let elements = CheckedArithmetic::<Op, _>::checked_eval(lhs_buffer, other.elements())?;

        Some(PVector::new(elements, validity))
    }
}

/// Implementation that allocates a new output vector.
impl<Op, T> CheckedArithmetic<Op, &PVector<T>> for &PVector<T>
where
    T: NativePType,
    for<'a> &'a Buffer<T>: CheckedArithmetic<Op, &'a Buffer<T>, Output = Buffer<T>>,
{
    type Output = PVector<T>;

    fn checked_eval(self, rhs: &PVector<T>) -> Option<Self::Output> {
        assert_eq!(self.len(), rhs.len());

        // TODO(ngates): based on the true count of the validity, we may wish to short-circuit here
        //  or choose a different implementation.
        let validity = self.validity().bitand(rhs.validity());

        let elements = CheckedArithmetic::<Op, _>::checked_eval(self.elements(), rhs.elements())?;
        Some(PVector::new(elements, validity))
    }
}

/// Implementation that attempts to downcast to a mutable vector and operates in-place against
/// a scalar RHS value.
impl<Op, T> CheckedArithmetic<Op, &T> for PVector<T>
where
    T: NativePType,
    PVectorMut<T>: for<'a> CheckedArithmetic<Op, &'a T, Output = PVector<T>>,
    for<'a> &'a PVector<T>: CheckedArithmetic<Op, &'a T, Output = PVector<T>>,
{
    type Output = PVector<T>;

    fn checked_eval(self, rhs: &T) -> Option<Self::Output> {
        match self.try_into_mut() {
            Ok(lhs) => CheckedArithmetic::<Op, _>::checked_eval(lhs, rhs),
            Err(lhs) => CheckedArithmetic::<Op, _>::checked_eval(&lhs, rhs),
        }
    }
}

/// Implementation that operates in-place over a mutable vector against a scalar RHS value.
impl<Op, T> CheckedArithmetic<Op, &T> for PVectorMut<T>
where
    T: NativePType,
    BufferMut<T>: for<'a> CheckedArithmetic<Op, &'a T, Output = Buffer<T>>,
{
    type Output = PVector<T>;

    fn checked_eval(self, rhs: &T) -> Option<Self::Output> {
        let (lhs_buffer, lhs_validity) = self.into_parts();
        let validity = lhs_validity.freeze();

        let elements = CheckedArithmetic::<Op, _>::checked_eval(lhs_buffer, rhs)?;

        Some(PVector::new(elements, validity))
    }
}

/// Implementation that allocates a new output vector against a scalar RHS value.
impl<Op, T> CheckedArithmetic<Op, &T> for &PVector<T>
where
    T: NativePType,
    for<'a> &'a Buffer<T>: CheckedArithmetic<Op, &'a T, Output = Buffer<T>>,
{
    type Output = PVector<T>;

    fn checked_eval(self, rhs: &T) -> Option<Self::Output> {
        let buffer = CheckedArithmetic::<Op, _>::checked_eval(self.elements(), rhs)?;
        Some(PVector::new(buffer, self.validity().clone()))
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_mask::Mask;
    use vortex_vector::VectorOps;
    use vortex_vector::primitive::PVector;

    use crate::arithmetic::{Add, CheckedArithmetic, Div, Mul, Sub};

    #[test]
    fn test_add_pvectors() {
        let left = PVector::new(buffer![1u32, 2, 3, 4], Mask::new_true(4));
        let right = PVector::new(buffer![10u32, 20, 30, 40], Mask::new_true(4));

        let result = CheckedArithmetic::<Add, _>::checked_eval(left, &right).unwrap();
        assert_eq!(result.elements(), &buffer![11u32, 22, 33, 44]);
    }

    #[test]
    fn test_add_scalar() {
        let vec = PVector::new(buffer![1u32, 2, 3, 4], Mask::new_true(4));
        let result = CheckedArithmetic::<Add, _>::checked_eval(vec, &10).unwrap();
        assert_eq!(result.elements(), &buffer![11u32, 12, 13, 14]);
    }

    #[test]
    fn test_add_with_nulls() {
        let left = PVector::new(buffer![1u32, 2, 3], Mask::from_iter([true, false, true]));
        let right = PVector::new(buffer![10u32, 20, 30], Mask::new_true(3));

        let result = CheckedArithmetic::<Add, _>::checked_eval(left, &right).unwrap();
        // Validity is AND'd, so if either side is null, result is null
        assert_eq!(result.validity(), &Mask::from_iter([true, false, true]));
        assert_eq!(result.elements(), &buffer![11u32, 22, 33]);
    }

    #[test]
    fn test_sub_pvectors() {
        let left = PVector::new(buffer![10u32, 20, 30, 40], Mask::new_true(4));
        let right = PVector::new(buffer![1u32, 2, 3, 4], Mask::new_true(4));

        let result = CheckedArithmetic::<Sub, _>::checked_eval(left, &right).unwrap();
        assert_eq!(result.elements(), &buffer![9u32, 18, 27, 36]);
    }

    #[test]
    fn test_sub_scalar() {
        let vec = PVector::new(buffer![10u32, 20, 30, 40], Mask::new_true(4));
        let result = CheckedArithmetic::<Sub, _>::checked_eval(vec, &5).unwrap();
        assert_eq!(result.elements(), &buffer![5u32, 15, 25, 35]);
    }

    #[test]
    fn test_mul_pvectors() {
        let left = PVector::new(buffer![2u32, 3, 4, 5], Mask::new_true(4));
        let right = PVector::new(buffer![10u32, 20, 30, 40], Mask::new_true(4));

        let result = CheckedArithmetic::<Mul, _>::checked_eval(left, &right).unwrap();
        assert_eq!(result.elements(), &buffer![20u32, 60, 120, 200]);
    }

    #[test]
    fn test_mul_scalar() {
        let vec = PVector::new(buffer![1u32, 2, 3, 4], Mask::new_true(4));
        let result = CheckedArithmetic::<Mul, _>::checked_eval(vec, &10).unwrap();
        assert_eq!(result.elements(), &buffer![10u32, 20, 30, 40]);
    }

    #[test]
    fn test_div_pvectors() {
        let left = PVector::new(buffer![100u32, 200, 300, 400], Mask::new_true(4));
        let right = PVector::new(buffer![10u32, 20, 30, 40], Mask::new_true(4));

        let result = CheckedArithmetic::<Div, _>::checked_eval(left, &right).unwrap();
        assert_eq!(result.elements(), &buffer![10u32, 10, 10, 10]);
    }

    #[test]
    fn test_div_scalar() {
        let vec = PVector::new(buffer![100u32, 200, 300, 400], Mask::new_true(4));
        let result = CheckedArithmetic::<Div, _>::checked_eval(vec, &10).unwrap();
        assert_eq!(result.elements(), &buffer![10u32, 20, 30, 40]);
    }

    #[test]
    fn test_overflow_returns_none() {
        let left = PVector::new(buffer![u8::MAX, 100], Mask::new_true(2));
        let right = PVector::new(buffer![1u8, 50], Mask::new_true(2));

        let result = CheckedArithmetic::<Add, _>::checked_eval(left, &right);
        assert!(result.is_none());
    }

    #[test]
    fn test_div_by_zero_returns_none() {
        let left = PVector::new(buffer![10u32, 20, 30], Mask::new_true(3));
        let right = PVector::new(buffer![2u32, 0, 3], Mask::new_true(3));

        let result = CheckedArithmetic::<Div, _>::checked_eval(left, &right);
        assert!(result.is_none());
    }

    #[test]
    fn test_scalar_preserves_validity() {
        let vec = PVector::new(buffer![1u32, 2, 3], Mask::from_iter([true, false, true]));
        let result = CheckedArithmetic::<Add, _>::checked_eval(vec, &10).unwrap();

        assert_eq!(result.validity(), &Mask::from_iter([true, false, true]));
        assert_eq!(result.elements(), &buffer![11u32, 12, 13]);
    }
}
