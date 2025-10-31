// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;

use vortex_dtype::NativePType;
use vortex_vector::{BoolVector, PVector, VectorOps};

use crate::comparison::collection::ComparableCollectionAdapter;
use crate::comparison::{Compare, ComparisonOperator};

impl<Op, T> Compare<Op> for &PVector<T>
where
    T: NativePType,
    Op: ComparisonOperator<T>,
{
    type Output = BoolVector;

    fn compare(self, rhs: &PVector<T>) -> Self::Output {
        let validity = self.validity().bitand(rhs.validity());

        // TODO(ngates): match on density of validity mask to choose optimal implementation

        let bits = Compare::<Op>::compare(
            ComparableCollectionAdapter(self.elements().as_slice()),
            ComparableCollectionAdapter(rhs.elements().as_slice()),
        );

        BoolVector::new(bits, validity)
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::{bitbuffer, buffer};
    use vortex_mask::Mask;

    use super::*;
    use crate::comparison::{
        Equal, GreaterThan, GreaterThanOrEqual, LessThan, LessThanOrEqual, NotEqual,
    };

    #[test]
    fn test_equal() {
        let left = PVector::new(buffer![1u32, 2, 3, 4], Mask::new_true(4));
        let right = PVector::new(buffer![1u32, 2, 5, 4], Mask::new_true(4));

        let result = Compare::<Equal>::compare(&left, &right);
        assert_eq!(result.bits(), &bitbuffer![1 1 0 1]);
    }

    #[test]
    fn test_not_equal() {
        let left = PVector::new(buffer![1u32, 2, 3, 4], Mask::new_true(4));
        let right = PVector::new(buffer![1u32, 2, 5, 4], Mask::new_true(4));

        let result = Compare::<NotEqual>::compare(&left, &right);
        assert_eq!(result.bits(), &bitbuffer![0 0 1 0]);
    }

    #[test]
    fn test_less_than() {
        let left = PVector::new(buffer![1u32, 2, 3, 4], Mask::new_true(4));
        let right = PVector::new(buffer![2u32, 2, 1, 5], Mask::new_true(4));

        let result = Compare::<LessThan>::compare(&left, &right);
        assert_eq!(result.bits(), &bitbuffer![1 0 0 1]);
    }

    #[test]
    fn test_less_than_or_equal() {
        let left = PVector::new(buffer![1u32, 2, 3, 4], Mask::new_true(4));
        let right = PVector::new(buffer![2u32, 2, 1, 5], Mask::new_true(4));

        let result = Compare::<LessThanOrEqual>::compare(&left, &right);
        assert_eq!(result.bits(), &bitbuffer![1 1 0 1]);
    }

    #[test]
    fn test_greater_than() {
        let left = PVector::new(buffer![3u32, 2, 1, 5], Mask::new_true(4));
        let right = PVector::new(buffer![1u32, 2, 3, 4], Mask::new_true(4));

        let result = Compare::<GreaterThan>::compare(&left, &right);
        assert_eq!(result.bits(), &bitbuffer![1 0 0 1]);
    }

    #[test]
    fn test_greater_than_or_equal() {
        let left = PVector::new(buffer![3u32, 2, 1, 5], Mask::new_true(4));
        let right = PVector::new(buffer![1u32, 2, 3, 4], Mask::new_true(4));

        let result = Compare::<GreaterThanOrEqual>::compare(&left, &right);
        assert_eq!(result.bits(), &bitbuffer![1 1 0 1]);
    }

    #[test]
    fn test_compare_with_nulls() {
        let left = PVector::new(buffer![1u32, 2, 3], Mask::from_iter([true, false, true]));
        let right = PVector::new(buffer![1u32, 2, 3], Mask::new_true(3));

        let result = Compare::<Equal>::compare(&left, &right);
        // Validity is AND'd, so if either side is null, result validity is null
        assert_eq!(result.validity(), &Mask::from_iter([true, false, true]));
        assert_eq!(result.bits(), &bitbuffer![1 1 1]);
    }
}
