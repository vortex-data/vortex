// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;

use crate::comparison::{Compare, ComparisonOperator};

/// Adapter to implement `Compare` for any `ComparableCollection`.
pub(crate) struct ComparableCollectionAdapter<C>(pub C);

impl<Op, C> Compare<Op> for ComparableCollectionAdapter<C>
where
    C: ComparableCollection,
    Op: ComparisonOperator<C::Item>,
{
    type Output = BitBuffer;

    fn compare(self, rhs: Self) -> Self::Output {
        assert_eq!(self.0.len(), rhs.0.len());

        BitBuffer::from_iter((0..self.0.len()).map(|i| {
            let left = unsafe { self.0.item_unchecked(i) };
            let right = unsafe { rhs.0.item_unchecked(i) };
            Op::apply(&left, &right)
        }))
    }
}

/// Marker trait for comparable collections.
pub trait ComparableCollection {
    /// The item type that can be compared.
    type Item;

    /// Get the length of the comparable collection.
    fn len(&self) -> usize;

    /// Get the item at the specified index without bounds checking.
    unsafe fn item_unchecked(&self, index: usize) -> Self::Item;
}

impl<T: Copy> ComparableCollection for &[T] {
    type Item = T;

    fn len(&self) -> usize {
        <[T]>::len(self)
    }

    unsafe fn item_unchecked(&self, index: usize) -> Self::Item {
        unsafe { *self.get_unchecked(index) }
    }
}

impl<Op, T> Compare<Op> for &[T]
where
    T: Copy,
    Op: ComparisonOperator<T>,
{
    type Output = BitBuffer;

    fn compare(self, rhs: Self) -> Self::Output {
        Compare::<Op>::compare(
            ComparableCollectionAdapter(self),
            ComparableCollectionAdapter(rhs),
        )
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::bitbuffer;

    use super::*;
    use crate::comparison::{Equal, GreaterThan, LessThan, NotEqual};

    #[test]
    fn test_slice_equal() {
        let left: &[u32] = &[1, 2, 3, 4];
        let right: &[u32] = &[1, 2, 5, 4];

        let result = Compare::<Equal>::compare(left, right);
        assert_eq!(result, bitbuffer![1 1 0 1]);
    }

    #[test]
    fn test_slice_not_equal() {
        let left: &[u32] = &[1, 2, 3, 4];
        let right: &[u32] = &[1, 2, 5, 4];

        let result = Compare::<NotEqual>::compare(left, right);
        assert_eq!(result, bitbuffer![0 0 1 0]);
    }

    #[test]
    fn test_slice_less_than() {
        let left: &[u32] = &[1, 2, 3, 4];
        let right: &[u32] = &[2, 2, 1, 5];

        let result = Compare::<LessThan>::compare(left, right);
        assert_eq!(result, bitbuffer![1 0 0 1]);
    }

    #[test]
    fn test_slice_greater_than() {
        let left: &[u32] = &[3, 2, 1, 5];
        let right: &[u32] = &[1, 2, 3, 4];

        let result = Compare::<GreaterThan>::compare(left, right);
        assert_eq!(result, bitbuffer![1 0 0 1]);
    }
}
