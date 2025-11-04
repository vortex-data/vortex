// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Helper methods for [`PVectorMut<T>`] that mimic the behavior of [`std::vec::Vec`].

use vortex_buffer::BufferMut;
use vortex_dtype::NativePType;
use vortex_mask::MaskMut;

use crate::VectorMutOps;
use crate::primitive::PVectorMut;

/// Point operations for [`PVectorMut`].
impl<T: NativePType> PVectorMut<T> {
    /// Gets a nullable element at the given index, panicking on out-of-bounds.
    ///
    /// If the element at the given index is null, returns `None`. Otherwise, returns `Some(x)`,
    /// where `x: T`.
    ///
    /// Note that this `get` method is different from the standard library [`slice::get`], which
    /// returns `None` if the index is out of bounds. This method will panic if the index is out of
    /// bounds, and return `None` if the elements is null.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn get(&self, index: usize) -> Option<T> {
        self.validity.value(index).then(|| self.elements[index])
    }

    /// Appends an element to the back of the vector.
    ///
    /// The element is treated as valid.
    pub fn push(&mut self, value: T) {
        self.elements.push(value);
        self.validity.append_n(true, 1);
    }

    /// Pushes a value without bounds checking or validity updates.
    ///
    /// # Safety
    ///
    /// The caller must ensure that there is sufficient capacity in both elements and validity
    /// buffers.
    #[inline]
    pub unsafe fn push_unchecked(&mut self, value: T) {
        // SAFETY: The caller guarantees there is sufficient capacity in the elements buffer,
        // so we can write to the spare capacity and increment the length without bounds checks.
        unsafe {
            self.elements.spare_capacity_mut()[0].write(value);
            self.elements.set_len(self.len() + 1);
        }
        self.validity.append_n(true, 1);
    }

    /// Appends an optional element to the back of the vector, where `None` represents a null
    /// element.
    pub fn push_opt(&mut self, value: Option<T>) {
        if let Some(value) = value {
            self.push(value);
        } else {
            self.elements.push(T::default());
            self.validity.append_n(false, 1);
        }
    }
}

impl<T: NativePType> AsRef<[T]> for PVectorMut<T> {
    /// Returns an immutable slice over the internal mutable buffer with elements of type `T`.
    ///
    /// Note that this slice may contain garbage data where the [`validity()`] mask from the frozen
    /// [`PVector`](crate::primitive::PVector) type states that an element is invalid.
    ///
    /// The caller should check the frozen [`validity()`] before performing any operations.
    ///
    /// [`validity()`]: crate::VectorOps::validity
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.elements.as_slice()
    }
}

impl<T: NativePType> AsMut<[T]> for PVectorMut<T> {
    /// Returns a mutable slice over the internal mutable buffer with elements of type `T`.
    ///
    /// Note that this slice may contain garbage data where the [`validity()`] mask from the frozen
    /// [`PVector`](crate::primitive::PVector) type states that an element is invalid.
    ///
    /// The caller should check the frozen [`validity()`] before performing any operations.
    ///
    /// [`validity()`]: crate::VectorOps::validity
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.elements.as_mut_slice()
    }
}

/// Batch operations for [`PVectorMut`].
impl<T: NativePType> PVectorMut<T> {
    /// Returns the internal [`BufferMut`] of the [`PVectorMut`].
    ///
    /// Note that the internal buffer may hold garbage data in place of nulls. That information is
    /// tracked by the [`validity()`](Self::validity).
    #[inline]
    pub fn elements(&self) -> &BufferMut<T> {
        &self.elements
    }

    /// Returns the validity of the [`PVectorMut`].
    #[inline]
    pub fn validity(&self) -> &MaskMut {
        &self.validity
    }

    /// Resizes the `Vec` in-place so that `len` is equal to `new_len`.
    ///
    /// If `new_len` is greater than `len`, the `Vec` is extended by the difference, with each
    /// additional slot filled with `value`, where `None` represent a null.
    ///
    /// If `new_len` is less than `len`, the `Vec` is simply truncated.
    pub fn resize(&mut self, new_len: usize, value: Option<T>) {
        let current_len = self.len();

        if new_len < current_len {
            self.truncate(new_len);
        } else {
            let additional = new_len - current_len;

            match value {
                Some(value) => {
                    self.elements.push_n(value, additional);
                    self.validity.append_n(true, additional);
                }
                None => {
                    self.elements.push_n(T::default(), additional);
                    self.validity.append_n(false, additional);
                }
            }
        }
    }

    /// Clear the vector, removing all elements.
    pub fn clear(&mut self) {
        self.elements.clear();
        self.validity.clear();
    }

    /// Shortens the vector, keeping the first `len` elements.
    pub fn truncate(&mut self, len: usize) {
        self.elements.truncate(len);
        self.validity.truncate(len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VectorOps;

    #[test]
    fn test_get_methods() {
        let vec = PVectorMut::from_iter([Some(1), None, Some(3), None, Some(5)]);

        // Test get_checked - bounds and nulls.
        assert_eq!(vec.get(0), Some(1));
        assert_eq!(vec.get(1), None);

        // Test get - nulls.
        assert_eq!(vec.get(0), Some(1));
        assert_eq!(vec.get(1), None);
        assert_eq!(vec.get(2), Some(3));

        assert_eq!(vec.elements()[0], 1);
        assert_eq!(vec.elements()[2], 3);

        // Also test PVector methods.
        let frozen = vec.freeze();
        assert_eq!(frozen.get(0), Some(&1));
        assert_eq!(frozen.get(1), None);
        assert_eq!(frozen.get(2), Some(&3));
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_get_panic() {
        let vec = PVectorMut::from_iter([Some(1), Some(2)]);
        let _ = vec.get(10);
    }

    #[test]
    fn test_push_variants() {
        let mut vec = PVectorMut::<i32>::with_capacity(10);
        vec.push(1);
        vec.push_opt(None);
        vec.push_opt(Some(3));

        assert_eq!(vec.len(), 3);
        assert_eq!(vec.get(0), Some(1));
        assert_eq!(vec.get(1), None);
        assert_eq!(vec.get(2), Some(3));

        // Test push_unchecked with pre-reserved capacity.
        vec.reserve(1);
        unsafe {
            vec.push_unchecked(4);
        }
        assert_eq!(vec.get(3), Some(4));
    }

    #[test]
    fn test_resize_operations() {
        let mut vec = PVectorMut::from_iter([1i32, 2, 3]);

        // Grow with valid values.
        vec.resize(5, Some(99));
        assert_eq!(vec.len(), 5);
        assert_eq!(vec.get(3), Some(99));
        assert_eq!(vec.get(4), Some(99));

        // Grow with nulls.
        vec.resize(7, None);
        assert_eq!(vec.get(5), None);
        assert_eq!(vec.get(6), None);

        // Shrink.
        vec.resize(2, Some(0));
        assert_eq!(vec.len(), 2);
        assert_eq!(vec.get(0), Some(1));
        assert_eq!(vec.get(1), Some(2));
    }

    #[test]
    fn test_clear_truncate() {
        let mut vec = PVectorMut::from_iter([Some(1), None, Some(3), None, Some(5)]);
        let cap = vec.capacity();

        vec.truncate(3);
        assert_eq!(vec.len(), 3);
        assert!(vec.capacity() >= cap); // Capacity preserved.

        vec.truncate(10); // Truncate beyond length - no-op.
        assert_eq!(vec.len(), 3);

        vec.clear();
        assert_eq!(vec.len(), 0);
        assert!(vec.capacity() >= cap); // Capacity still preserved.
    }

    #[test]
    fn test_slice_access() {
        let mut vec = PVectorMut::from_iter([Some(1i32), None, Some(3)]);
        let slice = vec.as_ref();
        assert_eq!(slice[0], 1);
        assert_eq!(slice[2], 3);
        // slice[1] is undefined for null but safe to access.

        let mut_slice = vec.as_mut();
        mut_slice[0] = 10;
        assert_eq!(vec.get(0), Some(10));

        let frozen = vec.freeze();
        assert_eq!(frozen.as_ref()[0], 10);
    }

    #[test]
    fn test_from_iter_variants() {
        // FromIterator<T> - all non-null.
        let vec1 = PVectorMut::from_iter([1i32, 2, 3]);
        assert_eq!(vec1.len(), 3);
        assert!(vec1.freeze().validity().all_true());

        // FromIterator<Option<T>> - mixed null/non-null.
        let vec2 = PVectorMut::from_iter([Some(1i32), None, Some(3)]);
        assert_eq!(vec2.len(), 3);
        assert_eq!(vec2.freeze().validity().true_count(), 2);

        // Empty iterators.
        let empty1 = PVectorMut::from_iter::<[i32; 0]>([]);
        let empty2 = PVectorMut::<i32>::from_iter(std::iter::empty::<Option<i32>>());
        assert_eq!(empty1.len(), 0);
        assert_eq!(empty2.len(), 0);
    }

    #[test]
    fn test_extend_operations() {
        let mut vec = PVectorMut::from_iter([1i32, 2]);

        // Extend<T> - all non-null.
        vec.extend([3, 4]);
        assert_eq!(vec.len(), 4);
        assert_eq!(vec.get(3), Some(4));

        // Extend<Option<T>> - mixed null/non-null.
        vec.extend([Some(5), None, Some(7)]);
        assert_eq!(vec.len(), 7);
        assert_eq!(vec.get(5), None);
        assert_eq!(vec.get(6), Some(7));

        // Extend with iterator that has size hint.
        let iter = 8..10;
        vec.extend(iter);
        assert_eq!(vec.get(8), Some(9));
    }

    #[test]
    fn test_empty_vector_edge_cases() {
        let empty = PVectorMut::<i32>::with_capacity(0);
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.as_ref().len(), 0);

        let mut mutable_empty = PVectorMut::<i32>::with_capacity(0);
        mutable_empty.clear(); // No-op on empty.
        mutable_empty.truncate(0); // No-op.
        mutable_empty.resize(0, None); // No-op.
        assert_eq!(mutable_empty.len(), 0);
    }

    #[test]
    fn test_complex_workflow() {
        // Integration test combining multiple operations.
        let mut vec = PVectorMut::<i32>::with_capacity(2);
        vec.extend([1, 2]); // Extend<T>.
        vec.push_opt(None);
        vec.resize(5, Some(99));
        vec.truncate(4);
        vec.extend([Some(10), None]); // Extend<Option<T>>.

        assert_eq!(vec.len(), 6);
        let frozen = vec.freeze();
        assert_eq!(frozen.validity().true_count(), 4);
        assert_eq!(frozen.get(0), Some(&1));
        assert_eq!(frozen.get(2), None);
        assert_eq!(frozen.get(3), Some(&99));
        assert_eq!(frozen.get(5), None);
    }

    #[test]
    fn test_into_iter_roundtrip() {
        // Test that from_iter followed by into_iter preserves the data.
        let original_data = vec![
            Some(1i32),
            None,
            Some(3),
            Some(4),
            None,
            Some(6),
            None,
            Some(8),
        ];

        // Create vector from iterator.
        let vec = PVectorMut::<i32>::from_iter(original_data.clone());

        // Convert back to iterator and collect.
        let roundtrip: Vec<_> = vec.into_iter().collect();

        // Should be identical.
        assert_eq!(roundtrip, original_data);

        // Also test with all valid values.
        let all_valid = vec![1, 2, 3, 4, 5];
        let vec = PVectorMut::<i32>::from_iter(all_valid.clone());
        let roundtrip: Vec<_> = vec.into_iter().collect();
        let expected: Vec<_> = all_valid.into_iter().map(Some).collect();
        assert_eq!(roundtrip, expected);

        // Test with empty.
        let empty: Vec<Option<i32>> = vec![];
        let vec = PVectorMut::<i32>::from_iter(empty.clone());
        let roundtrip: Vec<_> = vec.into_iter().collect();
        assert_eq!(roundtrip, empty);
    }
}
