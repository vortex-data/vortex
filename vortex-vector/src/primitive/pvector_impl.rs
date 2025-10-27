// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Helper methods for [`PVectorMut<T>`] that mimic the behavior of [`std::vec::Vec`].

use vortex_dtype::NativePType;
use vortex_error::VortexExpect;

use crate::{PVectorMut, VectorMutOps};

// TODO(connor): Implement proper `IntoIterator` trait instead of relying on `to_vec`.

/// Conversion methods from [`PVectorMut`] to [`std::vec::Vec`].
impl<T: NativePType> PVectorMut<T> {
    /// Converts the vector to a `Vec<Option<T>>`.
    ///
    /// This method borrows the [`PVectorMut`] and creates a new `Vec` with `Some(value)` for
    /// non-null elements and `None` for null elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use vortex_vector::PVectorMut;
    ///
    /// let vec = PVectorMut::from_iter([Some(1i32), None, Some(3)]);
    /// let values = vec.to_vec();
    /// assert_eq!(values, vec![Some(1), None, Some(3)]);
    /// ```
    pub fn to_vec(&self) -> Vec<Option<T>> {
        (0..self.len())
            .map(|i| self.validity.value(i).then(|| self.elements[i]))
            .collect()
    }

    /// Attempts to convert the vector to a `Vec<T>` containing only non-null values.
    ///
    /// Returns `None` if the [`PVectorMut`] contains any null values. Otherwise, returns
    /// `Some(Vec<T>)` with all the non-null values.
    ///
    /// # Examples
    ///
    /// ```
    /// use vortex_vector::PVectorMut;
    ///
    /// // All non-null values.
    /// let vec = PVectorMut::from_iter([1i32, 2, 3]);
    /// let values = vec.to_nonnull_vec();
    /// assert_eq!(values, Some(vec![1, 2, 3]));
    ///
    /// // Contains null values.
    /// let vec = PVectorMut::from_iter([Some(1i32), None, Some(3)]);
    /// let values = vec.to_nonnull_vec();
    /// assert_eq!(values, None);
    /// ```
    pub fn to_nonnull_vec(&self) -> Option<Vec<T>> {
        let validity_frozen = self.validity.clone().freeze();
        validity_frozen
            .all_true()
            .then(|| self.elements.as_slice().to_vec())
    }
}

/// Point operations for [`PVectorMut`].
impl<T: NativePType> PVectorMut<T> {
    /// Returns the first element of the vector, or `None` if it is empty.
    ///
    /// If the first element is null, returns `Some(None)`. Otherwise, returns `Some(Some(x))`,
    /// where `x: T`.
    pub fn first(&self) -> Option<Option<T>> {
        self.get_checked(0)
    }

    /// Returns the last element of the vector, or `None` if it is empty.
    ///
    /// If the last element is null, returns `Some(None)`. Otherwise, returns `Some(Some(x))`,
    /// where `x: T`.
    pub fn last(&self) -> Option<Option<T>> {
        if self.len() == 0 {
            None
        } else {
            self.get_checked(self.len() - 1)
        }
    }

    /// Gets a nullable element at the given index, with bounds checking.
    ///
    /// If the index is out of bounds, returns `None`. If the element at the given index is null,
    /// returns `Some(None)`. Otherwise, returns `Some(Some(x))`, where `x: T`.
    pub fn get_checked(&self, index: usize) -> Option<Option<T>> {
        (index < self.len()).then(|| {
            self.validity.value(index).then(|| {
                self.elements
                    .get(index)
                    .copied()
                    .vortex_expect("length of elements was somehow incorrect")
            })
        })
    }

    /// Gets a nullable element at the given index, **WITHOUT** bounds checking.
    ///
    /// If the element at the given index is null, returns `None`. Otherwise, returns `Some(x)`,
    /// where `x: T`.
    ///
    /// Note that this `get` method is different from the standard library [`slice::get`], which
    /// returns `None` if the index is out of bounds. This method will panic if the index is out of
    /// bounds, and return `None` if the elements is null.
    ///
    /// If you want bounds checking, use [`get_checked()`](Self::get_checked) instead.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn get(&self, index: usize) -> Option<T> {
        assert!(
            index < self.len(),
            "index out of bounds: the length is {} but the index is {index}",
            self.len()
        );

        self.validity.value(index).then(|| {
            self.elements
                .get(index)
                .copied()
                .vortex_expect("length of elements was somehow incorrect")
        })
    }

    /// Gets a nullable element at the given index, without checking bounds and without checking
    /// nullability.
    ///
    /// The caller should ensure that the element at the given index is not null (though doing so
    /// will not cause undefined behavior).
    ///
    /// # Safety
    ///
    /// The caller must ensure that the index is within bounds.
    pub unsafe fn get_unchecked(&self, index: usize) -> T {
        debug_assert!(
            index < self.len(),
            "index out of bounds: the length is {} but the index is {index}",
            self.len()
        );

        // SAFETY: The caller ensures that the index is in bounds.
        unsafe { *self.elements.get_unchecked(index) }
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

/// Batch operations for [`PVectorMut`].
impl<T: NativePType> PVectorMut<T> {
    /// Returns an immutable slice over the internal mutable buffer with elements of type `T`.
    ///
    /// Note that this slice may contain garbage data where the [`validity()`] mask from the frozen
    /// [`PVector`](crate::PVector) type states that an element is invalid.
    ///
    /// The caller should check the frozen [`validity()`] before performing any operations.
    ///
    /// [`validity()`]: crate::VectorOps::validity
    pub fn as_slice(&self) -> &[T] {
        self.elements.as_slice()
    }

    /// Returns a mutable slice over the internal mutable buffer with elements of type `T`.
    ///
    /// Note that this slice may contain garbage data where the [`validity()`] mask from the frozen
    /// [`PVector`](crate::PVector) type states that an element is invalid.
    ///
    /// The caller should check the frozen [`validity()`] before performing any operations.
    ///
    /// [`validity()`]: crate::VectorOps::validity
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.elements.as_mut_slice()
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

impl<T: NativePType> Extend<Option<T>> for PVectorMut<T> {
    /// Extends the vector from an iterator of optional values.
    ///
    /// `None` values will be marked as null in the validity mask.
    ///
    /// # Examples
    ///
    /// ```
    /// use vortex_vector::{PVectorMut, VectorMutOps, VectorOps};
    ///
    /// let mut vec = PVectorMut::from_iter([Some(1i32), None]);
    /// vec.extend([Some(3), None, Some(5)]);
    /// assert_eq!(vec.len(), 5);
    ///
    /// let frozen = vec.freeze();
    /// assert_eq!(frozen.validity().true_count(), 3); // Only 3 non-null values.
    /// ```
    fn extend<I: IntoIterator<Item = Option<T>>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        // Since we do not know the length of the iterator, we can only guess how much memory we
        // need to reserve. Note that these hints may be inaccurate.
        let (lower_bound, _) = iter.size_hint();

        // We choose not to use the optional upper bound size hint to match the standard library.

        self.reserve(lower_bound);

        // We have to update validity per-element since it depends on Option variant.
        for opt_val in iter {
            match opt_val {
                Some(val) => {
                    self.elements.push(val);
                    self.validity.append_n(true, 1);
                }
                None => {
                    self.elements.push(T::default());
                    self.validity.append_n(false, 1);
                }
            }
        }
    }
}

impl<T: NativePType> FromIterator<Option<T>> for PVectorMut<T> {
    /// Creates a new [`PVectorMut<T>`] from an iterator of `Option<T>` values.
    ///
    /// `None` values will be marked as invalid in the validity mask.
    ///
    /// Internally, this uses the [`Extend<Option<T>>`] trait implementation.
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Option<T>>,
    {
        let mut vec = Self::with_capacity(0);
        vec.extend(iter);
        vec
    }
}

impl<T: NativePType> Extend<T> for PVectorMut<T> {
    /// Extends the vector from an iterator of values.
    ///
    /// All values from the iterator will be marked as non-null in the validity mask.
    ///
    /// Internally, this uses the [`Extend<T>`] trait implementation.
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let start_len = self.len();

        // Allow the `BufferMut` implementation to handle extending efficiently.
        self.elements.extend(iter);
        self.validity.append_n(true, self.len() - start_len);
    }
}

impl<T: NativePType> FromIterator<T> for PVectorMut<T> {
    /// Creates a new [`PVectorMut<T>`] from an iterator of `T` values.
    ///
    /// All values will be treated as non-null.
    ///
    /// # Examples
    ///
    /// ```
    /// use vortex_vector::{PVectorMut, VectorMutOps};
    ///
    /// let mut vec = PVectorMut::from_iter([1i32, 2, 3, 4]);
    /// assert_eq!(vec.len(), 4);
    /// ```
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut vec = Self::with_capacity(0);
        vec.extend(iter);
        vec
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VectorOps;

    #[test]
    fn test_first_last() {
        let empty = PVectorMut::<i32>::with_capacity(0);
        assert_eq!(empty.first(), None);
        assert_eq!(empty.last(), None);

        let single = PVectorMut::from_iter([Some(42)]);
        assert_eq!(single.first(), Some(Some(42)));
        assert_eq!(single.last(), Some(Some(42)));

        let mixed = PVectorMut::from_iter([Some(1), None, Some(3), None, Some(5)]);
        assert_eq!(mixed.first(), Some(Some(1)));
        assert_eq!(mixed.last(), Some(Some(5)));

        let null_first = PVectorMut::from_iter([None, Some(2i32)]);
        assert_eq!(null_first.first(), Some(None));
        assert_eq!(null_first.last(), Some(Some(2)));
    }

    #[test]
    fn test_get_methods() {
        let vec = PVectorMut::from_iter([Some(1), None, Some(3), None, Some(5)]);

        // Test get_checked - bounds and nulls.
        assert_eq!(vec.get_checked(0), Some(Some(1)));
        assert_eq!(vec.get_checked(1), Some(None));
        assert_eq!(vec.get_checked(5), None); // Out of bounds.

        // Test get - nulls.
        assert_eq!(vec.get(0), Some(1));
        assert_eq!(vec.get(1), None);
        assert_eq!(vec.get(2), Some(3));

        // Test get_unchecked.
        unsafe {
            assert_eq!(vec.get_unchecked(0), 1);
            assert_eq!(vec.get_unchecked(2), 3);
            // Note: get_unchecked(1) would return default value but is safe.
        }

        // Also test PVector methods.
        let frozen = vec.freeze();
        assert_eq!(frozen.get_checked(0), Some(Some(1)));
        assert_eq!(frozen.get(1), None);
        unsafe {
            assert_eq!(frozen.get_unchecked(2), 3);
        }
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
        assert_eq!(vec.last(), Some(Some(3)));
        assert!(vec.capacity() >= cap); // Capacity preserved.

        vec.truncate(10); // Truncate beyond length - no-op.
        assert_eq!(vec.len(), 3);

        vec.clear();
        assert_eq!(vec.len(), 0);
        assert!(vec.capacity() >= cap); // Capacity still preserved.
        assert_eq!(vec.first(), None);
    }

    #[test]
    fn test_slice_access() {
        let mut vec = PVectorMut::from_iter([Some(1i32), None, Some(3)]);
        let slice = vec.as_slice();
        assert_eq!(slice[0], 1);
        assert_eq!(slice[2], 3);
        // slice[1] is undefined for null but safe to access.

        let mut_slice = vec.as_mut_slice();
        mut_slice[0] = 10;
        assert_eq!(vec.get(0), Some(10));

        let frozen = vec.freeze();
        assert_eq!(frozen.as_slice()[0], 10);
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
        assert_eq!(empty.first(), None);
        assert_eq!(empty.last(), None);
        assert_eq!(empty.get_checked(0), None);
        assert_eq!(empty.as_slice().len(), 0);

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
        assert_eq!(frozen.get(0), Some(1));
        assert_eq!(frozen.get(2), None);
        assert_eq!(frozen.get(3), Some(99));
        assert_eq!(frozen.get(5), None);
    }

    #[test]
    fn test_to_vec_variants() {
        // Test to_vec with mixed null/non-null values.
        let mixed = PVectorMut::from_iter([Some(1i32), None, Some(3), None, Some(5)]);
        assert_eq!(mixed.to_vec(), vec![Some(1), None, Some(3), None, Some(5)]);
        assert_eq!(mixed.len(), 5); // Vector still usable after to_vec.

        // Test to_vec with all non-null.
        let all_valid = PVectorMut::from_iter([1i32, 2, 3]);
        assert_eq!(all_valid.to_vec(), vec![Some(1), Some(2), Some(3)]);

        // Test to_vec with all null.
        let mut all_null = PVectorMut::<i32>::with_capacity(2);
        all_null.push_opt(None);
        all_null.push_opt(None);
        assert_eq!(all_null.to_vec(), vec![None, None]);

        // Test to_vec with empty vector.
        let empty = PVectorMut::<i32>::with_capacity(0);
        assert_eq!(empty.to_vec(), Vec::<Option<i32>>::new());
    }

    #[test]
    fn test_to_nonnull_vec_variants() {
        // Test with all non-null values - should return Some(vec).
        let all_valid = PVectorMut::from_iter([1i32, 2, 3, 4, 5]);
        assert_eq!(all_valid.to_nonnull_vec(), Some(vec![1, 2, 3, 4, 5]));
        assert_eq!(all_valid.len(), 5); // Vector still usable.

        // Test with mixed values - should return None.
        let mixed = PVectorMut::from_iter([Some(1i32), None, Some(3)]);
        assert_eq!(mixed.to_nonnull_vec(), None);

        // Test with all nulls - should return None.
        let mut all_null = PVectorMut::<i32>::with_capacity(3);
        all_null.extend([None, None, None]);
        assert_eq!(all_null.to_nonnull_vec(), None);

        // Test empty vector - should return Some(empty vec).
        let empty = PVectorMut::<i32>::with_capacity(0);
        assert_eq!(empty.to_nonnull_vec(), Some(Vec::<i32>::new()));
    }
}
