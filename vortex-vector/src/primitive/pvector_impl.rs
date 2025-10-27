// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Helper methods for [`PVectorMut<T>`] that mimic the behavior of [`std::vec::Vec`].

use vortex_dtype::NativePType;
use vortex_error::VortexExpect;

use crate::{PVectorMut, VectorMutOps};

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

    /// Gets a nullable element at the given index.
    ///
    /// If the element at the given index is null, returns `None`. Otherwise, returns `Some(x)`,
    /// where `x: T`.
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
    /// [`PVector`] type states that an element is invalid.
    ///
    /// The caller should check the frozen [`validity()`] before performing any operations.
    ///
    /// [`validity()`]: PVector::validity
    pub fn as_slice(&self) -> &[T] {
        self.elements.as_slice()
    }

    /// Returns a mutable slice over the internal mutable buffer with elements of type `T`.
    ///
    /// Note that this slice may contain garbage data where the [`validity()`] mask from the frozen
    /// [`PVector`] type states that an element is invalid.
    ///
    /// The caller should check the frozen [`validity()`] before performing any operations.
    ///
    /// [`validity()`]:  PVector::validity
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
