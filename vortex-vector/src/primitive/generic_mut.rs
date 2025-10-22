// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`PVectorMut<T>`].

use vortex_buffer::BufferMut;
use vortex_dtype::NativePType;
use vortex_mask::MaskMut;

use crate::{PVector, VectorMutOps, VectorOps};

/// A mutable vector of generic primitive values.
///
/// `T` is expected to be bound by [`NativePType`], which templates an internal [`BufferMut<T>`]
/// that stores the elements of the vector.
///
/// `PVectorMut<T>` is the primary way to construct primitive vectors. It provides efficient methods
/// for building vectors incrementally before converting them to an immutable [`PVector<T>`] using
/// the [`freeze`](crate::VectorMutOps::freeze) method.
///
/// # Examples
///
/// ## Creating and building a vector
///
/// ```
/// use vortex_vector::{PVectorMut, VectorMutOps};
///
/// // Create with initial capacity for i32 values.
/// let mut vec = PVectorMut::<i32>::with_capacity(10);
/// assert_eq!(vec.len(), 0);
/// assert!(vec.capacity() >= 10);
///
/// // Create from an iterator of optional values.
/// let mut vec = PVectorMut::<i32>::from_iter([Some(1), None, Some(3)]);
/// assert_eq!(vec.len(), 3);
///
/// // Works with different primitive types.
/// let mut f64_vec = PVectorMut::<f64>::from_iter([1.5, 2.5, 3.5].map(Some));
/// assert_eq!(f64_vec.len(), 3);
/// ```
///
/// ## Extending and appending
///
/// ```
/// use vortex_vector::{PVectorMut, VectorMutOps};
///
/// let mut vec1 = PVectorMut::<i32>::from_iter([1, 2].map(Some));
/// let vec2 = PVectorMut::<i32>::from_iter([3, 4].map(Some)).freeze();
///
/// // Extend from another vector.
/// vec1.extend_from_vector(&vec2);
/// assert_eq!(vec1.len(), 4);
///
/// // Append null values.
/// vec1.append_nulls(2);
/// assert_eq!(vec1.len(), 6);
/// ```
///
/// ## Splitting and unsplitting
///
/// ```
/// use vortex_vector::{PVectorMut, VectorMutOps};
///
/// let mut vec = PVectorMut::<i64>::from_iter([10, 20, 30, 40, 50].map(Some));
///
/// // Split the vector at index 3.
/// let mut second_half = vec.split_off(3);
/// assert_eq!(vec.len(), 3);
/// assert_eq!(second_half.len(), 2);
///
/// // Rejoin the vectors.
/// vec.unsplit(second_half);
/// assert_eq!(vec.len(), 5);
/// ```
///
/// ## Working with nulls
///
/// ```
/// use vortex_vector::{PVectorMut, VectorMutOps};
///
/// // Create a vector with some null values.
/// let mut vec = PVectorMut::<u32>::from_iter([Some(100), None, Some(200), None]);
/// assert_eq!(vec.len(), 4);
///
/// // Add more nulls.
/// vec.append_nulls(3);
/// assert_eq!(vec.len(), 7);
/// ```
///
/// ## Converting to immutable
///
/// ```
/// use vortex_vector::{PVectorMut, VectorMutOps, VectorOps};
///
/// let mut vec = PVectorMut::<f32>::from_iter([1.0, 2.0, 3.0].map(Some));
///
/// // Freeze into an immutable vector.
/// let immutable = vec.freeze();
/// assert_eq!(immutable.len(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct PVectorMut<T> {
    pub(super) elements: BufferMut<T>,
    pub(super) validity: MaskMut,
}

impl<T> PVectorMut<T> {
    /// Create a new mutable primitive vector with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            elements: BufferMut::with_capacity(capacity),
            validity: MaskMut::with_capacity(capacity),
        }
    }
}

impl<T: NativePType> FromIterator<Option<T>> for PVectorMut<T> {
    /// Creates a new [`PVectorMut<T>`] from an iterator of `Option<T>` values.
    ///
    /// `None` values will be marked as invalid in the validity mask.
    ///
    /// # Examples
    ///
    /// ```
    /// use vortex_vector::{PVectorMut, VectorMutOps};
    ///
    /// let mut vec = PVectorMut::<i32>::from_iter([Some(1), None, Some(3)]);
    /// assert_eq!(vec.len(), 3);
    /// ```
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Option<T>>,
    {
        let iter = iter.into_iter();
        let (lower_bound, _) = iter.size_hint();

        let mut elements = Vec::with_capacity(lower_bound);
        let mut validity = MaskMut::with_capacity(lower_bound);

        for opt_val in iter {
            match opt_val {
                Some(val) => {
                    elements.push(val);
                    validity.append_n(true, 1);
                }
                None => {
                    elements.push(T::default()); // Use default for invalid entries.
                    validity.append_n(false, 1);
                }
            }
        }

        PVectorMut {
            elements: BufferMut::from_iter(elements),
            validity,
        }
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
    /// let mut vec = PVectorMut::<i32>::from_iter([1, 2, 3, 4]);
    /// assert_eq!(vec.len(), 4);
    /// ```
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let buffer = BufferMut::from_iter(iter);
        let validity = MaskMut::new_true(buffer.len());

        PVectorMut {
            elements: buffer,
            validity,
        }
    }
}

impl<T: NativePType> VectorMutOps for PVectorMut<T> {
    type Immutable = PVector<T>;

    fn len(&self) -> usize {
        self.elements.len()
    }

    fn capacity(&self) -> usize {
        self.elements.capacity()
    }

    fn reserve(&mut self, additional: usize) {
        self.elements.reserve(additional);
        self.validity.reserve(additional);
    }

    /// Extends the vector by appending elements from another vector.
    fn extend_from_vector(&mut self, other: &PVector<T>) {
        self.elements.extend_from_slice(other.elements.as_slice());
        self.validity.append_mask(other.validity());
    }

    fn append_nulls(&mut self, n: usize) {
        self.elements.push_n(T::zero(), n);
        self.validity.append_n(false, n);
    }

    /// Freeze the vector into an immutable one.
    fn freeze(self) -> PVector<T> {
        PVector {
            elements: self.elements.freeze(),
            validity: self.validity.freeze(),
        }
    }

    fn split_off(&mut self, at: usize) -> Self {
        PVectorMut {
            elements: self.elements.split_off(at),
            validity: self.validity.split_off(at),
        }
    }

    fn unsplit(&mut self, other: Self) {
        self.elements.unsplit(other.elements);
        self.validity.unsplit(other.validity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_iter_with_options() {
        // Test FromIterator<Option<T>> with different types.
        let vec_i32 = PVectorMut::<i32>::from_iter(vec![Some(1), None, Some(3), None, Some(5)]);
        assert_eq!(vec_i32.len(), 5);
        let frozen = vec_i32.freeze();
        assert_eq!(frozen.validity().true_count(), 3);

        // Test empty iterator.
        let vec_empty = PVectorMut::<f64>::from_iter(std::iter::empty::<Option<f64>>());
        assert_eq!(vec_empty.len(), 0);

        // Test that None values use T::default().
        let vec_nulls = PVectorMut::<i32>::from_iter([None, None, None]);
        assert_eq!(vec_nulls.elements[0], 0); // Default value for i32.
        let frozen = vec_nulls.freeze();
        assert_eq!(frozen.validity().true_count(), 0);
    }

    #[test]
    fn test_from_iter_non_null() {
        // Test FromIterator<T> for different primitive types.
        let vec_f64 = PVectorMut::<f64>::from_iter([1.5, 2.5, 3.5, 4.5, 5.5]);
        assert_eq!(vec_f64.len(), 5);
        let frozen = vec_f64.freeze();
        assert_eq!(frozen.validity().true_count(), 5); // All valid.

        let vec_u16 = PVectorMut::<u16>::from_iter([1u16, 2, 3, 4, 5]);
        assert_eq!(vec_u16.len(), 5);
        let frozen = vec_u16.freeze();
        assert_eq!(frozen.validity().true_count(), 5);
    }

    #[test]
    fn test_operations_preserve_validity() {
        // Test split/unsplit/extend with different primitive types.
        let mut vec = PVectorMut::<i64>::from_iter([Some(100), None, Some(300), None, Some(500)]);

        let second_half = vec.split_off(2);
        assert_eq!(vec.len(), 2);
        assert_eq!(second_half.len(), 3);

        let first_frozen = vec.freeze();
        let second_frozen = second_half.freeze();
        assert_eq!(first_frozen.validity().true_count(), 1);
        assert_eq!(second_frozen.validity().true_count(), 2);

        // Test unsplit.
        let mut vec1 = PVectorMut::<u32>::from_iter([Some(1000), None]);
        let vec2 = PVectorMut::<u32>::from_iter([None, Some(2000)]);
        vec1.unsplit(vec2);
        assert_eq!(vec1.len(), 4);
        let frozen = vec1.freeze();
        assert_eq!(frozen.validity().true_count(), 2);
    }
}
