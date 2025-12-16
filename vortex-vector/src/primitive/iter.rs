// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Iterator implementations for [`PVectorMut`].

use vortex_dtype::NativePType;

use crate::VectorMutOps;
use crate::primitive::PVector;
use crate::primitive::PVectorMut;
use crate::primitive::PrimitiveVector;

impl<T: NativePType> Extend<Option<T>> for PVectorMut<T> {
    /// Extends the vector from an iterator of optional values.
    ///
    /// `None` values will be marked as null in the validity mask.
    ///
    /// # Examples
    ///
    /// ```
    /// use vortex_vector::primitive::PVectorMut;
    /// use vortex_vector::{VectorMutOps, VectorOps};
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
        let iter = iter.into_iter();

        let mut vec = Self::with_capacity(iter.size_hint().0);
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
    /// use vortex_vector::primitive::PVectorMut;
    /// use vortex_vector::VectorMutOps;
    ///
    /// let mut vec = PVectorMut::from_iter([1i32, 2, 3, 4]);
    /// assert_eq!(vec.len(), 4);
    /// ```
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let iter = iter.into_iter();

        let mut vec = Self::with_capacity(iter.size_hint().0);
        vec.extend(iter);

        vec
    }
}

impl<T: NativePType> FromIterator<T> for PVector<T> {
    /// Creates a new [`PVector<T>`] from an iterator of `T` values.
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let vec_mut: PVectorMut<T> = iter.into_iter().collect();

        vec_mut.freeze()
    }
}

impl<T: NativePType> FromIterator<T> for PrimitiveVector {
    /// Creates a new [`PVector<T>`] from an iterator of `T` values.
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let vec_mut: PVectorMut<T> = iter.into_iter().collect();

        PrimitiveVector::from(vec_mut.freeze())
    }
}

/// Iterator over a [`PVectorMut<T>`] that yields [`Option<T>`] values.
///
/// This iterator is created by calling [`IntoIterator::into_iter`] on a [`PVectorMut<T>`].
///
/// It consumes the mutable vector and iterates over the elements, yielding `None` for null values
/// and `Some(value)` for valid values.
#[derive(Debug)]
pub struct PVectorMutIterator<T: NativePType> {
    /// The vector being iterated over.
    vector: PVectorMut<T>,
    /// The current index into the vector.
    index: usize,
}

impl<T: NativePType> Iterator for PVectorMutIterator<T> {
    type Item = Option<T>;

    fn next(&mut self) -> Option<Self::Item> {
        (self.index < self.vector.len()).then(|| {
            let value = self
                .vector
                .validity
                .value(self.index)
                .then(|| self.vector.elements[self.index]);
            self.index += 1;
            value
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.vector.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<T: NativePType> IntoIterator for PVectorMut<T> {
    type Item = Option<T>;
    type IntoIter = PVectorMutIterator<T>;

    /// Converts the mutable vector into an iterator over [`Option<T>`] values.
    ///
    /// This method consumes the [`PVectorMut<T>`] and returns an iterator that yields `None` for
    /// null values and `Some(value)` for valid values.
    ///
    /// # Examples
    ///
    /// ```
    /// use vortex_vector::primitive::PVectorMut;
    ///
    /// let vec = PVectorMut::<i32>::from_iter([Some(1), None, Some(3), Some(4)]);
    /// let collected: Vec<_> = vec.into_iter().collect();
    /// assert_eq!(collected, vec![Some(1), None, Some(3), Some(4)]);
    /// ```
    fn into_iter(self) -> Self::IntoIter {
        PVectorMutIterator {
            vector: self,
            index: 0,
        }
    }
}
