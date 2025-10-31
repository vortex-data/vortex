// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Iterator implementations for [`BoolVectorMut`].

use vortex_buffer::BitBufferMut;
use vortex_mask::MaskMut;

use crate::{BoolVectorMut, VectorMutOps};

impl FromIterator<Option<bool>> for BoolVectorMut {
    /// Creates a new [`BoolVectorMut`] from an iterator of `Option<bool>` values.
    ///
    /// `None` values will be marked as invalid in the validity mask.
    ///
    /// # Examples
    ///
    /// ```
    /// use vortex_vector::{BoolVectorMut, VectorMutOps};
    ///
    /// let mut vec = BoolVectorMut::from_iter([Some(true), None, Some(false)]);
    /// assert_eq!(vec.len(), 3);
    /// ```
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Option<bool>>,
    {
        let iter = iter.into_iter();
        // Since we do not know the length of the iterator, we can only guess how much memory we
        // need to reserve. Note that these hints may be inaccurate.
        let (lower_bound, _) = iter.size_hint();

        // We choose not to use the optional upper bound size hint to match the standard library.

        let mut bits = BitBufferMut::with_capacity(lower_bound);
        let mut validity = MaskMut::with_capacity(lower_bound);

        for opt_val in iter {
            match opt_val {
                Some(val) => {
                    bits.append(val);
                    validity.append_n(true, 1);
                }
                None => {
                    bits.append(false); // Value doesn't matter for invalid entries.
                    validity.append_n(false, 1);
                }
            }
        }

        BoolVectorMut { bits, validity }
    }
}

impl FromIterator<bool> for BoolVectorMut {
    /// Creates a new [`BoolVectorMut`] from an iterator of `bool` values.
    ///
    /// All values will be treated as non-null.
    ///
    /// # Examples
    ///
    /// ```
    /// use vortex_vector::{BoolVectorMut, VectorMutOps};
    ///
    /// let mut vec = BoolVectorMut::from_iter([true, false, false, true]);
    /// assert_eq!(vec.len(), 4);
    /// ```
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = bool>,
    {
        let buffer = BitBufferMut::from_iter(iter);
        let validity = MaskMut::new_true(buffer.len());

        BoolVectorMut {
            bits: buffer,
            validity,
        }
    }
}

/// Iterator over a [`BoolVectorMut`] that yields [`Option<bool>`] values.
///
/// This iterator is created by calling [`IntoIterator::into_iter`] on a [`BoolVectorMut`].
///
/// It consumes the mutable vector and iterates over the elements, yielding `None` for null values
/// and `Some(value)` for valid values.
#[derive(Debug)]
pub struct BoolVectorMutIterator {
    /// The vector being iterated over.
    vector: BoolVectorMut,
    /// The current index into the vector.
    index: usize,
}

impl Iterator for BoolVectorMutIterator {
    type Item = Option<bool>;

    fn next(&mut self) -> Option<Self::Item> {
        (self.index < self.vector.len()).then(|| {
            let value = self
                .vector
                .validity
                .value(self.index)
                .then(|| self.vector.bits.value(self.index));
            self.index += 1;
            value
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.vector.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl IntoIterator for BoolVectorMut {
    type Item = Option<bool>;
    type IntoIter = BoolVectorMutIterator;

    /// Converts the mutable vector into an iterator over `Option<bool>` values.
    ///
    /// This method consumes the `BoolVectorMut` and returns an iterator that yields `None` for
    /// null values and `Some(value)` for valid values.
    ///
    /// # Examples
    ///
    /// ```
    /// use vortex_vector::BoolVectorMut;
    ///
    /// let vec = BoolVectorMut::from_iter([Some(true), None, Some(false), Some(true)]);
    /// let collected: Vec<_> = vec.into_iter().collect();
    /// assert_eq!(collected, vec![Some(true), None, Some(false), Some(true)]);
    /// ```
    fn into_iter(self) -> Self::IntoIter {
        BoolVectorMutIterator {
            vector: self,
            index: 0,
        }
    }
}
