// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Iterator implementations for [`BoolVector`].

use vortex_buffer::BitBufferMut;
use vortex_mask::MaskMut;

use crate::VectorOps;
use crate::bool::BoolVector;

impl FromIterator<Option<bool>> for BoolVector {
    /// Creates a new [`BoolVector`] from an iterator of `Option<bool>` values.
    ///
    /// `None` values will be marked as invalid in the validity mask.
    ///
    /// # Examples
    ///
    /// ```
    /// use vortex_vector::bool::BoolVector;
    /// use vortex_vector::VectorOps;
    ///
    /// let mut vec = BoolVector::from_iter([Some(true), None, Some(false)]);
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

        BoolVector { bits, validity }
    }
}

impl FromIterator<bool> for BoolVector {
    /// Creates a new [`BoolVector`] from an iterator of `bool` values.
    ///
    /// All values will be treated as non-null.
    ///
    /// # Examples
    ///
    /// ```
    /// use vortex_vector::bool::BoolVector;
    /// use vortex_vector::VectorOps;
    ///
    /// let mut vec = BoolVector::from_iter([true, false, false, true]);
    /// assert_eq!(vec.len(), 4);
    /// ```
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = bool>,
    {
        let buffer = BitBufferMut::from_iter(iter);
        let validity = MaskMut::new_true(buffer.len());

        BoolVector {
            bits: buffer,
            validity,
        }
    }
}

/// Iterator over a [`BoolVector`] that yields [`Option<bool>`] values.
///
/// This iterator is created by calling [`IntoIterator::into_iter`] on a [`BoolVector`].
///
/// It consumes the mutable vector and iterates over the elements, yielding `None` for null values
/// and `Some(value)` for valid values.
#[derive(Debug)]
pub struct BoolVectorIterator {
    /// The vector being iterated over.
    vector: BoolVector,
    /// The current index into the vector.
    index: usize,
}

impl Iterator for BoolVectorIterator {
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

impl IntoIterator for BoolVector {
    type Item = Option<bool>;
    type IntoIter = BoolVectorIterator;

    /// Converts the mutable vector into an iterator over `Option<bool>` values.
    ///
    /// This method consumes the `BoolVector` and returns an iterator that yields `None` for
    /// null values and `Some(value)` for valid values.
    ///
    /// # Examples
    ///
    /// ```
    /// use vortex_vector::bool::BoolVector;
    ///
    /// let vec = BoolVector::from_iter([Some(true), None, Some(false), Some(true)]);
    /// let collected: Vec<_> = vec.into_iter().collect();
    /// assert_eq!(collected, vec![Some(true), None, Some(false), Some(true)]);
    /// ```
    fn into_iter(self) -> Self::IntoIter {
        BoolVectorIterator {
            vector: self,
            index: 0,
        }
    }
}
