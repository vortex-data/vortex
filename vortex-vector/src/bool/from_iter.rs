// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`FromIterator`] and related implementations for [`BoolVectorMut`].

use vortex_buffer::BitBufferMut;
use vortex_mask::MaskMut;

use crate::BoolVectorMut;

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
        let (lower_bound, _) = iter.size_hint();

        let mut bits = Vec::with_capacity(lower_bound);
        let mut validity = MaskMut::with_capacity(lower_bound);

        for opt_val in iter {
            match opt_val {
                Some(val) => {
                    bits.push(val);
                    validity.append_n(true, 1);
                }
                None => {
                    bits.push(false); // Value doesn't matter for invalid entries.
                    validity.append_n(false, 1);
                }
            }
        }

        BoolVectorMut {
            bits: BitBufferMut::from_iter(bits),
            validity,
        }
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
