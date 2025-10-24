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
        // Since we do not know the length of the iterator, we can only guess how much memory we
        // need to reserve. Note that these hints may be inaccurate.
        let (lower_bound, upper_bound_opt) = iter.size_hint();

        // In the case that the upper bound is adversarial, we put a hard limit on the amount of
        // memory we reserve (and the OS should handle the rest with zero pages).
        let reserve_amount = upper_bound_opt.unwrap_or(lower_bound);

        let mut bits = BitBufferMut::with_capacity(reserve_amount);
        let mut validity = MaskMut::with_capacity(reserve_amount);

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
