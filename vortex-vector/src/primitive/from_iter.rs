// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`FromIterator`] and related implementations for [`PVectorMut<T>`].

use vortex_buffer::BufferMut;
use vortex_dtype::NativePType;
use vortex_mask::MaskMut;

use crate::PVectorMut;

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
        // Since we do not know the length of the iterator, we can only guess how much memory we
        // need to reserve. Note that these hints may be inaccurate.
        let (lower_bound, upper_bound_opt) = iter.size_hint();

        // In the case that the upper bound is adversarial, we put a hard limit on the amount of
        // memory we reserve (and the OS should handle the rest with zero pages).
        let reserve_amount = upper_bound_opt
            .unwrap_or(lower_bound)
            .min(i32::MAX as usize);

        let mut elements = BufferMut::with_capacity(reserve_amount);
        let mut validity = MaskMut::with_capacity(reserve_amount);

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

        PVectorMut { elements, validity }
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
