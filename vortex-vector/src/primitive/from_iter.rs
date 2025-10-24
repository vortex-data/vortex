// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`FromIterator`] and related implementations for [`PVecMut<T>`].

use vortex_buffer::BufferMut;
use vortex_dtype::NativePType;
use vortex_mask::MaskMut;

use crate::PVecMut;

impl<T: NativePType> FromIterator<Option<T>> for PVecMut<T> {
    /// Creates a new [`PVecMut<T>`] from an iterator of `Option<T>` values.
    ///
    /// `None` values will be marked as invalid in the validity mask.
    ///
    /// # Examples
    ///
    /// ```
    /// use vortex_vector::{PVecMut, VectorMutOps};
    ///
    /// let mut vec = PVecMut::<i32>::from_iter([Some(1), None, Some(3)]);
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

        PVecMut {
            elements: BufferMut::from_iter(elements),
            validity,
        }
    }
}

impl<T: NativePType> FromIterator<T> for PVecMut<T> {
    /// Creates a new [`PVecMut<T>`] from an iterator of `T` values.
    ///
    /// All values will be treated as non-null.
    ///
    /// # Examples
    ///
    /// ```
    /// use vortex_vector::{PVecMut, VectorMutOps};
    ///
    /// let mut vec = PVecMut::<i32>::from_iter([1, 2, 3, 4]);
    /// assert_eq!(vec.len(), 4);
    /// ```
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let buffer = BufferMut::from_iter(iter);
        let validity = MaskMut::new_true(buffer.len());

        PVecMut {
            elements: buffer,
            validity,
        }
    }
}
