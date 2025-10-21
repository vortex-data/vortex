// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`PVector<T>`].

use vortex_buffer::Buffer;
use vortex_dtype::NativePType;
use vortex_mask::Mask;

use crate::{PVectorMut, VectorOps};

/// An immutable vector of generic primitive values.
///
/// `T` is expected to be bound by [`NativePType`], which templates an internal [`Buffer<T>`] that
/// stores the elements of the vector.
///
/// The mutable equivalent of this type is [`PVectorMut<T>`].
#[derive(Debug, Clone)]
pub struct PVector<T> {
    pub(super) elements: Buffer<T>,
    pub(super) validity: Mask,
}

impl<T: NativePType> PVector<T> {
    /// Creates a new [`PVector<T>`] from an iterator of `Option<T>` values.
    ///
    /// `None` values will be marked as invalid in the validity mask.
    ///
    /// # Examples
    ///
    /// ```
    /// use vortex_vector::{PVector, VectorOps};
    ///
    /// let vec = PVector::<i32>::from_option_iter([Some(1), None, Some(3)]);
    /// assert_eq!(vec.len(), 3);
    /// ```
    pub fn from_option_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Option<T>>,
    {
        let iter = iter.into_iter();
        let (lower_bound, _) = iter.size_hint();

        let mut elements = Vec::with_capacity(lower_bound);
        let mut validity = Vec::with_capacity(lower_bound);

        for opt_val in iter {
            match opt_val {
                Some(val) => {
                    elements.push(val);
                    validity.push(true);
                }
                None => {
                    elements.push(T::default()); // Use default for invalid entries.
                    validity.push(false);
                }
            }
        }

        PVector {
            elements: Buffer::from(elements),
            validity: Mask::from_iter(validity),
        }
    }
}

impl<T: NativePType> VectorOps for PVector<T> {
    type Mutable = PVectorMut<T>;

    fn len(&self) -> usize {
        self.elements.len()
    }

    fn validity(&self) -> &Mask {
        &self.validity
    }

    /// Try to convert self into a mutable vector.
    fn try_into_mut(self) -> Result<PVectorMut<T>, Self> {
        let elements = match self.elements.try_into_mut() {
            Ok(elements) => elements,
            Err(elements) => {
                return Err(PVector {
                    elements,
                    validity: self.validity,
                });
            }
        };

        match self.validity.try_into_mut() {
            Ok(validity_mut) => Ok(PVectorMut {
                elements,
                validity: validity_mut,
            }),
            Err(validity) => Err(PVector {
                elements: elements.freeze(),
                validity,
            }),
        }
    }
}
