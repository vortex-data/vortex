// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`PVector<T>`].

use vortex_buffer::Buffer;
use vortex_dtype::NativePType;
use vortex_error::{VortexExpect, VortexResult, vortex_ensure};
use vortex_mask::Mask;

use crate::{PVectorMut, VectorOps};

/// An immutable vector of generic primitive values.
///
/// `T` is expected to be bound by [`NativePType`], which templates an internal [`Buffer<T>`] that
/// stores the elements of the vector.
///
/// `PVector<T>` can be considered a borrowed / frozen  version of [`PVectorMut<T>`], which is
/// created via the [`freeze`](crate::VectorMutOps::freeze) method.
///
/// See the documentation for [`PVectorMut<T>`] for more information.
#[derive(Debug, Clone)]
pub struct PVector<T> {
    /// The buffer representing the vector elements.
    pub(super) elements: Buffer<T>,
    /// The validity mask (where `true` represents an element is **not** null).
    pub(super) validity: Mask,
}

impl<T> PVector<T> {
    /// Creates a new [`PVector<T>`] from the given elements buffer and validity mask.
    ///
    /// # Panics
    ///
    /// Panics if the length of the validity mask does not match the length of the elements buffer.
    pub fn new(elements: Buffer<T>, validity: Mask) -> Self {
        Self::try_new(elements, validity).vortex_expect("Failed to create `PVector`")
    }

    /// Tries to create a new [`PVector<T>`] from the given elements buffer and validity mask.
    ///
    /// # Errors
    ///
    /// Returns an error if the length of the validity mask does not match the length of the
    /// elements buffer.
    pub fn try_new(elements: Buffer<T>, validity: Mask) -> VortexResult<Self> {
        vortex_ensure!(
            validity.len() == elements.len(),
            "`PVector` validity mask must have the same length as elements"
        );

        Ok(Self { elements, validity })
    }

    /// Creates a new [`PVector<T>`] from the given elements buffer and validity mask without
    /// validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the validity mask has the same length as the elements buffer.
    pub unsafe fn new_unchecked(elements: Buffer<T>, validity: Mask) -> Self {
        if cfg!(debug_assertions) {
            Self::new(elements, validity)
        } else {
            Self { elements, validity }
        }
    }

    /// Decomposes the primitive vector into its constituent parts (buffer and validity).
    pub fn into_parts(self) -> (Buffer<T>, Mask) {
        (self.elements, self.validity)
    }

    /// Gets a nullable element at the given index, panicking on out-of-bounds.
    ///
    /// If the element at the given index is null, returns `None`. Otherwise, returns `Some(x)`,
    /// where `x: T`.
    ///
    /// Note that this `get` method is different from the standard library [`slice::get`], which
    /// returns `None` if the index is out of bounds. This method will panic if the index is out of
    /// bounds, and return `None` if the elements is null.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn get(&self, index: usize) -> Option<&T> {
        self.validity.value(index).then(|| &self.elements[index])
    }

    /// Returns the internal [`Buffer`] of the [`PVector`].
    ///
    /// Note that the internal buffer may hold garbage data in place of nulls. That information is
    /// tracked by the [`validity()`](Self::validity).
    #[inline]
    pub fn elements(&self) -> &Buffer<T> {
        &self.elements
    }
}

impl<T: NativePType> AsRef<[T]> for PVector<T> {
    /// Returns an immutable slice over the internal buffer with elements of type `T`.
    ///
    /// Note that this slice may contain garbage data where the [`validity()`] mask states that an
    /// element is invalid.
    ///
    /// The caller should check the [`validity()`] before performing any operations.
    ///
    /// [`validity()`]: crate::VectorOps::validity
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.elements.as_slice()
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
