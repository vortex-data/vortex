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

impl<T: NativePType> PVector<T> {
    /// Creates a new [`PVector<T>`] from the given elements buffer and validity mask.
    ///
    /// # Panics
    ///
    /// Panics if the length of the validity mask does not match the length of the elements buffer.
    pub fn new(elements: Buffer<T>, validity: Mask) -> Self {
        Self::try_new(elements, validity)
            .vortex_expect("`PVector` validity mask must have the same length as elements")
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
    pub fn new_unchecked(elements: Buffer<T>, validity: Mask) -> Self {
        debug_assert_eq!(
            validity.len(),
            elements.len(),
            "`PVector` validity mask must have the same length as elements"
        );

        Self { elements, validity }
    }

    /// Gets a nullable element at the given index, with bounds checking.
    ///
    /// If the index is out of bounds, returns `None`. If the element at the given index is null,
    /// returns `Some(None)`. Otherwise, returns `Some(Some(x))`, where `x: T`.
    pub fn get_checked(&self, index: usize) -> Option<Option<T>> {
        (index < self.len()).then(|| {
            self.validity.value(index).then(|| {
                self.elements
                    .get(index)
                    .copied()
                    .vortex_expect("length of elements was somehow incorrect")
            })
        })
    }

    /// Gets a nullable element at the given index.
    ///
    /// If the element at the given index is null, returns `None`. Otherwise, returns `Some(x)`,
    /// where `x: T`.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn get(&self, index: usize) -> Option<T> {
        assert!(
            index < self.len(),
            "index out of bounds: the length is {} but the index is {index}",
            self.len()
        );

        self.validity.value(index).then(|| {
            self.elements
                .get(index)
                .copied()
                .vortex_expect("length of elements was somehow incorrect")
        })
    }

    /// Gets a nullable element at the given index, without checking bounds and without checking
    /// nullability.
    ///
    /// The caller should ensure that the element at the given index is not null (though doing so
    /// will not cause undefined behavior).
    ///
    /// # Safety
    ///
    /// The caller must ensure that the index is within bounds.
    pub unsafe fn get_unchecked(&self, index: usize) -> T {
        debug_assert!(
            index < self.len(),
            "index out of bounds: the length is {} but the index is {index}",
            self.len()
        );

        // SAFETY: The caller ensures that the index is in bounds.
        unsafe { *self.elements.get_unchecked(index) }
    }

    /// Returns a slice over the internal buffer with elements of type `T`.
    ///
    /// Note that this slice may contain garbage data where the [`validity()`] mask states that an
    /// element is invalid.
    ///
    /// The caller should check the [`validity()`] before performing any operations.
    ///
    /// [`validity()`]: Self::validity
    pub fn as_slice(&self) -> &[T] {
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
