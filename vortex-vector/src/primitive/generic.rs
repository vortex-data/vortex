// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`PVec<T>`].

use vortex_buffer::Buffer;
use vortex_dtype::NativePType;
use vortex_error::{VortexExpect, VortexResult, vortex_ensure};
use vortex_mask::Mask;

use crate::{PVecMut, VectorOps};

/// An immutable vector of generic primitive values.
///
/// `T` is expected to be bound by [`NativePType`], which templates an internal [`Buffer<T>`] that
/// stores the elements of the vector.
///
/// `PVec<T>` can be considered a borrowed / frozen  version of [`PVecMut<T>`], which is
/// created via the [`freeze`](crate::VectorMutOps::freeze) method.
///
/// See the documentation for [`PVecMut<T>`] for more information.
#[derive(Debug, Clone)]
pub struct PVec<T> {
    /// The buffer representing the vector elements.
    pub(super) elements: Buffer<T>,
    /// The validity mask (where `true` represents an element is **not** null).
    pub(super) validity: Mask,
}

impl<T: NativePType> PVec<T> {
    /// Creates a new [`PVec<T>`] from the given elements buffer and validity mask.
    ///
    /// # Panics
    ///
    /// Panics if the length of the validity mask does not match the length of the elements buffer.
    pub fn new(elements: Buffer<T>, validity: Mask) -> Self {
        Self::try_new(elements, validity)
            .vortex_expect("`PVec` validity mask must have the same length as elements")
    }

    /// Tries to create a new [`PVec<T>`] from the given elements buffer and validity mask.
    ///
    /// # Errors
    ///
    /// Returns an error if the length of the validity mask does not match the length of the
    /// elements buffer.
    pub fn try_new(elements: Buffer<T>, validity: Mask) -> VortexResult<Self> {
        vortex_ensure!(
            validity.len() == elements.len(),
            "`PVec` validity mask must have the same length as elements"
        );

        Ok(Self { elements, validity })
    }

    /// Creates a new [`PVec<T>`] from the given elements buffer and validity mask without
    /// validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the validity mask has the same length as the elements buffer.
    pub fn new_unchecked(elements: Buffer<T>, validity: Mask) -> Self {
        debug_assert_eq!(
            validity.len(),
            elements.len(),
            "`PVec` validity mask must have the same length as elements"
        );

        Self { elements, validity }
    }
}

impl<T: NativePType> VectorOps for PVec<T> {
    type Mutable = PVecMut<T>;

    fn len(&self) -> usize {
        self.elements.len()
    }

    fn validity(&self) -> &Mask {
        &self.validity
    }

    /// Try to convert self into a mutable vector.
    fn try_into_mut(self) -> Result<PVecMut<T>, Self> {
        let elements = match self.elements.try_into_mut() {
            Ok(elements) => elements,
            Err(elements) => {
                return Err(PVec {
                    elements,
                    validity: self.validity,
                });
            }
        };

        match self.validity.try_into_mut() {
            Ok(validity_mut) => Ok(PVecMut {
                elements,
                validity: validity_mut,
            }),
            Err(validity) => Err(PVec {
                elements: elements.freeze(),
                validity,
            }),
        }
    }
}
