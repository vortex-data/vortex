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
