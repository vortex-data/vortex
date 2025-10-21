// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`PVector<T>`].

use vortex_buffer::Buffer;
use vortex_dtype::{NativePType, Nullability};
use vortex_mask::Mask;

use crate::{PVectorMut, VectorOps};

/// An immutable vector of generic primitive values.
///
/// `T` is expected to be bound by [`NativePType`], which templates an internal [`Buffer<T>`] that
/// stores the elements of the vector. Additionally, an optional [`Mask`] is stored to track null
/// primitive elements (where `true` represents a _valid_ primitive and `false` represents a `null`
/// primitive).
///
/// The mutable equivalent of this type is [`PVectorMut<T>`].
#[derive(Debug, Clone)]
pub struct PVector<T> {
    pub(super) elements: Buffer<T>,
    pub(super) validity: Option<Mask>,
}

impl<T: NativePType> VectorOps for PVector<T> {
    type Mutable = PVectorMut<T>;

    fn nullability(&self) -> Nullability {
        Nullability::from(self.validity.is_some())
    }

    fn len(&self) -> usize {
        self.elements.len()
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

        let validity = match self.validity {
            Some(v) => match v.try_into_mut() {
                Ok(v) => Some(v),
                Err(v) => {
                    return Err(PVector {
                        elements: elements.freeze(),
                        validity: Some(v),
                    });
                }
            },
            None => None,
        };

        Ok(PVectorMut { elements, validity })
    }
}
