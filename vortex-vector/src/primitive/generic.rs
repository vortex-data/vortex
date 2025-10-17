// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_dtype::{DType, NativePType, Nullability};
use vortex_mask::Mask;

use crate::{GenericPVectorMut, PrimitiveVector, Vector, VectorOps};

/// An immutable vector of generic primitive values.
pub struct GenericPVector<T> {
    pub(super) elements: Buffer<T>,
    pub(super) validity: Option<Mask>,
}

impl<T: NativePType> From<GenericPVector<T>> for Vector {
    fn from(v: GenericPVector<T>) -> Self {
        Self::Primitive(PrimitiveVector::from(v))
    }
}

impl<T: NativePType> VectorOps for GenericPVector<T> {
    type Mutable = GenericPVectorMut<T>;

    fn nullability(&self) -> Nullability {
        Nullability::from(self.validity.is_some())
    }

    fn dtype(&self) -> DType {
        DType::Primitive(T::PTYPE, self.nullability())
    }

    fn len(&self) -> usize {
        self.elements.len()
    }

    /// Try to convert self into a mutable vector.
    fn try_into_mut(self) -> Result<GenericPVectorMut<T>, Self> {
        let elements = match self.elements.try_into_mut() {
            Ok(elements) => elements,
            Err(elements) => {
                return Err(GenericPVector {
                    elements,
                    validity: self.validity,
                });
            }
        };

        let validity = match self.validity {
            Some(v) => match v.try_into_mut() {
                Ok(v) => Some(v),
                Err(v) => {
                    return Err(GenericPVector {
                        elements: elements.freeze(),
                        validity: Some(v),
                    });
                }
            },
            None => None,
        };

        Ok(GenericPVectorMut { elements, validity })
    }
}
