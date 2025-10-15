// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_dtype::{DType, NativePType};
use vortex_mask::Mask;

use crate::ops::VectorOps;
use crate::{PVector, PrimitiveVectorMut, Vector};

/// An immutable vector of primitive values.
pub struct PrimitiveVector<T> {
    pub(super) dtype: DType,
    pub(super) elements: Buffer<T>,
    pub(super) validity: Mask,
}

impl<T: NativePType> From<PrimitiveVector<T>> for Vector {
    fn from(v: PrimitiveVector<T>) -> Self {
        Self::Primitive(PVector::from(v))
    }
}

impl<T: NativePType> VectorOps for PrimitiveVector<T> {
    type Mutable = PrimitiveVectorMut<T>;

    /// Returns the length of the vector.
    fn len(&self) -> usize {
        self.elements.len()
    }

    /// Returns the data type of the vector.
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Try to convert self into a mutable vector.
    fn try_into_mut(self) -> Result<PrimitiveVectorMut<T>, Self> {
        let elements = match self.elements.try_into_mut() {
            Ok(elements) => elements,
            Err(elements) => {
                return Err(PrimitiveVector {
                    dtype: self.dtype,
                    elements,
                    validity: self.validity,
                });
            }
        };

        let validity = match self.validity.try_into_mut() {
            Ok(validity) => validity,
            Err(validity) => {
                return Err(PrimitiveVector {
                    dtype: self.dtype,
                    elements: elements.freeze(),
                    validity,
                });
            }
        };

        Ok(PrimitiveVectorMut {
            dtype: self.dtype,
            elements,
            validity,
        })
    }
}
