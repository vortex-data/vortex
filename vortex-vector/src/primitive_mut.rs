// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BufferMut;
use vortex_dtype::{DType, NativePType, Nullability};
use vortex_mask::MaskMut;

use crate::ops::VectorMutOps;
use crate::{PVectorMut, PrimitiveVector, VectorMut};

/// A mutable vector of primitive values.
pub struct PrimitiveVectorMut<T> {
    pub(super) dtype: DType,
    pub(super) elements: BufferMut<T>,
    pub(super) validity: MaskMut,
}

impl<T: NativePType> PrimitiveVectorMut<T> {
    /// Create a new mutable primitive vector with the given capacity and nullability.
    pub fn with_capacity(capacity: usize, nullability: Nullability) -> Self {
        Self {
            dtype: DType::Primitive(T::PTYPE, nullability),
            elements: BufferMut::with_capacity(capacity),
            validity: MaskMut::with_capacity(capacity),
        }
    }
}

impl<T: NativePType> From<PrimitiveVectorMut<T>> for VectorMut {
    fn from(val: PrimitiveVectorMut<T>) -> Self {
        VectorMut::Primitive(PVectorMut::from(val))
    }
}

impl<T: NativePType> VectorMutOps for PrimitiveVectorMut<T> {
    type Immutable = PrimitiveVector<T>;

    fn len(&self) -> usize {
        self.elements.len()
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn capacity(&self) -> usize {
        self.elements.capacity()
    }

    fn reserve(&mut self, additional: usize) {
        self.elements.reserve(additional);
        self.validity.reserve(additional);
    }

    fn split_off(&mut self, at: usize) -> Self {
        PrimitiveVectorMut {
            dtype: self.dtype.clone(),
            elements: self.elements.split_off(at),
            validity: self.validity.split_off(at),
        }
    }

    fn unsplit(&mut self, other: Self) {
        self.elements.unsplit(other.elements);
        self.validity.unsplit(other.validity);
    }

    /// Extends the vector by appending elements from another vector.
    fn extend_from_vector(&mut self, other: &PrimitiveVector<T>) {
        self.elements.extend_from_slice(other.elements.as_slice());
        self.validity.append_mask(&other.validity);
    }

    /// Freeze the vector into an immutable one.
    fn freeze(self) -> PrimitiveVector<T> {
        PrimitiveVector {
            dtype: self.dtype,
            elements: self.elements.freeze(),
            validity: self.validity.freeze(),
        }
    }
}
