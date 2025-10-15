// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBufferMut;
use vortex_dtype::{DType, Nullability};
use vortex_mask::MaskMut;

use crate::BoolVector;
use crate::ops::VectorMutOps;

/// A mutable vector of boolean values.
pub struct BoolVectorMut {
    pub(super) dtype: DType,
    pub(super) bits: BitBufferMut,
    pub(super) validity: MaskMut,
}

impl BoolVectorMut {
    /// Create a new mutable boolean vector with the given capacity and nullability.
    pub fn with_capacity(capacity: usize, nullability: Nullability) -> Self {
        Self {
            dtype: DType::Bool(nullability),
            bits: BitBufferMut::with_capacity(capacity),
            validity: MaskMut::with_capacity(capacity),
        }
    }
}

impl From<BoolVectorMut> for crate::VectorMut {
    fn from(v: BoolVectorMut) -> Self {
        Self::Bool(v)
    }
}

impl VectorMutOps for BoolVectorMut {
    type Immutable = BoolVector;

    fn len(&self) -> usize {
        self.bits.len()
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn capacity(&self) -> usize {
        self.bits.capacity()
    }

    fn reserve(&mut self, additional: usize) {
        self.bits.reserve(additional);
        self.validity.reserve(additional);
    }

    fn split_off(&mut self, at: usize) -> Self {
        BoolVectorMut {
            dtype: self.dtype.clone(),
            bits: self.bits.split_off(at),
            validity: self.validity.split_off(at),
        }
    }

    fn unsplit(&mut self, other: Self) {
        self.bits.unsplit(other.bits);
        self.validity.unsplit(other.validity);
    }

    fn extend_from_vector(&mut self, other: &BoolVector) {
        self.bits.append_buffer(&other.bits);
        self.validity.append_mask(&other.validity);
    }

    fn freeze(self) -> Self::Immutable {
        BoolVector {
            dtype: self.dtype,
            bits: self.bits.freeze(),
            validity: self.validity.freeze(),
        }
    }
}
