// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBufferMut;
use vortex_dtype::{DType, Nullability};
use vortex_mask::MaskMut;

use super::BoolVector;
use crate::VectorMutOps;

/// A mutable vector of boolean values.
pub struct BoolVectorMut {
    pub(super) bits: BitBufferMut,
    pub(super) validity: Option<MaskMut>,
}

impl BoolVectorMut {
    /// Create a new mutable boolean vector with the given capacity and nullability.
    pub fn with_capacity(capacity: usize, nullability: Nullability) -> Self {
        let validity = match nullability {
            Nullability::NonNullable => None,
            Nullability::Nullable => Some(MaskMut::with_capacity(capacity)),
        };

        Self {
            bits: BitBufferMut::with_capacity(capacity),
            validity,
        }
    }
}

impl VectorMutOps for BoolVectorMut {
    type Immutable = BoolVector;

    fn nullability(&self) -> Nullability {
        Nullability::from(self.validity.is_some())
    }

    fn dtype(&self) -> DType {
        DType::Bool(self.nullability())
    }

    fn len(&self) -> usize {
        self.bits.len()
    }

    fn capacity(&self) -> usize {
        self.bits.capacity()
    }

    fn reserve(&mut self, additional: usize) {
        self.bits.reserve(additional);
        if let Some(v) = self.validity.as_mut() {
            v.reserve(additional);
        }
    }

    fn extend_from_vector(&mut self, other: &BoolVector) {
        self.bits.append_buffer(&other.bits);
        match (&mut self.validity, &other.validity) {
            (Some(self_v), Some(other_v)) => self_v.append_mask(other_v),
            (Some(self_v), None) => self_v.append_n(true, other.bits.len()),
            (None, Some(other_v)) => {
                let mut new_validity = MaskMut::new_true(self.bits.len() - other.bits.len());
                new_validity.append_mask(other_v);
                self.validity = Some(new_validity);
            }
            (None, None) => {}
        }
    }

    fn freeze(self) -> Self::Immutable {
        BoolVector {
            bits: self.bits.freeze(),
            validity: self.validity.map(|v| v.freeze()),
        }
    }

    fn split_off(&mut self, at: usize) -> Self {
        BoolVectorMut {
            bits: self.bits.split_off(at),
            validity: self.validity.as_mut().map(|v| v.split_off(at)),
        }
    }

    fn unsplit(&mut self, other: Self) {
        let other_len = other.bits.len();
        self.bits.unsplit(other.bits);
        match (&mut self.validity, other.validity) {
            (Some(self_v), Some(other_v)) => self_v.unsplit(other_v),
            (Some(self_v), None) => self_v.append_n(true, other_len),
            (None, Some(other_v)) => {
                let mut new_validity = MaskMut::new_true(self.bits.len() - other_len);
                new_validity.unsplit(other_v);
                self.validity = Some(new_validity);
            }
            (None, None) => {}
        }
    }
}
