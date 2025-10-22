// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`BoolVectorMut`].

use vortex_buffer::BitBufferMut;
use vortex_mask::MaskMut;

use super::BoolVector;
use crate::{VectorMutOps, VectorOps};

/// A mutable vector of boolean values.
///
/// The immutable equivalent of this type is [`BoolVector`].
#[derive(Debug, Clone)]
pub struct BoolVectorMut {
    pub(super) bits: BitBufferMut,
    pub(super) validity: MaskMut,
}

impl BoolVectorMut {
    /// Creates a new mutable boolean vector with the given `capacity`.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bits: BitBufferMut::with_capacity(capacity),
            validity: MaskMut::with_capacity(capacity),
        }
    }
}

impl VectorMutOps for BoolVectorMut {
    type Immutable = BoolVector;

    fn len(&self) -> usize {
        debug_assert!(self.validity.len() == self.bits.len());

        self.bits.len()
    }

    fn capacity(&self) -> usize {
        self.bits.capacity()
    }

    fn reserve(&mut self, additional: usize) {
        self.bits.reserve(additional);
        self.validity.reserve(additional);
    }

    fn extend_from_vector(&mut self, other: &BoolVector) {
        self.bits.append_buffer(&other.bits);
        self.validity.append_mask(other.validity());
    }

    fn append_nulls(&mut self, n: usize) {
        self.bits.append_n(false, n);
        self.validity.append_n(false, n);
    }

    fn freeze(self) -> Self::Immutable {
        BoolVector {
            bits: self.bits.freeze(),
            validity: self.validity.freeze(),
        }
    }

    fn split_off(&mut self, at: usize) -> Self {
        BoolVectorMut {
            bits: self.bits.split_off(at),
            validity: self.validity.split_off(at),
        }
    }

    fn unsplit(&mut self, other: Self) {
        self.bits.unsplit(other.bits);
        self.validity.unsplit(other.validity);
    }
}
