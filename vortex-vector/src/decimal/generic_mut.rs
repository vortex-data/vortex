// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BufferMut;
use vortex_dtype::{NativeDecimalType, PrecisionScale};
use vortex_mask::MaskMut;

use crate::{DVector, VectorMutOps, VectorOps};

/// A specifically typed mutable decimal vector.
#[derive(Debug, Clone)]
pub struct DVectorMut<D> {
    pub(super) ps: PrecisionScale<D>,
    pub(super) elements: BufferMut<D>,
    pub(super) validity: MaskMut,
}

impl<D: NativeDecimalType> VectorMutOps for DVectorMut<D> {
    type Immutable = DVector<D>;

    fn len(&self) -> usize {
        self.elements.len()
    }

    fn capacity(&self) -> usize {
        self.elements.capacity()
    }

    fn reserve(&mut self, additional: usize) {
        self.elements.reserve(additional);
        self.validity.reserve(additional);
    }

    fn extend_from_vector(&mut self, other: &DVector<D>) {
        self.elements.extend_from_slice(&other.elements);
        self.validity.append_mask(other.validity());
    }

    fn append_nulls(&mut self, n: usize) {
        self.elements.extend((0..n).map(|_| D::default()));
        self.validity.append_n(false, n);
    }

    fn freeze(self) -> DVector<D> {
        DVector {
            ps: self.ps,
            elements: self.elements.freeze(),
            validity: self.validity.freeze(),
        }
    }

    fn split_off(&mut self, at: usize) -> Self {
        DVectorMut {
            ps: self.ps,
            elements: self.elements.split_off(at),
            validity: self.validity.split_off(at),
        }
    }

    fn unsplit(&mut self, other: Self) {
        self.elements.unsplit(other.elements);
        self.validity.unsplit(other.validity);
    }
}
