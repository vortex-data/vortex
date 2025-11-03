// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BufferMut;
use vortex_dtype::{NativeDecimalType, PrecisionScale};
use vortex_error::{VortexResult, vortex_bail};
use vortex_mask::MaskMut;

use crate::{DVector, VectorMutOps, VectorOps};

/// A specifically typed mutable decimal vector.
#[derive(Debug, Clone)]
pub struct DVectorMut<D> {
    pub(super) ps: PrecisionScale<D>,
    pub(super) elements: BufferMut<D>,
    pub(super) validity: MaskMut,
}

impl<D: NativeDecimalType> DVectorMut<D> {
    /// Get the precision/scale of the decimal vector.
    pub fn precision_scale(&self) -> PrecisionScale<D> {
        self.ps
    }

    /// Get a nullable element at the given index.
    pub fn get(&self, index: usize) -> Option<&D> {
        self.validity.value(index).then(|| &self.elements[index])
    }

    /// Appends a new element to the end of the vector.
    ///
    /// # Errors
    ///
    /// Returns an error if the value is out of bounds for the vector's precision/scale.
    pub fn try_push(&mut self, value: D) -> VortexResult<()> {
        if !self.ps.is_valid(value) {
            vortex_bail!("Value {:?} is out of bounds for {}", value, self.ps,);
        }

        self.elements.push(value);
        self.validity.append_n(true, 1);
        Ok(())
    }

    /// Returns a mutable reference to the underlying elements buffer.
    ///
    /// # Safety
    ///
    /// Modifying the elements buffer directly may violate the precision/scale constraints.
    /// The caller must ensure that any modifications maintain these invariants.
    pub unsafe fn elements_mut(&mut self) -> &mut [D] {
        &mut self.elements
    }
}

impl<D: NativeDecimalType> AsRef<[D]> for DVectorMut<D> {
    fn as_ref(&self) -> &[D] {
        &self.elements
    }
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

    fn extend_from_vector(&mut self, other: &Self::Immutable) {
        self.elements.extend_from_slice(&other.elements);
        self.validity.append_mask(other.validity());
    }

    fn append_nulls(&mut self, n: usize) {
        self.elements.extend((0..n).map(|_| D::default()));
        self.validity.append_n(false, n);
    }

    fn freeze(self) -> Self::Immutable {
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
