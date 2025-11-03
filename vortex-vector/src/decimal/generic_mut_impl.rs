// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::{NativeDecimalType, PrecisionScale};
use vortex_error::{VortexResult, vortex_bail};

use crate::DVectorMut;

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
