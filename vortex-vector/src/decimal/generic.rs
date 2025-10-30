// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_dtype::{NativeDecimalType, PrecisionScale};
use vortex_error::{vortex_bail, VortexResult};
use vortex_mask::Mask;

/// A specifically typed decimal vector.
#[derive(Debug, Clone)]
pub struct DVector<D> {
    ps: PrecisionScale<D>,
    elements: Buffer<D>,
    validity: Mask,
}

impl<D: NativeDecimalType> DVector<D> {
    /// Try to create a new decimal vector from the given elements and validity.
    ///
    /// # Errors
    ///
    /// Returns an error if the precision/scale is invalid, the lengths of the elements
    /// and validity do not match, or any of the elements are out of bounds for the given
    /// precision/scale.
    pub fn try_new(
        ps: PrecisionScale<D>,
        elements: Buffer<D>,
        validity: Mask,
    ) -> VortexResult<Self> {
        if elements.len() != validity.len() {
            vortex_bail!(
                "Elements length {} does not match validity length {}",
                elements.len(),
                validity.len()
            );
        }

        // We assert that each element is within bounds for the given precision/scale.
        if !elements.iter().all(|e| ps.is_valid(*e)) {
            vortex_bail!(
                "One or more elements are out of bounds for precision {} and scale {}",
                ps.precision(),
                ps.scale()
            );
        }

        Ok(Self {
            ps,
            elements,
            validity,
        })
    }
}
