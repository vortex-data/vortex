// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::decimal::precision::PrecisionScale;
use vortex_buffer::Buffer;
use vortex_error::{vortex_bail, VortexResult};
use vortex_mask::Mask;

/// A specifically typed decimal vector.
#[derive(Debug, Clone)]
pub struct DVector<D> {
    ps: PrecisionScale<D>,
    elements: Buffer<D>,
    validity: Mask,
}

impl<D> DVector<D> {
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
        vortex_bail!("Not implemented")
    }
}
