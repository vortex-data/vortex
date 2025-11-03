// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_dtype::{NativeDecimalType, PrecisionScale};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_mask::Mask;

use crate::{DVectorMut, VectorOps};

/// A specifically typed decimal vector.
#[derive(Debug, Clone)]
pub struct DVector<D> {
    pub(super) ps: PrecisionScale<D>,
    pub(super) elements: Buffer<D>,
    pub(super) validity: Mask,
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

    /// Create a new decimal vector from the given elements and validity without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the precision/scale is valid, the lengths of the elements
    /// and validity match, and all the elements are within bounds for the given precision/scale.
    pub unsafe fn new_unchecked(
        ps: PrecisionScale<D>,
        elements: Buffer<D>,
        validity: Mask,
    ) -> Self {
        if cfg!(debug_assertions) {
            Self::try_new(ps, elements, validity).vortex_expect("Failed to create `DVector`")
        } else {
            Self {
                ps,
                elements,
                validity,
            }
        }
    }

    /// Get the precision/scale of the decimal vector.
    pub fn precision_scale(&self) -> PrecisionScale<D> {
        self.ps
    }

    /// Decomposes the decimal vector into its constituent parts (precision/scale, buffer and validity).
    pub fn into_parts(self) -> (PrecisionScale<D>, Buffer<D>, Mask) {
        (self.ps, self.elements, self.validity)
    }
}

impl<D: NativeDecimalType> VectorOps for DVector<D> {
    type Mutable = DVectorMut<D>;

    fn len(&self) -> usize {
        self.elements.len()
    }

    fn validity(&self) -> &Mask {
        &self.validity
    }

    fn try_into_mut(self) -> Result<DVectorMut<D>, Self>
    where
        Self: Sized,
    {
        let elements = match self.elements.try_into_mut() {
            Ok(elements) => elements,
            Err(elements) => {
                return Err(DVector {
                    ps: self.ps,
                    elements,
                    validity: self.validity,
                });
            }
        };

        match self.validity.try_into_mut() {
            Ok(validity_mut) => Ok(DVectorMut {
                ps: self.ps,
                elements,
                validity: validity_mut,
            }),
            Err(validity) => Err(DVector {
                ps: self.ps,
                elements: elements.freeze(),
                validity,
            }),
        }
    }
}
