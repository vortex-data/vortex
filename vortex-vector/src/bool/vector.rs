// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`BoolVector`].

use vortex_buffer::BitBuffer;
use vortex_error::{VortexExpect, VortexResult, vortex_ensure};
use vortex_mask::Mask;

use crate::VectorOps;
use crate::bool::BoolVectorMut;

/// An immutable vector of boolean values.
///
/// Internally, this `BoolVector` is a wrapper around a [`BitBuffer`] and a validity mask.
#[derive(Debug, Clone)]
pub struct BoolVector {
    /// The bits that we use to represent booleans.
    pub(super) bits: BitBuffer,
    /// The validity mask (where `true` represents an element is **not** null).
    pub(super) validity: Mask,
}

impl BoolVector {
    /// Creates a new [`BoolVector`] from the given bits and validity mask.
    ///
    /// # Panics
    ///
    /// Panics if the length of the validity mask does not match the length of the bits.
    pub fn new(bits: BitBuffer, validity: Mask) -> Self {
        Self::try_new(bits, validity).vortex_expect("Failed to create `BoolVector`")
    }

    /// Tries to create a new [`BoolVector`] from the given bits and validity mask.
    ///
    /// # Errors
    ///
    /// Returns an error if the length of the validity mask does not match the length of the bits.
    pub fn try_new(bits: BitBuffer, validity: Mask) -> VortexResult<Self> {
        vortex_ensure!(
            validity.len() == bits.len(),
            "`BoolVector` validity mask must have the same length as bits"
        );

        Ok(Self { bits, validity })
    }

    /// Creates a new [`BoolVector`] from the given bits and validity mask without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the validity mask has the same length as the bits.
    pub unsafe fn new_unchecked(bits: BitBuffer, validity: Mask) -> Self {
        if cfg!(debug_assertions) {
            Self::new(bits, validity)
        } else {
            Self { bits, validity }
        }
    }

    /// Decomposes the boolean vector into its constituent parts (bit buffer and validity).
    pub fn into_parts(self) -> (BitBuffer, Mask) {
        (self.bits, self.validity)
    }

    /// Returns the bits buffer of the boolean vector.
    pub fn bits(&self) -> &BitBuffer {
        &self.bits
    }
}

impl VectorOps for BoolVector {
    type Mutable = BoolVectorMut;

    fn len(&self) -> usize {
        debug_assert!(self.validity.len() == self.bits.len());
        self.bits.len()
    }

    fn validity(&self) -> &Mask {
        &self.validity
    }

    fn try_into_mut(self) -> Result<BoolVectorMut, Self>
    where
        Self: Sized,
    {
        let bits = match self.bits.try_into_mut() {
            Ok(bits) => bits,
            Err(bits) => {
                return Err(BoolVector {
                    bits,
                    validity: self.validity,
                });
            }
        };

        match self.validity.try_into_mut() {
            Ok(validity_mut) => Ok(BoolVectorMut {
                bits,
                validity: validity_mut,
            }),
            Err(validity) => Err(BoolVector {
                bits: bits.freeze(),
                validity,
            }),
        }
    }
}
