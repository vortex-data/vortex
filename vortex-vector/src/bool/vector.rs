// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`BoolVector`].

use vortex_buffer::BitBuffer;
use vortex_mask::Mask;

use super::BoolVectorMut;
use crate::VectorOps;

/// An immutable vector of boolean values.
///
/// Internally, the boolean values are stored as the bits of a [`BitBuffer`] plus an optional
/// [`Mask`] for null booleans (where `true` represents a _valid_ boolean and `false` represents a
/// `null` boolean).
///
/// The mutable equivalent of this type is [`BoolVectorMut`].
#[derive(Debug, Clone)]
pub struct BoolVector {
    pub(super) bits: BitBuffer,
    pub(super) validity: Mask,
}

impl BoolVector {
    /// Creates a new [`BoolVector`] from the given bits and validity mask.
    ///
    /// # Panics
    ///
    /// Panics if the length of the validity mask does not match the length of the bits.
    pub fn new(bits: BitBuffer, validity: Mask) -> Self {
        assert_eq!(
            validity.len(),
            bits.len(),
            "BoolVector validity mask must have the same length as bits"
        );
        Self { bits, validity }
    }

    /// Returns the bits buffer of the boolean vector.
    pub fn bits(&self) -> &BitBuffer {
        &self.bits
    }
}

impl VectorOps for BoolVector {
    type Mutable = BoolVectorMut;

    fn validity(&self) -> &Mask {
        &self.validity
    }

    fn len(&self) -> usize {
        debug_assert!(self.validity.len() == self.bits.len(),);
        self.bits.len()
    }

    fn try_into_mut(self) -> Result<Self::Mutable, Self>
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

        let validity = match self.validity {
            Some(v) => match v.try_into_mut() {
                Ok(v) => Some(v),
                Err(v) => {
                    return Err(BoolVector {
                        bits: bits.freeze(),
                        validity: Some(v),
                    });
                }
            },
            None => None,
        };

        Ok(BoolVectorMut { bits, validity })
    }
}
