// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`BoolVector`].

use vortex_buffer::BitBuffer;
use vortex_mask::Mask;

use super::BoolVectorMut;
use crate::VectorOps;

/// An immutable vector of boolean values.
///
/// `BoolVector` can be considered a borrowed / frozen version of [`BoolVectorMut`], which is
/// created via the [`freeze`](crate::VectorMutOps::freeze) method.
///
/// See the documentation for [`BoolVectorMut`] for more information.
#[derive(Debug, Clone)]
pub struct BoolVector {
    /// The bits that we use to represent booleans.
    pub(super) bits: BitBuffer,
    /// The validity mask (where `true` represents an element is **not** null).
    pub(super) validity: Mask,
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
