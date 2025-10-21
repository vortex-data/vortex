// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`BoolVector`].

use vortex_buffer::BitBuffer;
use vortex_mask::Mask;

use super::BoolVectorMut;
use crate::VectorOps;

/// An immutable vector of boolean values.
///
/// The mutable equivalent of this type is [`BoolVectorMut`].
#[derive(Debug, Clone)]
pub struct BoolVector {
    pub(super) bits: BitBuffer,
    pub(super) validity: Mask,
}

impl BoolVector {
    /// Creates a new [`BoolVector`] from an iterator of `Option<bool>` values.
    ///
    /// `None` values will be marked as invalid in the validity mask.
    ///
    /// # Examples
    ///
    /// ```
    /// use vortex_vector::{BoolVector, VectorOps};
    ///
    /// let vec = BoolVector::from_option_iter([Some(true), None, Some(false)]);
    /// assert_eq!(vec.len(), 3);
    /// ```
    pub fn from_option_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Option<bool>>,
    {
        let iter = iter.into_iter();
        let (lower_bound, _) = iter.size_hint();

        let mut bits = Vec::with_capacity(lower_bound);
        let mut validity = Vec::with_capacity(lower_bound);

        for opt_val in iter {
            match opt_val {
                Some(val) => {
                    bits.push(val);
                    validity.push(true);
                }
                None => {
                    bits.push(false); // Value doesn't matter for invalid entries.
                    validity.push(false);
                }
            }
        }

        BoolVector {
            bits: BitBuffer::from_iter(bits),
            validity: Mask::from_iter(validity),
        }
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
