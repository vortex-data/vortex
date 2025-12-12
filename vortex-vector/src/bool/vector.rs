// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`BoolVector`].

use std::fmt::Debug;
use std::ops::BitAnd;
use std::ops::RangeBounds;

use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use crate::VectorOps;
use crate::bool::BoolScalar;
use crate::bool::BoolVectorMut;

/// An immutable vector of boolean values.
///
/// Internally, this `BoolVector` is a wrapper around a [`BitBuffer`] and a validity mask.
#[derive(Debug, Clone, Eq)]
pub struct BoolVector {
    /// The bits that we use to represent booleans.
    pub(super) bits: BitBuffer,
    /// The validity mask (where `true` represents an element is **not** null).
    pub(super) validity: Mask,
}

impl PartialEq for BoolVector {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        // Validity patterns must match
        if self.validity != other.validity {
            return false;
        }
        // Use XNOR comparison: bits are equal where !(lhs ^ rhs) is true
        let lhs_chunks = self.bits.chunks();
        let rhs_chunks = other.bits.chunks();
        let validity_bits = self.validity.to_bit_buffer();
        let validity_chunks = validity_bits.chunks();

        // For equality: check that !(lhs ^ rhs) & validity == validity at each chunk
        for ((lhs, rhs), valid) in lhs_chunks
            .iter_padded()
            .zip(rhs_chunks.iter_padded())
            .zip(validity_chunks.iter_padded())
        {
            let equal_bits = !(lhs ^ rhs); // XNOR: true where bits are equal
            if (equal_bits & valid) != valid {
                return false;
            }
        }
        true
    }
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

    /// Consumes the boolean vector and returns the bits buffer.
    pub fn into_bits(self) -> BitBuffer {
        self.bits
    }

    /// Gets a nullable element at the given index, panicking on out-of-bounds.
    ///
    /// If the element at the given index is null, returns `None`. Otherwise, returns `Some(x)`,
    /// where `x: bool`.
    ///
    /// Note that this `get` method is different from the standard library [`slice::get`], which
    /// returns `None` if the index is out of bounds. This method will panic if the index is out of
    /// bounds, and return `None` if the element is null.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn get(&self, index: usize) -> Option<bool> {
        self.validity.value(index).then(|| self.bits.value(index))
    }
}

impl VectorOps for BoolVector {
    type Mutable = BoolVectorMut;
    type Scalar = BoolScalar;

    fn len(&self) -> usize {
        debug_assert!(self.validity.len() == self.bits.len());
        self.bits.len()
    }

    fn validity(&self) -> &Mask {
        &self.validity
    }

    fn mask_validity(&mut self, mask: &Mask) {
        self.validity = self.validity.bitand(mask);
    }

    fn scalar_at(&self, index: usize) -> BoolScalar {
        assert!(index < self.len());

        let is_valid = self.validity.value(index);
        let value = is_valid.then(|| self.bits.value(index));

        BoolScalar::new(value)
    }

    fn slice(&self, range: impl RangeBounds<usize> + Clone + Debug) -> Self {
        let bits = self.bits.slice(range.clone());
        let validity = self.validity.slice(range);
        Self { bits, validity }
    }

    fn clear(&mut self) {
        self.bits.clear();
        self.validity.clear();
    }

    fn try_into_mut(self) -> Result<BoolVectorMut, Self> {
        let bits = match self.bits.try_into_mut() {
            Ok(bits) => bits,
            Err(bits) => {
                return Err(Self {
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
            Err(validity) => Err(Self {
                bits: bits.freeze(),
                validity,
            }),
        }
    }

    fn into_mut(self) -> BoolVectorMut {
        BoolVectorMut {
            bits: self.bits.into_mut(),
            validity: self.validity.into_mut(),
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::BitBuffer;
    use vortex_mask::Mask;

    use super::*;

    #[test]
    fn test_bool_vector_eq_with_validity_127() {
        // Test with 127 elements (not a multiple of 64, tests edge cases)
        let len = 127;

        // Create bits: alternating true/false pattern
        let bits1: Vec<bool> = (0..len).map(|i| i % 2 == 0).collect();
        let mut bits2: Vec<bool> = bits1.clone();

        // Create validity: every 3rd element is invalid
        let validity_bools: Vec<bool> = (0..len).map(|i| i % 3 != 0).collect();
        let validity = Mask::from_buffer(BitBuffer::from(validity_bools));

        let v1 = BoolVector::new(BitBuffer::from(bits1.clone()), validity.clone());
        let v2 = BoolVector::new(BitBuffer::from(bits2.clone()), validity.clone());

        // Should be equal - same bits at valid positions
        assert_eq!(v1, v2);

        // Now modify bits2 at an INVALID position - should still be equal
        bits2[0] = !bits2[0]; // Flip bit 0, which is invalid (0 % 3 == 0)
        let v3 = BoolVector::new(BitBuffer::from(bits2.clone()), validity.clone());
        assert_eq!(v1, v3);

        // Now modify bits2 at a VALID position - should NOT be equal
        bits2[1] = !bits2[1]; // Flip bit 1, which is valid (1 % 3 != 0)
        let v4 = BoolVector::new(BitBuffer::from(bits2), validity);
        assert_ne!(v1, v4);

        // Test with different validity patterns - should NOT be equal
        let validity2 = Mask::new_true(len);
        let v5 = BoolVector::new(BitBuffer::from(bits1), validity2);
        assert_ne!(v1, v5);
    }
}
