// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`BoolVectorMut`].

use vortex_buffer::BitBufferMut;
use vortex_error::{VortexExpect, VortexResult, vortex_ensure};
use vortex_mask::MaskMut;

use crate::bool::{BoolScalar, BoolVector};
use crate::{VectorMutOps, VectorOps};

/// A mutable vector of boolean values.
///
/// Internally, this `BoolVectorMut` is a wrapper around a [`BitBufferMut`] and a validity mask.
#[derive(Debug, Clone)]
pub struct BoolVectorMut {
    /// The mutable bits that we use to represent booleans.
    pub(super) bits: BitBufferMut,
    /// The validity mask (where `true` represents an element is **not** null).
    pub(super) validity: MaskMut,
}

impl BoolVectorMut {
    /// Creates a new [`BoolVectorMut`] from the given bits and validity mask.
    ///
    /// # Panics
    ///
    /// Panics if the length of the validity mask does not match the length of the bits.
    pub fn new(bits: BitBufferMut, validity: MaskMut) -> Self {
        Self::try_new(bits, validity).vortex_expect("Failed to create `BoolVectorMut`")
    }

    /// Tries to create a new [`BoolVectorMut`] from the given bits and validity mask.
    ///
    /// # Errors
    ///
    /// Returns an error if the length of the validity mask does not match the length of the bits.
    pub fn try_new(bits: BitBufferMut, validity: MaskMut) -> VortexResult<Self> {
        vortex_ensure!(
            validity.len() == bits.len(),
            "`BoolVector` validity mask must have the same length as bits"
        );

        Ok(Self { bits, validity })
    }

    /// Creates a new [`BoolVectorMut`] from the given bits and validity mask without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the validity mask has the same length as the bits.
    ///
    /// Ideally, they are taken from `into_parts`, mutated in a way that doesn't re-allocate, and
    /// then passed back to this function.
    pub unsafe fn new_unchecked(bits: BitBufferMut, validity: MaskMut) -> Self {
        if cfg!(debug_assertions) {
            Self::new(bits, validity)
        } else {
            Self { bits, validity }
        }
    }

    /// Creates a new mutable boolean vector with the given `capacity`.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bits: BitBufferMut::with_capacity(capacity),
            validity: MaskMut::with_capacity(capacity),
        }
    }

    /// Decomposes the boolean vector into its constituent parts (bit buffer and validity).
    pub fn into_parts(self) -> (BitBufferMut, MaskMut) {
        (self.bits, self.validity)
    }

    /// Append n values to the vector.
    pub fn append_values(&mut self, value: bool, n: usize) {
        self.bits.append_n(value, n);
        self.validity.append_n(true, n);
    }

    /// Returns a readonly handle to the bits backing the vector.
    pub fn bits(&self) -> &BitBufferMut {
        &self.bits
    }

    /// Returns a mutable handle to the bits backing the vector.
    ///
    /// # Safety
    ///
    /// Caller must ensure that bits and validity always have same length.
    pub unsafe fn bits_mut(&mut self) -> &mut BitBufferMut {
        &mut self.bits
    }

    /// Get a mutable handle to the validity mask of the vector.
    ///
    /// # Safety
    ///
    /// Caller must ensure that length of the validity always matches
    /// length of the bits.
    pub unsafe fn validity_mut(&mut self) -> &mut MaskMut {
        &mut self.validity
    }
}

impl VectorMutOps for BoolVectorMut {
    type Immutable = BoolVector;

    fn len(&self) -> usize {
        debug_assert!(self.validity.len() == self.bits.len());

        self.bits.len()
    }

    fn validity(&self) -> &MaskMut {
        &self.validity
    }

    fn capacity(&self) -> usize {
        self.bits.capacity()
    }

    fn reserve(&mut self, additional: usize) {
        self.bits.reserve(additional);
        self.validity.reserve(additional);
    }

    fn clear(&mut self) {
        self.bits.clear();
        self.validity.clear();
    }

    fn truncate(&mut self, len: usize) {
        self.bits.truncate(len);
        self.validity.truncate(len);
    }

    fn extend_from_vector(&mut self, other: &BoolVector) {
        self.bits.append_buffer(&other.bits);
        self.validity.append_mask(other.validity());
    }

    fn append_nulls(&mut self, n: usize) {
        self.bits.append_n(false, n); // Note that the value we push doesn't actually matter.
        self.validity.append_n(false, n);
    }

    fn append_zeros(&mut self, n: usize) {
        self.bits.append_n(false, n);
        self.validity.append_n(true, n);
    }

    fn append_scalars(&mut self, scalar: &BoolScalar, n: usize) {
        match scalar.value() {
            None => self.append_nulls(n),
            Some(value) => self.append_values(value, n),
        }
    }

    fn freeze(self) -> BoolVector {
        BoolVector {
            bits: self.bits.freeze(),
            validity: self.validity.freeze(),
        }
    }

    fn split_off(&mut self, at: usize) -> Self {
        Self {
            bits: self.bits.split_off(at),
            validity: self.validity.split_off(at),
        }
    }

    fn unsplit(&mut self, other: Self) {
        if self.is_empty() {
            *self = other;
            return;
        }
        self.bits.unsplit(other.bits);
        self.validity.unsplit(other.validity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_iter_with_options() {
        // Test FromIterator<Option<bool>> with nulls and empty.
        let vec_empty = BoolVectorMut::from_iter(std::iter::empty::<Option<bool>>());
        assert_eq!(vec_empty.len(), 0);

        let vec = BoolVectorMut::from_iter([Some(true), None, Some(false), None, Some(true)]);
        assert_eq!(vec.len(), 5);
        let frozen = vec.freeze();
        assert_eq!(frozen.validity().true_count(), 3);
    }

    #[test]
    fn test_from_iter_non_null() {
        // Test FromIterator<bool> creates all-valid vector.
        let vec = BoolVectorMut::from_iter([true, false, true, true, false]);
        assert_eq!(vec.len(), 5);
        let frozen = vec.freeze();
        assert_eq!(frozen.validity().true_count(), 5);
    }

    #[test]
    fn test_operations_preserve_validity() {
        // Comprehensive test for split/unsplit/extend preserving validity.
        let mut vec = BoolVectorMut::from_iter([Some(true), None, Some(false), None, Some(true)]);

        // Test split.
        let second_half = vec.split_off(2);
        assert_eq!(vec.len(), 2);
        assert_eq!(second_half.len(), 3);

        // Test validity after split.
        let frozen_first = vec.freeze();
        assert_eq!(frozen_first.validity().true_count(), 1);
        let frozen_second = second_half.freeze();
        assert_eq!(frozen_second.validity().true_count(), 2);

        // Test unsplit.
        let mut vec1 = BoolVectorMut::from_iter([Some(true), None]);
        let vec2 = BoolVectorMut::from_iter([Some(false), Some(true)]);
        vec1.unsplit(vec2);
        assert_eq!(vec1.len(), 4);
        let frozen = vec1.freeze();
        assert_eq!(frozen.validity().true_count(), 3);
    }

    #[test]
    fn test_into_iter_roundtrip() {
        // Test that from_iter followed by into_iter preserves the data.
        let original_data = vec![
            Some(true),
            None,
            Some(false),
            Some(true),
            None,
            Some(false),
            None,
            Some(true),
        ];

        // Create vector from iterator.
        let vec = BoolVectorMut::from_iter(original_data.clone());

        // Convert back to iterator and collect.
        let roundtrip: Vec<_> = vec.into_iter().collect();

        // Should be identical.
        assert_eq!(roundtrip, original_data);

        // Also test with all valid values.
        let all_valid = vec![true, false, true, false, true];
        let vec = BoolVectorMut::from_iter(all_valid.clone());
        let roundtrip: Vec<_> = vec.into_iter().collect();
        let expected: Vec<_> = all_valid.into_iter().map(Some).collect();
        assert_eq!(roundtrip, expected);

        // Test with empty.
        let empty: Vec<Option<bool>> = vec![];
        let vec = BoolVectorMut::from_iter(empty.clone());
        let roundtrip: Vec<_> = vec.into_iter().collect();
        assert_eq!(roundtrip, empty);
    }
}
