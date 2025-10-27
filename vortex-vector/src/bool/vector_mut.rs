// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`BoolVectorMut`].

use vortex_buffer::BitBufferMut;
use vortex_error::{VortexExpect, VortexResult, vortex_ensure};
use vortex_mask::MaskMut;

use crate::{BoolVector, VectorMutOps, VectorOps};

// TODO(connor): Implement proper `IntoIterator` trait instead of relying on `to_vec`.

/// A mutable vector of boolean values.
///
/// `BoolVectorMut` is the primary way to construct boolean vectors. It provides efficient methods
/// for building vectors incrementally before converting them to an immutable [`BoolVector`] using
/// the [`freeze`](crate::VectorMutOps::freeze) method.
///
/// # Examples
///
/// ## Extending and appending
///
/// ```
/// use vortex_vector::{BoolVectorMut, VectorMutOps};
///
/// let mut vec1 = BoolVectorMut::from_iter([true, false].map(Some));
/// let vec2 = BoolVectorMut::from_iter([true, true].map(Some)).freeze();
///
/// // Extend from another vector.
/// vec1.extend_from_vector(&vec2);
/// assert_eq!(vec1.len(), 4);
///
/// // Append null values.
/// vec1.append_nulls(2);
/// assert_eq!(vec1.len(), 6);
/// ```
///
/// ## Splitting and unsplitting
///
/// ```
/// use vortex_vector::{BoolVectorMut, VectorMutOps};
///
/// let mut vec = BoolVectorMut::from_iter([true, false, true, false, true].map(Some));
///
/// // Split the vector at index 3.
/// let mut second_half = vec.split_off(3);
/// assert_eq!(vec.len(), 3);
/// assert_eq!(second_half.len(), 2);
///
/// // Rejoin the vectors.
/// vec.unsplit(second_half);
/// assert_eq!(vec.len(), 5);
/// ```
///
/// ## Converting to immutable
///
/// ```
/// use vortex_vector::{BoolVectorMut, VectorMutOps, VectorOps};
///
/// let mut vec = BoolVectorMut::from_iter([true, false, true].map(Some));
///
/// // Freeze into an immutable vector.
/// let immutable = vec.freeze();
/// assert_eq!(immutable.len(), 3);
/// ```
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
        Self::try_new(bits, validity)
            .vortex_expect("`BoolVector` validity mask must have the same length as bits")
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
        debug_assert_eq!(
            bits.len(),
            validity.len(),
            "`BoolVector` validity mask must have the same length as bits"
        );

        Self { bits, validity }
    }

    /// Creates a new mutable boolean vector with the given `capacity`.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bits: BitBufferMut::with_capacity(capacity),
            validity: MaskMut::with_capacity(capacity),
        }
    }

    /// Returns the parts of the mutable vector.
    pub fn into_parts(self) -> (BitBufferMut, MaskMut) {
        (self.bits, self.validity)
    }

    /// Converts the vector to a `Vec<Option<bool>>`.
    ///
    /// This method borrows the vector and creates a new `Vec` with `Some(value)` for non-null
    /// elements and `None` for null elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use vortex_vector::BoolVectorMut;
    ///
    /// let vec = BoolVectorMut::from_iter([Some(true), None, Some(false)]);
    /// let values = vec.to_vec();
    /// assert_eq!(values, vec![Some(true), None, Some(false)]);
    /// ```
    pub fn to_vec(&self) -> Vec<Option<bool>> {
        (0..self.len())
            .map(|i| self.validity.value(i).then(|| self.bits.value(i)))
            .collect()
    }

    /// Attempts to convert the vector to a `Vec<bool>` containing only non-null values.
    ///
    /// Returns `None` if the vector contains any null values. Otherwise, returns `Some(Vec<bool>)`
    /// with all the non-null values.
    ///
    /// # Examples
    ///
    /// ```
    /// use vortex_vector::BoolVectorMut;
    ///
    /// // All non-null values.
    /// let vec = BoolVectorMut::from_iter([true, false, true]);
    /// let values = vec.to_nonnull_vec();
    /// assert_eq!(values, Some(vec![true, false, true]));
    ///
    /// // Contains null values.
    /// let vec = BoolVectorMut::from_iter([Some(true), None, Some(false)]);
    /// let values = vec.to_nonnull_vec();
    /// assert_eq!(values, None);
    /// ```
    pub fn to_nonnull_vec(&self) -> Option<Vec<bool>> {
        let validity_frozen = self.validity.clone().freeze();
        validity_frozen
            .all_true()
            .then(|| (0..self.len()).map(|i| self.bits.value(i)).collect())
    }
}

impl VectorMutOps for BoolVectorMut {
    type Immutable = BoolVector;

    fn len(&self) -> usize {
        debug_assert!(self.validity.len() == self.bits.len());

        self.bits.len()
    }

    fn capacity(&self) -> usize {
        self.bits.capacity()
    }

    fn reserve(&mut self, additional: usize) {
        self.bits.reserve(additional);
        self.validity.reserve(additional);
    }

    fn extend_from_vector(&mut self, other: &BoolVector) {
        self.bits.append_buffer(&other.bits);
        self.validity.append_mask(other.validity());
    }

    fn append_nulls(&mut self, n: usize) {
        self.bits.append_n(false, n);
        self.validity.append_n(false, n);
    }

    fn freeze(self) -> Self::Immutable {
        BoolVector {
            bits: self.bits.freeze(),
            validity: self.validity.freeze(),
        }
    }

    fn split_off(&mut self, at: usize) -> Self {
        BoolVectorMut {
            bits: self.bits.split_off(at),
            validity: self.validity.split_off(at),
        }
    }

    fn unsplit(&mut self, other: Self) {
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
    fn test_to_vec_variants() {
        // Test to_vec with mixed null/non-null values.
        let mixed = BoolVectorMut::from_iter([Some(true), None, Some(false), None, Some(true)]);
        assert_eq!(
            mixed.to_vec(),
            vec![Some(true), None, Some(false), None, Some(true)]
        );
        assert_eq!(mixed.len(), 5); // Vector still usable after to_vec.

        // Test to_vec with all non-null.
        let all_valid = BoolVectorMut::from_iter([true, false, true]);
        assert_eq!(
            all_valid.to_vec(),
            vec![Some(true), Some(false), Some(true)]
        );

        // Test to_vec with all null.
        let all_null = BoolVectorMut::from_iter([None, None, None]);
        assert_eq!(all_null.to_vec(), vec![None, None, None]);

        // Test to_vec with empty vector.
        let empty = BoolVectorMut::with_capacity(0);
        assert_eq!(empty.to_vec(), Vec::<Option<bool>>::new());
    }

    #[test]
    fn test_to_nonnull_vec_variants() {
        // Test with all non-null values - should return Some(vec).
        let all_valid = BoolVectorMut::from_iter([true, false, true, false, true]);
        assert_eq!(
            all_valid.to_nonnull_vec(),
            Some(vec![true, false, true, false, true])
        );
        assert_eq!(all_valid.len(), 5); // Vector still usable.

        // Test with mixed values - should return None.
        let mixed = BoolVectorMut::from_iter([Some(true), None, Some(false)]);
        assert_eq!(mixed.to_nonnull_vec(), None);

        // Test with all nulls - should return None.
        let all_null = BoolVectorMut::from_iter([None, None, None]);
        assert_eq!(all_null.to_nonnull_vec(), None);

        // Test empty vector - should return Some(empty vec).
        let empty = BoolVectorMut::with_capacity(0);
        assert_eq!(empty.to_nonnull_vec(), Some(Vec::<bool>::new()));
    }
}
