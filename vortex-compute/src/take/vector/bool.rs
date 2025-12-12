// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Take implementations for [`BoolVector`].
//!
//! This module includes an optimization for small boolean value arrays (typical of dictionary
//! encoding) that avoids element-wise indexing when possible.

use std::ops::BitAnd;
use std::ops::Not;

use vortex_buffer::BitBuffer;
use vortex_dtype::UnsignedPType;
use vortex_mask::Mask;
use vortex_vector::VectorOps;
use vortex_vector::bool::BoolVector;
use vortex_vector::primitive::PVector;

use crate::take::Take;

// TODO(connor): Figure out good numbers for these heuristics.

/// The maximum length of a values array for which we unconditionally apply the optimized take.
const OPTIMIZED_TAKE_MAX_VALUES_LEN: usize = 8;

/// The minimum ratio of `indices.len() / values.len()` for which we apply the optimized take.
const OPTIMIZED_TAKE_MIN_RATIO: usize = 2;

/// Returns whether to use the optimized take path based on heuristics.
fn should_use_optimized_take(values_len: usize, indices_len: usize) -> bool {
    values_len <= OPTIMIZED_TAKE_MAX_VALUES_LEN
        || indices_len >= OPTIMIZED_TAKE_MIN_RATIO * values_len
}

impl<I: UnsignedPType> Take<PVector<I>> for &BoolVector {
    type Output = BoolVector;

    fn take(self, indices: &PVector<I>) -> BoolVector {
        if indices.validity().all_true() {
            // No null indices, delegate to slice implementation.
            self.take(indices.elements().as_slice())
        } else {
            // Has null indices, need to propagate nulls.
            take_with_nullable_indices(self, indices)
        }
    }
}

impl<I: UnsignedPType> Take<[I]> for &BoolVector {
    type Output = BoolVector;

    fn take(self, indices: &[I]) -> BoolVector {
        if should_use_optimized_take(self.len(), indices.len()) {
            optimized_take(self, indices, || self.validity().take(indices))
        } else {
            default_take(self, indices)
        }
    }
}

/// Default element-wise take from a slice of indices.
pub fn default_take<I: UnsignedPType>(values: &BoolVector, indices: &[I]) -> BoolVector {
    let taken_bits = values.bits().take(indices);
    let taken_validity = values.validity().take(indices);

    debug_assert_eq!(taken_bits.len(), taken_validity.len());

    // SAFETY: Both components were taken with the same indices, so they have the same length.
    unsafe { BoolVector::new_unchecked(taken_bits, taken_validity) }
}

/// Take with nullable indices, propagating nulls from both values and indices.
fn take_with_nullable_indices<I: UnsignedPType>(
    values: &BoolVector,
    indices: &PVector<I>,
) -> BoolVector {
    let indices_slice = indices.elements().as_slice();
    let indices_validity = indices.validity();

    // Validity must combine value validity with index validity.
    let compute_validity = || {
        values
            .validity()
            .take(indices_slice)
            .bitand(indices_validity)
    };

    if should_use_optimized_take(values.len(), indices.len()) {
        optimized_take(values, indices_slice, compute_validity)
    } else {
        // We ignore index nullability when taking the bits since the validity mask handles nulls.
        let taken_bits = values.bits().take(indices_slice);
        let taken_validity = compute_validity();

        debug_assert_eq!(taken_bits.len(), taken_validity.len());

        // SAFETY: Both components were taken with the same indices, so they have the same length.
        unsafe { BoolVector::new_unchecked(taken_bits, taken_validity) }
    }
}

// TODO(connor): Use the generic `compare` implementation when that gets implemented.

/// Creates a [`BitBuffer`] where each bit is set iff the corresponding index equals `target`.
fn broadcast_index_comparison<I: UnsignedPType>(indices: &[I], target: usize) -> BitBuffer {
    BitBuffer::collect_bool(indices.len(), |i| {
        // SAFETY: `i` is in bounds since `collect_bool` iterates from 0..len.
        let index: usize = unsafe { indices.get_unchecked(i).as_() };
        index == target
    })
}

/// Optimized take for boolean vectors with small value arrays.
///
/// Since booleans can only be `true` or `false`, we can optimize these specific cases:
///
/// - All of the values are `true`, so create a [`BoolVector`] with `n` `true`s.
/// - All of the values are `false`, so create a [`BoolVector`] with `n` `false`s.
/// - There is a single `true` value, so compare indices against that index.
/// - There is a single `false` value, so compare indices against that index and negate.
/// - Otherwise, there are multiple `true`s and `false`s in the `values` vector and we must do a
///   normal `take` on it.
///
/// The `compute_validity` closure computes the output validity mask, allowing callers to handle
/// nullable vs non-nullable indices differently.
pub fn optimized_take<I: UnsignedPType>(
    values: &BoolVector,
    indices: &[I],
    compute_validity: impl FnOnce() -> Mask,
) -> BoolVector {
    let len = indices.len();
    let (trues, falses) = count_true_and_false_positions(values);

    let (taken_bits, taken_validity) = match (trues, falses) {
        // All values are null.
        (Count::None, Count::None) => (BitBuffer::new_unset(len), Mask::new_false(len)),

        // No true values exist, so all output bits are false.
        (Count::None, _) => (BitBuffer::new_unset(len), compute_validity()),

        // No false values exist, so all output bits are true.
        (_, Count::None) => (BitBuffer::new_set(len), compute_validity()),

        // Single true value: output bit is set iff index equals the true position.
        (Count::One(true_idx), _) => {
            let bits = broadcast_index_comparison(indices, true_idx);
            (bits, compute_validity())
        }

        // Single false value: output bit is set iff index does NOT equal the false position.
        (_, Count::One(false_idx)) => {
            let bits = broadcast_index_comparison(indices, false_idx).not();
            (bits, compute_validity())
        }

        // Multiple true and false values, so fall back to the default `take`.
        (Count::More, Count::More) => {
            let taken_bits = values.bits().take(indices);
            (taken_bits, compute_validity())
        }
    };

    debug_assert_eq!(taken_bits.len(), taken_validity.len());

    // SAFETY: Both components have length `len` (the length of `indices`).
    unsafe { BoolVector::new_unchecked(taken_bits, taken_validity) }
}

/// Represents the count of true or false values found in a boolean vector.
enum Count {
    /// No values of this kind were found.
    None,
    /// Exactly one value was found at the given index.
    One(usize),
    /// Two or more values were found.
    More,
}

/// Scans a boolean vector to determine how many true and false values exist.
///
/// Returns `(true_count, false_count)` where each is a [`Count`] indicating none, one (with
/// position), or more than one. Null values are skipped. The scan exits early once both counts
/// reach "more than one".
fn count_true_and_false_positions(values: &BoolVector) -> (Count, Count) {
    let bits = values.bits();
    let validity = values.validity();

    let mut first_true: Option<usize> = None;
    let mut found_second_true = false;
    let mut first_false: Option<usize> = None;
    let mut found_second_false = false;

    for idx in 0..values.len() {
        if !validity.value(idx) {
            continue;
        }

        if bits.value(idx) {
            if first_true.is_none() {
                first_true = Some(idx);
            } else {
                found_second_true = true;
            }
        } else if first_false.is_none() {
            first_false = Some(idx);
        } else {
            found_second_false = true;
        }

        if found_second_true && found_second_false {
            break;
        }
    }

    let true_count = match (first_true, found_second_true) {
        (None, _) => Count::None,
        (Some(idx), false) => Count::One(idx),
        (Some(_), true) => Count::More,
    };

    let false_count = match (first_false, found_second_false) {
        (None, _) => Count::None,
        (Some(idx), false) => Count::One(idx),
        (Some(_), true) => Count::More,
    };

    (true_count, false_count)
}
