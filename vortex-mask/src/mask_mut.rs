// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::BitBufferMut;
use vortex_error::vortex_panic;

use crate::Mask;

/// A mutable mask, used for lazily allocating the bit buffer as required.
#[derive(Debug, Clone)]
pub struct MaskMut(Inner);

impl Default for MaskMut {
    fn default() -> Self {
        Self::empty()
    }
}

#[derive(Debug, Clone)]
enum Inner {
    /// Initially, the mask is empty but may have some capacity.
    Empty { capacity: usize },
    /// When the first value is pushed, the mask becomes constant.
    Constant {
        value: bool,
        len: usize,
        capacity: usize,
    },
    /// When the first non-constant value is written, we allocate the bit buffer and switch
    /// into the builder state.
    Builder(BitBufferMut),
}

impl MaskMut {
    /// Creates a new empty mask.
    pub fn empty() -> Self {
        Self::with_capacity(0)
    }

    /// Creates a new empty mask with the default capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self(Inner::Empty { capacity })
    }

    /// Creates a new mask with the specified capacity.
    pub fn new(len: usize, value: bool) -> Self {
        Self(Inner::Constant {
            value,
            len,
            capacity: len,
        })
    }

    /// Creates a new mask with all values set to `true`.
    pub fn new_true(len: usize) -> Self {
        Self(Inner::Constant {
            value: true,
            len,
            capacity: len,
        })
    }

    /// Creates a new mask with all values set to `false`.
    pub fn new_false(len: usize) -> Self {
        Self(Inner::Constant {
            value: false,
            len,
            capacity: len,
        })
    }

    /// Creates a new mask from an existing bit buffer.
    pub fn from_buffer(bit_buffer: BitBufferMut) -> Self {
        Self(Inner::Builder(bit_buffer))
    }

    /// Returns the boolean value at a given index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn value(&self, index: usize) -> bool {
        match &self.0 {
            Inner::Empty { .. } => {
                vortex_panic!("index out of bounds: the length is 0 but the index is {index}")
            }
            Inner::Constant { value, len, .. } => {
                assert!(
                    index < *len,
                    "index out of bounds: the length is {} but the index is {index}",
                    *len
                );

                *value
            }
            Inner::Builder(bit_buffer) => bit_buffer.value(index),
        }
    }

    /// Reserve capacity for at least `additional` more values to be appended.
    pub fn reserve(&mut self, additional: usize) {
        match &mut self.0 {
            Inner::Empty { capacity } => {
                *capacity += additional;
            }
            Inner::Constant { capacity, .. } => {
                *capacity += additional;
            }
            Inner::Builder(bits) => {
                bits.reserve(additional);
            }
        }
    }

    /// Set the length of the mask.
    ///
    /// # Safety
    ///
    /// - `new_len` must be less than or equal to [`capacity()`].
    /// - The elements at `old_len..new_len` must be initialized.
    ///
    /// [`capacity()`]: Self::capacity
    pub unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len < self.capacity());
        match &mut self.0 {
            Inner::Empty { capacity, .. } => {
                self.0 = Inner::Constant {
                    value: false, // Pick any value
                    len: new_len,
                    capacity: *capacity,
                }
            }
            Inner::Constant { len, .. } => {
                *len = new_len;
            }
            Inner::Builder(bits) => {
                unsafe { bits.set_len(new_len) };
            }
        }
    }

    /// Returns the capacity of the mask.
    pub fn capacity(&self) -> usize {
        match &self.0 {
            Inner::Empty { capacity } => *capacity,
            Inner::Constant { capacity, .. } => *capacity,
            Inner::Builder(bits) => bits.capacity(),
        }
    }

    /// Clears the mask.
    ///
    /// Note that this method has no effect on the allocated capacity of the mask.
    pub fn clear(&mut self) {
        match &mut self.0 {
            Inner::Empty { .. } => {}
            Inner::Constant { capacity, .. } => {
                self.0 = Inner::Empty {
                    capacity: *capacity,
                }
            }
            Inner::Builder(bit_buffer) => bit_buffer.clear(),
        };
    }

    /// Shortens the mask, keeping the first `len` bits.
    ///
    /// If `len` is greater or equal to the vector’s current length, this has no effect.
    ///
    /// Note that this method has no effect on the allocated capacity of the mask.
    pub fn truncate(&mut self, len: usize) {
        let truncated_len = len;
        if truncated_len > self.len() {
            return;
        }

        match &mut self.0 {
            Inner::Empty { .. } => {}
            Inner::Constant { len, .. } => *len = truncated_len.min(*len),
            Inner::Builder(bit_buffer) => bit_buffer.truncate(truncated_len),
        };
    }

    /// Append n values to the mask.
    pub fn append_n(&mut self, new_value: bool, n: usize) {
        match &mut self.0 {
            Inner::Empty { capacity } => {
                self.0 = Inner::Constant {
                    value: new_value,
                    len: n,
                    capacity: (*capacity).max(n),
                }
            }
            Inner::Constant {
                value,
                len,
                capacity,
            } => {
                if *value == new_value {
                    // Same value, just increase length.
                    self.0 = Inner::Constant {
                        value: *value,
                        len: *len + n,
                        capacity: (*capacity).max(*len + n),
                    }
                } else {
                    // Different value, need to allocate the bit buffer.
                    // Note: materialize() already appends the existing constant values
                    let bits = self.materialize();
                    bits.append_n(new_value, n);
                }
            }
            Inner::Builder(bits) => {
                bits.append_n(new_value, n);
            }
        }
    }

    /// Append a [`Mask`] to this mutable mask.
    pub fn append_mask(&mut self, other: &Mask) {
        match other {
            Mask::AllTrue(len) => self.append_n(true, *len),
            Mask::AllFalse(len) => self.append_n(false, *len),
            Mask::Values(values) => {
                let bitbuffer = values.buffer.clone();
                self.materialize().append_buffer(&bitbuffer);
            }
        }
    }

    /// Ensures that the internal bit buffer is allocated and returns a mutable reference to it.
    fn materialize(&mut self) -> &mut BitBufferMut {
        let needs_materialization = !matches!(self.0, Inner::Builder(_));

        if needs_materialization {
            let new_builder = match &self.0 {
                Inner::Empty { capacity } => BitBufferMut::with_capacity(*capacity),
                Inner::Constant {
                    value,
                    len,
                    capacity,
                } => {
                    let required_capacity = (*capacity).max(*len);
                    let mut bits = BitBufferMut::with_capacity(required_capacity);
                    bits.append_n(*value, *len);
                    bits
                }
                Inner::Builder(_) => unreachable!(),
            };
            self.0 = Inner::Builder(new_builder);
        }

        match &mut self.0 {
            Inner::Builder(bits) => bits,
            _ => unreachable!(),
        }
    }

    /// Split-off the mask at the given index, returning a new mask with the
    /// values from `at` to the end, and leaving `self` with the values from
    /// the start to `at`.
    pub fn split_off(&mut self, at: usize) -> Self {
        assert!(at <= self.capacity(), "split_off index out of bounds");
        match &mut self.0 {
            Inner::Empty { capacity } => {
                let new_capacity = *capacity - at;
                *capacity = at;
                Self(Inner::Empty {
                    capacity: new_capacity,
                })
            }
            Inner::Constant {
                value,
                len,
                capacity,
            } => {
                // Adjust the lengths, given that length may be < at
                let new_len = len.saturating_sub(at);
                let new_capacity = *capacity - at;
                *len = (*len).min(at);
                *capacity = at;

                Self(Inner::Constant {
                    value: *value,
                    len: new_len,
                    capacity: new_capacity,
                })
            }
            Inner::Builder(bits) => {
                let new_bits = bits.split_off(at);
                Self(Inner::Builder(new_bits))
            }
        }
    }

    /// Absorb another mask into this one, appending its values.
    pub fn unsplit(&mut self, other: Self) {
        match other.0 {
            Inner::Empty { .. } => {
                // No work to do
            }
            Inner::Constant { value, len, .. } => {
                self.append_n(value, len);
            }
            Inner::Builder(bits) => {
                self.materialize().unsplit(bits);
            }
        }
    }

    /// Freezes the mutable mask into an immutable one.
    pub fn freeze(self) -> Mask {
        match self.0 {
            Inner::Empty { .. } => Mask::new_true(0),
            Inner::Constant { value, len, .. } => {
                if value {
                    Mask::new_true(len)
                } else {
                    Mask::new_false(len)
                }
            }
            Inner::Builder(bits) => Mask::from_buffer(bits.freeze()),
        }
    }

    /// Returns the logical length of the mask.
    pub fn len(&self) -> usize {
        match &self.0 {
            Inner::Empty { .. } => 0,
            Inner::Constant { len, .. } => *len,
            Inner::Builder(bits) => bits.len(),
        }
    }

    /// Returns true if the mask is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns true if all values in the mask are true.
    pub fn all_true(&self) -> bool {
        match &self.0 {
            Inner::Empty { .. } => true,
            Inner::Constant { value, .. } => *value,
            Inner::Builder(bits) => bits.true_count() == bits.len(),
        }
    }

    /// Returns true if all values in the mask are false.
    pub fn all_false(&self) -> bool {
        match &self.0 {
            Inner::Empty { .. } => true,
            Inner::Constant { value, .. } => !*value,
            Inner::Builder(bits) => !bits.is_empty() && bits.true_count() == 0,
        }
    }

    /// Returns the internal bit buffer if it exists.
    pub fn as_bit_buffer_mut(&mut self) -> Option<&mut BitBufferMut> {
        match &mut self.0 {
            Inner::Builder(bits) => Some(bits),
            _ => None,
        }
    }

    /// Set the value at the given index to true.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn set(&mut self, index: usize) {
        self.set_to(index, true);
    }

    /// Set the value at the given index to false.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn unset(&mut self, index: usize) {
        self.set_to(index, false);
    }

    /// Set the value at the given index to the specified boolean value.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn set_to(&mut self, index: usize, value: bool) {
        match &mut self.0 {
            Inner::Empty { .. } => {
                vortex_panic!("index out of bounds: the length is 0 but the index is {index}")
            }
            Inner::Constant {
                value: current_value,
                len,
                ..
            } => {
                assert!(
                    index < *len,
                    "index out of bounds: the length is {} but the index is {index}",
                    *len
                );

                if *current_value != value {
                    // Need to materialize the buffer since we're changing from constant.
                    self.materialize().set_to(index, value);
                }
                // If the value is the same as the constant, no action needed.
            }
            Inner::Builder(bit_buffer) => {
                bit_buffer.set_to(index, value);
            }
        }
    }

    /// Set the value at the given index to true without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index < self.len()`.
    pub unsafe fn set_unchecked(&mut self, index: usize) {
        unsafe { self.set_to_unchecked(index, true) }
    }

    /// Set the value at the given index to false without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index < self.len()`.
    pub unsafe fn unset_unchecked(&mut self, index: usize) {
        unsafe { self.set_to_unchecked(index, false) }
    }

    /// Set the value at the given index to the specified boolean value without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index < self.len()`.
    pub unsafe fn set_to_unchecked(&mut self, index: usize, value: bool) {
        unsafe {
            match &mut self.0 {
                Inner::Empty { .. } => {
                    // In debug mode, we still want to catch this error.
                    debug_assert!(false, "cannot set value in empty mask");
                }
                Inner::Constant {
                    value: current_value,
                    len,
                    ..
                } => {
                    debug_assert!(
                        index < *len,
                        "index out of bounds: the length is {} but the index is {index}",
                        *len
                    );

                    if *current_value != value {
                        // Need to materialize the buffer since we're changing from constant.
                        self.materialize().set_to_unchecked(index, value);
                    }
                    // If the value is the same as the constant, no action needed.
                }
                Inner::Builder(bit_buffer) => {
                    bit_buffer.set_to_unchecked(index, value);
                }
            }
        }
    }
}

impl Mask {
    /// Attempts to convert an immutable mask into a mutable one, returning an error of `Self` if
    /// the underlying [`BitBuffer`](crate::BitBuffer) data if there are any other references.
    pub fn try_into_mut(self) -> Result<MaskMut, Self> {
        match self {
            Mask::AllTrue(len) => Ok(MaskMut::new_true(len)),
            Mask::AllFalse(len) => Ok(MaskMut::new_false(len)),
            Mask::Values(values) => {
                // We need to check for uniqueness twice, first for the `Arc` with `try_unwrap`,
                // then for the internal `BitBuffer` with `try_into_mut`.
                let owned_values = Arc::try_unwrap(values).map_err(Mask::Values)?;
                let bit_buffer = owned_values.into_buffer();
                let mut_buffer = bit_buffer.try_into_mut().map_err(Mask::from_buffer)?;

                Ok(MaskMut(Inner::Builder(mut_buffer)))
            }
        }
    }

    /// Convert an immutable mask into a mutable one, cloning the underlying
    /// [`BitBuffer`](crate::BitBuffer) data if there are any other references.
    pub fn into_mut(self) -> MaskMut {
        match self {
            Mask::AllTrue(len) => MaskMut::new_true(len),
            Mask::AllFalse(len) => MaskMut::new_false(len),
            Mask::Values(values) => {
                let bit_buffer_mut = match Arc::try_unwrap(values) {
                    Ok(mask_values) => mask_values
                        .into_buffer()
                        .try_into_mut()
                        .unwrap_or_else(|bb| BitBufferMut::copy_from(&bb)),
                    Err(arc_mask_values) => {
                        let bit_buffer = arc_mask_values.bit_buffer();
                        BitBufferMut::copy_from(bit_buffer)
                    }
                };

                MaskMut(Inner::Builder(bit_buffer_mut))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_off_empty() {
        let mut mask = MaskMut::with_capacity(10);
        assert_eq!(mask.len(), 0);

        let other = mask.split_off(0);
        assert_eq!(mask.len(), 0);
        assert_eq!(other.len(), 0);
    }

    #[test]
    fn test_split_off_constant_true_at_zero() {
        let mut mask = MaskMut::new_true(10);
        let other = mask.split_off(0);

        assert_eq!(mask.len(), 0);
        assert_eq!(other.len(), 10);

        let frozen = other.freeze();
        assert_eq!(frozen.true_count(), 10);
    }

    #[test]
    fn test_split_off_constant_true_at_end() {
        let mut mask = MaskMut::new_true(10);
        let other = mask.split_off(10);

        assert_eq!(mask.len(), 10);
        assert_eq!(other.len(), 0);

        let frozen = mask.freeze();
        assert_eq!(frozen.true_count(), 10);
    }

    #[test]
    fn test_split_off_constant_true_in_middle() {
        let mut mask = MaskMut::new_true(10);
        let other = mask.split_off(6);

        assert_eq!(mask.len(), 6);
        assert_eq!(other.len(), 4);

        let frozen_first = mask.freeze();
        assert_eq!(frozen_first.true_count(), 6);

        let frozen_second = other.freeze();
        assert_eq!(frozen_second.true_count(), 4);
    }

    #[test]
    fn test_split_off_constant_false() {
        let mut mask = MaskMut::new_false(20);
        let other = mask.split_off(12);

        assert_eq!(mask.len(), 12);
        assert_eq!(other.len(), 8);

        let frozen_first = mask.freeze();
        assert_eq!(frozen_first.true_count(), 0);

        let frozen_second = other.freeze();
        assert_eq!(frozen_second.true_count(), 0);
    }

    // Note: Tests using BitBuffer operations are marked as ignored under miri
    // because bitvec uses raw pointer operations that miri cannot verify.
    #[test]
    fn test_split_off_builder_at_byte_boundary() {
        let mut mask = MaskMut::with_capacity(16);
        // Create a pattern: 8 true, 8 false
        mask.append_n(true, 8);
        mask.append_n(false, 8);

        let mask_ptr = match &mask.0 {
            Inner::Builder(bits) => bits.as_slice().as_ptr(),
            _ => unreachable!(),
        };

        let other = mask.split_off(8);

        assert_eq!(mask.len(), 8);
        assert_eq!(other.len(), 8);

        // Ensure the unsplit was zero-copy.
        mask.unsplit(other);
        let new_mask_ptr = match &mask.0 {
            Inner::Builder(bits) => bits.as_slice().as_ptr(),
            _ => unreachable!(),
        };
        assert_eq!(mask_ptr, new_mask_ptr);
    }

    #[test]
    fn test_split_off_builder_not_byte_aligned() {
        let mut mask = MaskMut::with_capacity(20);
        // Create a pattern: 10 true, 10 false
        mask.append_n(true, 10);
        mask.append_n(false, 10);

        let other = mask.split_off(10);

        assert_eq!(mask.len(), 10);
        assert_eq!(other.len(), 10);

        let frozen_first = mask.freeze();
        assert_eq!(frozen_first.true_count(), 10);

        let frozen_second = other.freeze();
        assert_eq!(frozen_second.true_count(), 0);
    }

    #[test]
    fn test_split_off_builder_mixed_pattern() {
        let mut mask = MaskMut::with_capacity(15);
        // Create pattern: TFTFTFTFTFTFTFT (alternating)
        for i in 0..15 {
            mask.append_n(i % 2 == 0, 1);
        }

        let other = mask.split_off(7);

        assert_eq!(mask.len(), 7);
        assert_eq!(other.len(), 8);

        let frozen_first = mask.freeze();
        assert_eq!(frozen_first.true_count(), 4); // positions 0,2,4,6

        let frozen_second = other.freeze();
        assert_eq!(frozen_second.true_count(), 4); // positions 7,9,11,13 => 0,2,4,6 in split
    }

    #[test]
    fn test_unsplit_empty_with_empty() {
        let mut mask = MaskMut::with_capacity(10);
        let other = MaskMut::with_capacity(10);

        mask.unsplit(other);
        assert_eq!(mask.len(), 0);
    }

    #[test]
    fn test_unsplit_empty_with_constant() {
        let mut mask = MaskMut::with_capacity(10);
        let other = MaskMut::new_true(5);

        mask.unsplit(other);
        assert_eq!(mask.len(), 5);

        let frozen = mask.freeze();
        assert_eq!(frozen.true_count(), 5);
    }

    #[test]
    fn test_unsplit_constant_with_constant_same() {
        let mut mask = MaskMut::new_true(5);
        let other = MaskMut::new_true(5);

        mask.unsplit(other);
        assert_eq!(mask.len(), 10);

        let frozen = mask.freeze();
        assert_eq!(frozen.true_count(), 10);
    }

    #[test]
    fn test_unsplit_constant_with_constant_different() {
        let mut mask = MaskMut::new_true(5);
        let other = MaskMut::new_false(5);

        mask.unsplit(other);
        assert_eq!(mask.len(), 10);

        let frozen = mask.freeze();
        assert_eq!(frozen.true_count(), 5);
    }

    #[test]
    fn test_unsplit_constant_with_builder() {
        let mut mask = MaskMut::new_true(5);

        let mut other = MaskMut::with_capacity(10);
        other.append_n(true, 3);
        other.append_n(false, 2);

        mask.unsplit(other);
        assert_eq!(mask.len(), 10);

        let frozen = mask.freeze();
        assert_eq!(frozen.true_count(), 8); // 5 from first + 3 from second
    }

    #[test]
    fn test_unsplit_builder_with_constant() {
        let mut mask = MaskMut::with_capacity(10);
        mask.append_n(true, 3);
        mask.append_n(false, 2);

        let other = MaskMut::new_true(5);

        mask.unsplit(other);
        assert_eq!(mask.len(), 10);

        let frozen = mask.freeze();
        assert_eq!(frozen.true_count(), 8); // 3 from first + 5 from second
    }

    #[test]
    fn test_unsplit_builder_with_builder() {
        let mut mask = MaskMut::with_capacity(10);
        mask.append_n(true, 3);
        mask.append_n(false, 2);

        let mut other = MaskMut::with_capacity(10);
        other.append_n(false, 3);
        other.append_n(true, 2);

        mask.unsplit(other);
        assert_eq!(mask.len(), 10);

        let frozen = mask.freeze();
        assert_eq!(frozen.true_count(), 5); // 3 from first + 2 from second
    }

    #[test]
    fn test_round_trip_split_unsplit() {
        let mut original = MaskMut::with_capacity(20);
        // Pattern: 10 true, 10 false
        original.append_n(true, 10);
        original.append_n(false, 10);

        let original_frozen = original.freeze();
        let original_true_count = original_frozen.true_count();

        // Convert back to mutable for split
        let mut mask = original_frozen.try_into_mut().unwrap();

        // Split at 10
        let other = mask.split_off(10);

        // Unsplit back together
        mask.unsplit(other);

        assert_eq!(mask.len(), 20);
        let frozen = mask.freeze();
        assert_eq!(frozen.true_count(), original_true_count);
    }

    #[test]
    #[should_panic(expected = "split_off index out of bounds")]
    fn test_split_off_out_of_bounds() {
        let mut mask = MaskMut::new_true(10);
        mask.split_off(11);
    }

    #[test]
    fn test_split_off_builder_at_bit_1() {
        let mut mask = MaskMut::with_capacity(16);
        mask.append_n(true, 16);

        let other = mask.split_off(1);

        assert_eq!(mask.len(), 1);
        assert_eq!(other.len(), 15);

        let frozen_first = mask.freeze();
        assert_eq!(frozen_first.true_count(), 1);

        let frozen_second = other.freeze();
        assert_eq!(frozen_second.true_count(), 15);
    }

    #[test]
    fn test_multiple_split_unsplit() {
        let mut mask = MaskMut::new_true(30);

        // Split into 3 parts
        let third = mask.split_off(20); // 20-30
        let second = mask.split_off(10); // 10-20
        // first is 0-10

        assert_eq!(mask.len(), 10);
        assert_eq!(second.len(), 10);
        assert_eq!(third.len(), 10);

        // Recombine in order
        mask.unsplit(second);
        mask.unsplit(third);

        assert_eq!(mask.len(), 30);
        let frozen = mask.freeze();
        assert_eq!(frozen.true_count(), 30);
    }

    #[test]
    fn test_try_into_mut_all_variants() {
        // Test AllTrue and AllFalse variants.
        let mask_true = Mask::new_true(100);
        let mut_mask_true = mask_true.try_into_mut().unwrap();
        assert_eq!(mut_mask_true.len(), 100);
        assert_eq!(mut_mask_true.freeze().true_count(), 100);

        let mask_false = Mask::new_false(50);
        let mut_mask_false = mask_false.try_into_mut().unwrap();
        assert_eq!(mut_mask_false.len(), 50);
        assert_eq!(mut_mask_false.freeze().true_count(), 0);
    }

    #[test]
    fn test_try_into_mut_with_references() {
        // Create a MaskValues variant.
        let mut mask_mut = MaskMut::with_capacity(10);
        mask_mut.append_n(true, 5);
        mask_mut.append_n(false, 5);
        let mask = mask_mut.freeze();

        // Should succeed with unique reference (no clones).
        let mask2 = {
            let mut mask_mut2 = MaskMut::with_capacity(10);
            mask_mut2.append_n(true, 5);
            mask_mut2.append_n(false, 5);
            mask_mut2.freeze()
        };
        let result = mask2.try_into_mut();
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 10);

        // Should fail with shared references.
        let _cloned = mask.clone();
        let result = mask.try_into_mut();
        assert!(result.is_err());
        if let Err(returned_mask) = result {
            assert_eq!(returned_mask.len(), 10);
            assert_eq!(returned_mask.true_count(), 5);
        }
    }

    #[test]
    fn test_try_into_mut_round_trip() {
        // Test freeze -> try_into_mut -> modify -> freeze cycle.
        let mut original = MaskMut::with_capacity(20);
        original.append_n(true, 10);
        original.append_n(false, 10);

        let frozen = original.freeze();
        assert_eq!(frozen.true_count(), 10);

        let mut mut_mask = frozen.try_into_mut().unwrap();
        mut_mask.append_n(true, 5);
        assert_eq!(mut_mask.len(), 25);

        let frozen_again = mut_mask.freeze();
        assert_eq!(frozen_again.true_count(), 15);
    }
}
