// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use bitvec::prelude::Lsb0;
use bitvec::view::BitView;
use vortex_error::VortexExpect;

use crate::bit::get_bit;
use crate::{BitBuffer, BufferMut, ByteBuffer, ByteBufferMut, buffer_mut};

/// A mutable bitset buffer that allows random access to individual bits for set and get.
///
///
/// # Example
/// ```
/// use vortex_buffer::BitBufferMut;
///
/// let mut bools = BitBufferMut::new_unset(10);
/// bools.set_to(9, true);
/// for i in 0..9 {
///    assert!(!bools.value(i));
/// }
/// assert!(bools.value(9));
///
/// // Freeze into a new bools vector.
/// let bools = bools.freeze();
/// ```
///
/// See also: [`crate::BitBuffer`].
pub struct BitBufferMut {
    buffer: ByteBufferMut,
    len: usize,
}

impl BitBufferMut {
    /// Create new bit buffer from given byte buffer and logical bit length
    pub fn from_buffer(buffer: ByteBufferMut, len: usize) -> Self {
        assert!(
            len <= buffer.len() * 8,
            "Buffer len {} is too short for the given length {len}",
            buffer.len()
        );
        Self { buffer, len }
    }

    /// Create a new empty mutable bit buffer with requested capacity (in bits).
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: BufferMut::with_capacity(capacity.div_ceil(8)),
            len: 0,
        }
    }

    /// Create a new mutable buffer with requested `len` and all bits set to `true`.
    pub fn new_set(len: usize) -> Self {
        Self {
            buffer: buffer_mut![0xFF; len.div_ceil(8)],
            len,
        }
    }

    /// Create a new mutable buffer with requested `len` and all bits set to `false`.
    pub fn new_unset(len: usize) -> Self {
        Self {
            buffer: BufferMut::zeroed(len.div_ceil(8)),
            len,
        }
    }

    /// Create a new empty `BitBufferMut`.
    pub fn empty() -> Self {
        Self::with_capacity(0)
    }

    /// Get the current populated length of the buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if the buffer has length 0.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the value at the requested index.
    pub fn value(&self, index: usize) -> bool {
        get_bit(&self.buffer, index)
    }

    /// Get the bit capacity of the buffer.
    pub fn capacity(&self) -> usize {
        self.buffer.capacity() * 8
    }

    /// Reserve additional bit capacity for the buffer.
    pub fn reserve(&mut self, additional: usize) {
        let capacity = self.len + additional;
        if capacity > self.capacity() {
            // convert differential to bytes
            let additional = capacity.div_ceil(8) - self.buffer.len();
            self.buffer.reserve(additional);
        }
    }

    /// Set the bit at `index` to the given boolean value.
    ///
    /// This operation is checked so if `index` exceeds the buffer length, this will panic.
    pub fn set_to(&mut self, index: usize, value: bool) {
        if value {
            self.set(index);
        } else {
            self.unset(index);
        }
    }

    /// Set a position to `true`.
    ///
    /// This operation is checked so if `index` exceeds the buffer length, this will panic.
    pub fn set(&mut self, index: usize) {
        assert!(index < self.len, "index {index} exceeds len {}", self.len);

        // SAFETY: checked by assertion
        unsafe { self.set_unchecked(index) };
    }

    /// Set a position to `false`.
    ///
    /// This operation is checked so if `index` exceeds the buffer length, this will panic.
    pub fn unset(&mut self, index: usize) {
        assert!(index < self.len, "index {index} exceeds len {}", self.len);

        // SAFETY: checked by assertion
        unsafe { self.unset_unchecked(index) };
    }

    /// Set the bit at `index` to `true` without checking bounds.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index` does not exceed the largest bit index in the backing buffer.
    pub unsafe fn set_unchecked(&mut self, index: usize) {
        let word_index = index / 8;
        let bit_index = index % 8;
        // SAFETY: checked by caller
        unsafe {
            let word = self.buffer.as_mut_ptr().add(word_index);
            word.write(*word | 1 << bit_index);
        }
    }

    /// Unset the bit at `index` without checking bounds.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index` does not exceed the largest bit index in the backing buffer.
    pub unsafe fn unset_unchecked(&mut self, index: usize) {
        let word_index = index / 8;
        let bit_index = index % 8;

        // SAFETY: checked by caller
        unsafe {
            let word = self.buffer.as_mut_ptr().add(word_index);
            word.write(*word & !(1 << bit_index));
        }
    }

    /// Truncate the buffer to the given length.
    pub fn truncate(&mut self, len: usize) {
        if len > self.len {
            return;
        }

        let new_len_bytes = len.div_ceil(8);
        self.buffer.truncate(new_len_bytes);
        self.len = len;

        let remainder = self.len % 8;
        if remainder != 0 {
            let mask = (1u8 << remainder).wrapping_sub(1);
            *self.buffer.as_mut().last_mut().vortex_expect("non empty") &= mask;
        }
    }

    /// Append a new boolean into the bit buffer, incrementing the length.
    ///
    /// Panics if the buffer is full.
    pub fn append(&mut self, value: bool) {
        if value {
            self.append_true()
        } else {
            self.append_false()
        }
    }

    /// Append a new true value to the buffer.
    ///
    /// Panics if there is no remaining capacity.
    pub fn append_true(&mut self) {
        // TODO(ngates): this is surely pretty slow.
        if self.len % 8 == 0 {
            // Push a new word that starts with 1
            self.buffer.push(1u8);
        } else {
            // Push a 1 bit into the current word.
            let word = self.buffer.last_mut().vortex_expect("buffer is not empty");
            *word |= 1 << (self.len % 8);
        }

        self.len += 1;
    }

    /// Append a new false value to the buffer.
    ///
    /// Panics if there is no remaining capacity.
    pub fn append_false(&mut self) {
        if self.len % 8 == 0 {
            // push new word that starts with 0
            self.buffer.push(0u8);
        }

        self.len += 1;
    }
    /// Append several boolean values into the bit buffer. After this operation,
    /// the length will be incremented by `n`.
    ///
    /// Panics if the buffer does not have `n` slots left.
    pub fn append_n(&mut self, value: bool, n: usize) {
        match value {
            true => {
                let new_len = self.len + n;
                let new_len_bytes = new_len.div_ceil(8);
                let cur_remainder = self.len % 8;
                let new_remainder = new_len % 8;

                if cur_remainder != 0 {
                    // Pad cur_remainder high bits with 1s
                    *self
                        .buffer
                        .as_mut_slice()
                        .last_mut()
                        .vortex_expect("buffer is not empty") |= !((1 << cur_remainder) - 1);
                }

                // Push several full bytes.
                if new_len_bytes > self.buffer.len() {
                    // Push full bytes, except for the final byte.
                    self.buffer.push_n(0xFF, new_len_bytes - self.buffer.len());
                }

                // Patch zeros into remainder of last byte pushed
                if new_remainder > 0 {
                    // Set the new_remainder LSB to 1
                    *self
                        .buffer
                        .as_mut_slice()
                        .last_mut()
                        .vortex_expect("buffer is not empty") &= (1 << new_remainder) - 1;
                }
            }
            false => {
                let new_len = self.len + n;
                let new_len_bytes = new_len.div_ceil(8);

                // push new 0 bytes.
                if new_len_bytes > self.buffer.len() {
                    self.buffer.push_n(0, new_len_bytes - self.buffer.len());
                }
            }
        }

        self.len += n;
    }

    /// Append bits defined by range from values to this buffer
    pub fn append_packed_range(&mut self, range: Range<usize>, values: &ByteBuffer) {
        let bit_len = range.end - range.start;
        self.buffer.reserve(bit_len.div_ceil(8));
        // SAFETY: The copy below will populate the values
        unsafe { self.buffer.set_len((self.len + bit_len).div_ceil(8)) };

        let self_slice = self.buffer.as_mut_slice().view_bits_mut::<Lsb0>();
        let other_slice = values.as_slice().view_bits::<Lsb0>();

        let other_sliced = &other_slice[range.start..range.end];
        self_slice[self.len..][..bit_len].copy_from_bitslice(other_sliced);
        self.len += bit_len;
    }

    /// Append a [`BitBuffer`] to this [`BitBufferMut`]
    pub fn append_buffer(&mut self, buffer: &BitBuffer) {
        let buffer_range = buffer.offset()..buffer.offset() + buffer.len();
        self.append_packed_range(buffer_range, buffer.inner())
    }

    /// Freeze the buffer in its current state into an immutable `BoolBuffer`.
    pub fn freeze(self) -> BitBuffer {
        BitBuffer::new(self.buffer.freeze().into_byte_buffer(), self.len)
    }

    /// Get the underlying bytes as a slice
    pub fn as_slice(&self) -> &[u8] {
        self.buffer.as_slice()
    }

    /// Get the underlying bytes as a mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.buffer.as_mut_slice()
    }
}

impl Default for BitBufferMut {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

#[cfg(test)]
mod tests {
    use crate::bit::buf_mut::BitBufferMut;

    #[test]
    fn test_bits_mut() {
        let mut bools = BitBufferMut::new_unset(10);
        bools.set_to(0, true);
        bools.set_to(9, true);

        let bools = bools.freeze();
        assert!(bools.value(0));
        for i in 1..=8 {
            assert!(!bools.value(i));
        }
        assert!(bools.value(9));
    }

    #[test]
    fn test_append_n() {
        let mut bools = BitBufferMut::with_capacity(10);
        assert_eq!(bools.len(), 0);
        assert!(bools.is_empty());

        bools.append(true);
        bools.append_n(false, 8);
        bools.append_n(true, 1);

        let bools = bools.freeze();

        assert_eq!(bools.true_count(), 2);
        assert!(bools.value(0));
        assert!(bools.value(9));
    }
}
