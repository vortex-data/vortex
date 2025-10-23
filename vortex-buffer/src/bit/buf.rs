// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::{BitAnd, BitOr, BitXor, Not, Range};

use crate::bit::ops::{bitwise_binary_op, bitwise_unary_op};
use crate::bit::{
    BitChunks, BitIndexIterator, BitIterator, BitSliceIterator, UnalignedBitChunk,
    get_bit_unchecked,
};
use crate::{Alignment, BitBufferMut, Buffer, BufferMut, ByteBuffer, buffer};

/// An immutable bitset stored as a packed byte buffer.
#[derive(Debug, Clone, Eq)]
pub struct BitBuffer {
    buffer: ByteBuffer,
    offset: usize,
    len: usize,
}

impl PartialEq for BitBuffer {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }

        self.chunks()
            .iter_padded()
            .zip(other.chunks().iter_padded())
            .all(|(a, b)| a == b)
    }
}

impl BitBuffer {
    /// Create a new `BoolBuffer` backed by a [`ByteBuffer`] with `len` bits in view.
    ///
    /// Panics if the buffer is not large enough to hold `len` bits.
    pub fn new(buffer: ByteBuffer, len: usize) -> Self {
        assert!(
            buffer.len() * 8 >= len,
            "provided ByteBuffer not large enough to back BoolBuffer with len {len}"
        );

        // BitBuffers make no assumptions on byte alignment, so we strip any alignment.
        let buffer = buffer.aligned(Alignment::none());

        Self {
            buffer,
            len,
            offset: 0,
        }
    }

    /// Create a new `BoolBuffer` backed by a [`ByteBuffer`] with `len` bits in view, starting at
    /// the given `offset` (in bits).
    ///
    /// Panics if the buffer is not large enough to hold `len` bits after the offset.
    pub fn new_with_offset(buffer: ByteBuffer, len: usize, offset: usize) -> Self {
        assert!(
            len.saturating_add(offset) <= buffer.len().saturating_mul(8),
            "provided ByteBuffer (len={}) not large enough to back BoolBuffer with offset {offset} len {len}",
            buffer.len()
        );

        // BitBuffers make no assumptions on byte alignment, so we strip any alignment.
        let buffer = buffer.aligned(Alignment::none());

        Self {
            buffer,
            offset,
            len,
        }
    }

    /// Create a new `BoolBuffer` of length `len` where all bits are set (true).
    pub fn new_set(len: usize) -> Self {
        let words = len.div_ceil(8);
        let buffer = buffer![0xFF; words];

        Self {
            buffer,
            len,
            offset: 0,
        }
    }

    /// Create a new `BoolBuffer` of length `len` where all bits are unset (false).
    pub fn new_unset(len: usize) -> Self {
        let words = len.div_ceil(8);
        let buffer = Buffer::zeroed(words);

        Self {
            buffer,
            len,
            offset: 0,
        }
    }

    /// Create a new empty `BitBuffer`.
    pub fn empty() -> Self {
        Self::new_set(0)
    }

    /// Create a new `BitBuffer` of length `len` where all bits are set to `value`.
    pub fn full(value: bool, len: usize) -> Self {
        if value {
            Self::new_set(len)
        } else {
            Self::new_unset(len)
        }
    }

    /// Invokes `f` with indexes `0..len` collecting the boolean results into a new `BitBuffer`
    pub fn collect_bool<F: FnMut(usize) -> bool>(len: usize, mut f: F) -> Self {
        let mut buffer = BufferMut::with_capacity(len.div_ceil(64) * 8);

        let chunks = len / 64;
        let remainder = len % 64;
        for chunk in 0..chunks {
            let mut packed = 0;
            for bit_idx in 0..64 {
                let i = bit_idx + chunk * 64;
                packed |= (f(i) as u64) << bit_idx;
            }

            // SAFETY: Already allocated sufficient capacity
            unsafe { buffer.push_unchecked(packed) }
        }

        if remainder != 0 {
            let mut packed = 0;
            for bit_idx in 0..remainder {
                let i = bit_idx + chunks * 64;
                packed |= (f(i) as u64) << bit_idx;
            }

            // SAFETY: Already allocated sufficient capacity
            unsafe { buffer.push_unchecked(packed) }
        }

        buffer.truncate(len.div_ceil(8));

        Self::new(buffer.freeze().into_byte_buffer(), len)
    }

    /// Get the logical length of this `BoolBuffer`.
    ///
    /// This may differ from the physical length of the backing buffer, for example if it was
    /// created using the `new_with_offset` constructor, or if it was sliced.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the `BoolBuffer` is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Offset of the start of the buffer in bits.
    #[inline(always)]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get a reference to the underlying buffer.
    #[inline(always)]
    pub fn inner(&self) -> &ByteBuffer {
        &self.buffer
    }

    /// Retrieve the value at the given index.
    ///
    /// Panics if the index is out of bounds.
    ///
    /// Please note for repeatedly calling this function, please prefer [`crate::get_bit`].
    #[inline]
    pub fn value(&self, index: usize) -> bool {
        assert!(index < self.len);
        unsafe { self.value_unchecked(index) }
    }

    /// Retrieve the value at the given index without bounds checking
    ///
    /// # SAFETY
    /// Caller must ensure that index is within the range of the buffer
    #[inline]
    pub unsafe fn value_unchecked(&self, index: usize) -> bool {
        unsafe { get_bit_unchecked(self.buffer.as_ptr(), index + self.offset) }
    }

    /// Create a new zero-copy slice of this BoolBuffer that begins at the `start` index and extends
    /// for `len` bits.
    ///
    /// Panics if the slice would extend beyond the end of the buffer.
    pub fn slice(&self, range: Range<usize>) -> Self {
        assert!(
            range.len() <= self.len,
            "slice from {} to {} exceeds len {}",
            range.start,
            range.end,
            range.len()
        );

        Self::new_with_offset(self.buffer.clone(), range.len(), self.offset + range.start)
    }

    /// Slice any full bytes from the buffer, leaving the offset < 8.
    pub fn shrink_offset(self) -> Self {
        let bit_offset = self.offset % 8;
        let len = self.len;
        let buffer = self.into_inner();
        BitBuffer::new_with_offset(buffer, len, bit_offset)
    }

    /// Access chunks of the buffer aligned to 8 byte boundary as [prefix, \<full chunks\>, suffix]
    pub fn unaligned_chunks(&self) -> UnalignedBitChunk<'_> {
        UnalignedBitChunk::new(self.buffer.as_slice(), self.offset, self.len)
    }

    /// Access chunks of the underlying buffer as 8 byte chunks with a final trailer
    ///
    /// If you're performing operations on a single buffer, prefer [BitBuffer::unaligned_chunks]
    pub fn chunks(&self) -> BitChunks<'_> {
        BitChunks::new(self.buffer.as_slice(), self.offset, self.len)
    }

    /// Get the number of set bits in the buffer.
    pub fn true_count(&self) -> usize {
        self.unaligned_chunks().count_ones()
    }

    /// Get the number of unset bits in the buffer.
    pub fn false_count(&self) -> usize {
        self.len - self.true_count()
    }

    /// Iterator over bits in the buffer
    pub fn iter(&self) -> BitIterator<'_> {
        BitIterator::new(self.buffer.as_slice(), self.offset, self.len)
    }

    /// Iterator over set indices of the underlying buffer
    pub fn set_indices(&self) -> BitIndexIterator<'_> {
        BitIndexIterator::new(self.buffer.as_slice(), self.offset, self.len)
    }

    /// Iterator over set slices of the underlying buffer
    pub fn set_slices(&self) -> BitSliceIterator<'_> {
        BitSliceIterator::new(self.buffer.as_slice(), self.offset, self.len)
    }

    /// Created a new BitBuffer with offset reset to 0
    pub fn sliced(&self) -> Self {
        if self.offset % 8 == 0 {
            return Self::new(
                self.buffer.slice(self.offset / 8..self.len.div_ceil(8)),
                self.len,
            );
        }
        bitwise_unary_op(self, |a| a)
    }
}

// Conversions

impl BitBuffer {
    /// Consumes this `BoolBuffer` and returns the backing `Buffer<u8>` with any offset
    /// and length information applied.
    pub fn into_inner(self) -> ByteBuffer {
        let word_start = self.offset / 8;
        let word_end = (self.offset + self.len).div_ceil(8);

        self.buffer.slice(word_start..word_end)
    }

    /// Attempt to convert this `BitBuffer` into a mutable version.
    pub fn try_into_mut(self) -> Result<BitBufferMut, Self> {
        match self.buffer.try_into_mut() {
            Ok(buffer) => Ok(BitBufferMut::from_buffer(buffer, self.offset, self.len)),
            Err(buffer) => Err(BitBuffer::new_with_offset(buffer, self.len, self.offset)),
        }
    }

    /// Get a mutable version of this `BitBuffer` along with bit offset in the first byte.
    ///
    /// If the caller doesn't hold only reference to the underlying buffer, a copy is created.
    /// The second value of the tuple is a bit_offset of the first value in the first byte
    pub fn into_mut(self) -> BitBufferMut {
        let offset = self.offset;
        let len = self.len;
        // TODO(robert): if we are copying here we could strip offset bits
        let inner = self.into_inner().into_mut();
        BitBufferMut::from_buffer(inner, offset, len)
    }
}

impl From<&[bool]> for BitBuffer {
    fn from(value: &[bool]) -> Self {
        BitBufferMut::from(value).freeze()
    }
}

impl From<Vec<bool>> for BitBuffer {
    fn from(value: Vec<bool>) -> Self {
        BitBufferMut::from(value).freeze()
    }
}

impl FromIterator<bool> for BitBuffer {
    fn from_iter<T: IntoIterator<Item = bool>>(iter: T) -> Self {
        BitBufferMut::from_iter(iter).freeze()
    }
}

impl BitOr for &BitBuffer {
    type Output = BitBuffer;

    fn bitor(self, rhs: Self) -> Self::Output {
        bitwise_binary_op(self, rhs, |a, b| a | b)
    }
}

impl BitOr<&BitBuffer> for BitBuffer {
    type Output = BitBuffer;

    fn bitor(self, rhs: &BitBuffer) -> Self::Output {
        (&self).bitor(rhs)
    }
}

impl BitAnd for &BitBuffer {
    type Output = BitBuffer;

    fn bitand(self, rhs: Self) -> Self::Output {
        bitwise_binary_op(self, rhs, |a, b| a & b)
    }
}

impl BitAnd<&BitBuffer> for BitBuffer {
    type Output = BitBuffer;

    fn bitand(self, rhs: &BitBuffer) -> Self::Output {
        (&self).bitand(rhs)
    }
}

impl Not for &BitBuffer {
    type Output = BitBuffer;

    fn not(self) -> Self::Output {
        bitwise_unary_op(self, |a| !a)
    }
}

impl Not for BitBuffer {
    type Output = BitBuffer;

    fn not(self) -> Self::Output {
        (&self).not()
    }
}

impl BitXor for &BitBuffer {
    type Output = BitBuffer;

    fn bitxor(self, rhs: Self) -> Self::Output {
        bitwise_binary_op(self, rhs, |a, b| a ^ b)
    }
}

impl BitXor<&BitBuffer> for BitBuffer {
    type Output = BitBuffer;

    fn bitxor(self, rhs: &BitBuffer) -> Self::Output {
        (&self).bitxor(rhs)
    }
}

impl BitBuffer {
    /// Create a new BitBuffer by performing a bitwise AND NOT operation between two BitBuffers.
    ///
    /// This operation is sufficiently common that we provide a dedicated method for it avoid
    /// making two passes over the data.
    pub fn bitand_not(&self, rhs: &BitBuffer) -> BitBuffer {
        bitwise_binary_op(self, rhs, |a, b| a & !b)
    }
}

impl<'a> IntoIterator for &'a BitBuffer {
    type Item = bool;
    type IntoIter = BitIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use crate::bit::BitBuffer;
    use crate::{ByteBuffer, buffer};

    #[test]
    fn test_bool() {
        // Create a new Buffer<u64> of length 1024 where the 8th bit is set.
        let buffer: ByteBuffer = buffer![1 << 7; 1024];
        let bools = BitBuffer::new(buffer, 1024 * 8);

        // sanity checks
        assert_eq!(bools.len(), 1024 * 8);
        assert!(!bools.is_empty());
        assert_eq!(bools.true_count(), 1024);
        assert_eq!(bools.false_count(), 1024 * 7);

        // Check all the values
        for word in 0..1024 {
            for bit in 0..8 {
                if bit == 7 {
                    assert!(bools.value(word * 8 + bit));
                } else {
                    assert!(!bools.value(word * 8 + bit));
                }
            }
        }

        // Slice the buffer to create a new subset view.
        let sliced = bools.slice(64..72);

        // sanity checks
        assert_eq!(sliced.len(), 8);
        assert!(!sliced.is_empty());
        assert_eq!(sliced.true_count(), 1);
        assert_eq!(sliced.false_count(), 7);

        // Check all of the values like before
        for bit in 0..8 {
            if bit == 7 {
                assert!(sliced.value(bit));
            } else {
                assert!(!sliced.value(bit));
            }
        }
    }

    #[test]
    fn test_padded_equaltiy() {
        let buf1 = BitBuffer::new_set(64); // All bits set.
        let buf2 = BitBuffer::collect_bool(64, |x| x < 32); // First half set, other half unset.

        for i in 0..32 {
            assert_eq!(buf1.value(i), buf2.value(i), "Bit {} should be the same", i);
        }

        for i in 32..64 {
            assert_ne!(buf1.value(i), buf2.value(i), "Bit {} should differ", i);
        }

        assert_eq!(
            buf1.slice(0..32),
            buf2.slice(0..32),
            "Buffer slices with same bits should be equal (`PartialEq` needs `iter_padded()`)"
        );
        assert_ne!(
            buf1.slice(32..64),
            buf2.slice(32..64),
            "Buffer slices with different bits should not be equal (`PartialEq` needs `iter_padded()`)"
        );
    }
}
