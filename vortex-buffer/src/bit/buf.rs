// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;
use std::ops::BitAnd;
use std::ops::BitOr;
use std::ops::BitXor;
use std::ops::Bound;
use std::ops::Not;
use std::ops::RangeBounds;

use crate::Alignment;
use crate::BitBufferMut;
use crate::Buffer;
use crate::BufferMut;
use crate::ByteBuffer;
use crate::bit::BitChunks;
use crate::bit::BitIndexIterator;
use crate::bit::BitIterator;
use crate::bit::BitSliceIterator;
use crate::bit::UnalignedBitChunk;
use crate::bit::count_ones::count_ones;
use crate::bit::get_bit_unchecked;
use crate::bit::ops::bitwise_binary_op;
use crate::bit::ops::bitwise_unary_op;
use crate::bit::select::bit_select;
use crate::buffer;

/// An immutable bitset stored as a packed byte buffer.
#[derive(Debug, Clone, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BitBuffer {
    buffer: ByteBuffer,
    /// Represents the offset of the bit buffer into the first byte.
    ///
    /// This is always less than 8 (for when the bit buffer is not aligned to a byte).
    offset: usize,
    len: usize,
}

const LIMIT_LEN: usize = 16;
impl Display for BitBuffer {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let limit = f.precision().unwrap_or(LIMIT_LEN);
        let buf: Vec<bool> = self.into_iter().take(limit).collect();
        f.debug_struct("BitBuffer")
            .field("len", &self.len)
            .field("buffer", &buf)
            .finish()
    }
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

        // Slice the buffer to ensure the offset is within the first byte
        let byte_offset = offset / 8;
        let offset = offset % 8;
        let buffer = if byte_offset != 0 {
            buffer.slice(byte_offset..)
        } else {
            buffer
        };

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

    /// Create a bit buffer of `len` with `indices` set as true.
    pub fn from_indices(len: usize, indices: impl IntoIterator<Item = usize>) -> BitBuffer {
        BitBufferMut::from_indices(len, indices).freeze()
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

    /// Invokes `f` with indexes `0..len` collecting the boolean results into a new [`BitBuffer`].
    #[inline]
    pub fn collect_bool<F: FnMut(usize) -> bool>(len: usize, f: F) -> Self {
        BitBufferMut::collect_bool(len, f).freeze()
    }

    /// Maps over each bit in this buffer, calling `f(index, bit_value)` and collecting results.
    ///
    /// This is more efficient than `collect_bool` when you need to read the current bit value,
    /// as it unpacks each u64 chunk only once rather than doing random access for each bit.
    pub fn map_cmp<F>(&self, mut f: F) -> Self
    where
        F: FnMut(usize, bool) -> bool,
    {
        let len = self.len;
        let mut buffer: BufferMut<u64> = BufferMut::with_capacity(len.div_ceil(64));

        let chunks_count = len / 64;
        let remainder = len % 64;
        let chunks = self.chunks();

        for (chunk_idx, src_chunk) in chunks.iter().enumerate() {
            let mut packed = 0u64;
            for bit_idx in 0..64 {
                let i = bit_idx + chunk_idx * 64;
                let bit_value = (src_chunk >> bit_idx) & 1 == 1;
                packed |= (f(i, bit_value) as u64) << bit_idx;
            }

            // SAFETY: Already allocated sufficient capacity
            unsafe { buffer.push_unchecked(packed) }
        }

        if remainder != 0 {
            let src_chunk = chunks.remainder_bits();
            let mut packed = 0u64;
            for bit_idx in 0..remainder {
                let i = bit_idx + chunks_count * 64;
                let bit_value = (src_chunk >> bit_idx) & 1 == 1;
                packed |= (f(i, bit_value) as u64) << bit_idx;
            }

            // SAFETY: Already allocated sufficient capacity
            unsafe { buffer.push_unchecked(packed) }
        }

        buffer.truncate(len.div_ceil(8));

        Self {
            buffer: buffer.freeze().into_byte_buffer(),
            offset: 0,
            len,
        }
    }

    /// Clear all bits in the buffer, preserving existing capacity.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.len = 0;
        self.offset = 0;
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
    pub fn slice(&self, range: impl RangeBounds<usize>) -> Self {
        let start = match range.start_bound() {
            Bound::Included(&s) => s,
            Bound::Excluded(&s) => s + 1,
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&e) => e + 1,
            Bound::Excluded(&e) => e,
            Bound::Unbounded => self.len,
        };

        assert!(start <= end);
        assert!(start <= self.len);
        assert!(end <= self.len);
        let len = end - start;

        Self::new_with_offset(self.buffer.clone(), len, self.offset + start)
    }

    /// Slice any full bytes from the buffer, leaving the offset < 8.
    pub fn shrink_offset(self) -> Self {
        let word_start = self.offset / 8;
        let word_end = (self.offset + self.len).div_ceil(8);

        let buffer = self.buffer.slice(word_start..word_end);

        let bit_offset = self.offset % 8;
        let len = self.len;
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
        count_ones(self.buffer.as_slice(), self.offset, self.len)
    }

    /// Returns the position of the `nth` set bit (0-indexed).
    ///
    /// This is the "select" operation on a bitmap: given a rank `nth`, find
    /// which logical bit position holds that rank.
    ///
    /// Returns `None` if `nth` is greater than or equal to the number of set bits.
    pub fn select(&self, nth: usize) -> Option<usize> {
        bit_select(self.buffer.as_slice(), self.offset, self.len, nth)
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
        if self.offset.is_multiple_of(8) {
            return Self::new(
                self.buffer.slice(self.offset / 8..self.len.div_ceil(8)),
                self.len,
            );
        }

        bitwise_unary_op(self.clone(), |a| a)
    }
}

// Conversions

impl BitBuffer {
    /// Returns the offset, len and underlying buffer.
    pub fn into_inner(self) -> (usize, usize, ByteBuffer) {
        (self.offset, self.len, self.buffer)
    }

    /// Attempt to convert this `BitBuffer` into a mutable version.
    pub fn try_into_mut(self) -> Result<BitBufferMut, Self> {
        match self.buffer.try_into_mut() {
            Ok(buffer) => Ok(BitBufferMut::from_buffer(buffer, self.offset, self.len)),
            Err(buffer) => Err(BitBuffer::new_with_offset(buffer, self.len, self.offset)),
        }
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

impl BitOr for BitBuffer {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        BitOr::bitor(&self, &rhs)
    }
}

impl BitOr for &BitBuffer {
    type Output = BitBuffer;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        bitwise_binary_op(self, rhs, |a, b| a | b)
    }
}

impl BitOr<&BitBuffer> for BitBuffer {
    type Output = BitBuffer;

    #[inline]
    fn bitor(self, rhs: &BitBuffer) -> Self::Output {
        (&self).bitor(rhs)
    }
}

impl BitAnd for &BitBuffer {
    type Output = BitBuffer;

    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        bitwise_binary_op(self, rhs, |a, b| a & b)
    }
}

impl BitAnd<BitBuffer> for &BitBuffer {
    type Output = BitBuffer;

    #[inline]
    fn bitand(self, rhs: BitBuffer) -> Self::Output {
        self.bitand(&rhs)
    }
}

impl BitAnd<&BitBuffer> for BitBuffer {
    type Output = BitBuffer;

    #[inline]
    fn bitand(self, rhs: &BitBuffer) -> Self::Output {
        (&self).bitand(rhs)
    }
}

impl BitAnd<BitBuffer> for BitBuffer {
    type Output = BitBuffer;

    #[inline]
    fn bitand(self, rhs: BitBuffer) -> Self::Output {
        (&self).bitand(&rhs)
    }
}

impl Not for &BitBuffer {
    type Output = BitBuffer;

    #[inline]
    fn not(self) -> Self::Output {
        !self.clone()
    }
}

impl Not for BitBuffer {
    type Output = BitBuffer;

    #[inline]
    fn not(self) -> Self::Output {
        bitwise_unary_op(self, |a| !a)
    }
}

impl BitXor for &BitBuffer {
    type Output = BitBuffer;

    #[inline]
    fn bitxor(self, rhs: Self) -> Self::Output {
        bitwise_binary_op(self, rhs, |a, b| a ^ b)
    }
}

impl BitXor<&BitBuffer> for BitBuffer {
    type Output = BitBuffer;

    #[inline]
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

    /// Iterate through bits in a buffer.
    ///
    /// # Arguments
    ///
    /// * `f` - Callback function taking (bit_index, is_set)
    ///
    /// # Panics
    ///
    /// Panics if the range is outside valid bounds of the buffer.
    #[inline]
    pub fn iter_bits<F>(&self, mut f: F)
    where
        F: FnMut(usize, bool),
    {
        let total_bits = self.len;
        if total_bits == 0 {
            return;
        }

        let is_bit_set = |byte: u8, bit_idx: usize| (byte & (1 << bit_idx)) != 0;
        let bit_offset = self.offset % 8;
        let mut buffer_ptr = unsafe { self.buffer.as_ptr().add(self.offset / 8) };
        let mut callback_idx = 0;

        // Handle incomplete first byte.
        if bit_offset > 0 {
            let bits_in_first_byte = (8 - bit_offset).min(total_bits);
            let byte = unsafe { *buffer_ptr };

            for bit_idx in 0..bits_in_first_byte {
                f(callback_idx, is_bit_set(byte, bit_offset + bit_idx));
                callback_idx += 1;
            }

            buffer_ptr = unsafe { buffer_ptr.add(1) };
        }

        // Process complete bytes.
        let complete_bytes = (total_bits - callback_idx) / 8;
        for _ in 0..complete_bytes {
            let byte = unsafe { *buffer_ptr };

            for bit_idx in 0..8 {
                f(callback_idx, is_bit_set(byte, bit_idx));
                callback_idx += 1;
            }
            buffer_ptr = unsafe { buffer_ptr.add(1) };
        }

        // Handle remaining bits at the end.
        let remaining_bits = total_bits - callback_idx;
        if remaining_bits > 0 {
            let byte = unsafe { *buffer_ptr };

            for bit_idx in 0..remaining_bits {
                f(callback_idx, is_bit_set(byte, bit_idx));
                callback_idx += 1;
            }
        }
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
    use rstest::rstest;

    use crate::ByteBuffer;
    use crate::bit::BitBuffer;
    use crate::buffer;

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

    #[test]
    fn test_slice_offset_calculation() {
        let buf = BitBuffer::collect_bool(16, |_| true);
        let sliced = buf.slice(10..16);
        assert_eq!(sliced.len(), 6);
        // Ensure the offset is modulo 8
        assert_eq!(sliced.offset(), 2);
    }

    #[test]
    fn test_from_indices_dense_crosses_words() {
        let len = 130;
        let indices = (0..len).filter(|idx| idx % 3 != 1);
        let buf = BitBuffer::from_indices(len, indices);

        assert_eq!(buf.len(), len);
        for idx in 0..len {
            assert_eq!(buf.value(idx), idx % 3 != 1, "mismatch at {idx}");
        }
    }

    #[test]
    #[should_panic(expected = "index 5 exceeds len 5")]
    fn test_from_indices_out_of_bounds() {
        BitBuffer::from_indices(5, [0, 5]);
    }

    #[rstest]
    #[case(5)]
    #[case(8)]
    #[case(10)]
    #[case(13)]
    #[case(16)]
    #[case(23)]
    #[case(100)]
    fn test_iter_bits(#[case] len: usize) {
        let buf = BitBuffer::collect_bool(len, |i| i % 2 == 0);

        let mut collected = Vec::new();
        buf.iter_bits(|idx, is_set| {
            collected.push((idx, is_set));
        });

        assert_eq!(collected.len(), len);

        for (idx, is_set) in collected {
            assert_eq!(is_set, idx % 2 == 0);
        }
    }

    #[rstest]
    #[case(3, 5)]
    #[case(3, 8)]
    #[case(5, 10)]
    #[case(2, 16)]
    #[case(8, 16)]
    #[case(9, 16)]
    #[case(17, 16)]
    fn test_iter_bits_with_offset(#[case] offset: usize, #[case] len: usize) {
        let total_bits = offset + len;
        let buf = BitBuffer::collect_bool(total_bits, |i| i % 2 == 0);
        let buf_with_offset = BitBuffer::new_with_offset(buf.inner().clone(), len, offset);

        let mut collected = Vec::new();
        buf_with_offset.iter_bits(|idx, is_set| {
            collected.push((idx, is_set));
        });

        assert_eq!(collected.len(), len);

        for (idx, is_set) in collected {
            // The bits should match the original buffer at positions offset + idx
            assert_eq!(is_set, (offset + idx).is_multiple_of(2));
        }
    }

    #[rstest]
    #[case(8, 10)]
    #[case(9, 7)]
    #[case(16, 8)]
    #[case(17, 10)]
    fn test_iter_bits_catches_wrong_byte_offset(#[case] offset: usize, #[case] len: usize) {
        let total_bits = offset + len;
        // Alternating pattern to catch byte offset errors: Bits are set for even indexed bytes.
        let buf = BitBuffer::collect_bool(total_bits, |i| (i / 8) % 2 == 0);

        let buf_with_offset = BitBuffer::new_with_offset(buf.inner().clone(), len, offset);

        let mut collected = Vec::new();
        buf_with_offset.iter_bits(|idx, is_set| {
            collected.push((idx, is_set));
        });

        assert_eq!(collected.len(), len);

        for (idx, is_set) in collected {
            let bit_position = offset + idx;
            let byte_index = bit_position / 8;
            let expected_is_set = byte_index.is_multiple_of(2);

            assert_eq!(
                is_set, expected_is_set,
                "Bit mismatch at index {}: expected {} got {}",
                bit_position, expected_is_set, is_set
            );
        }
    }

    #[rstest]
    #[case(5)]
    #[case(8)]
    #[case(10)]
    #[case(64)]
    #[case(65)]
    #[case(100)]
    #[case(128)]
    fn test_map_cmp_identity(#[case] len: usize) {
        // map_cmp with identity function should return the same buffer
        let buf = BitBuffer::collect_bool(len, |i| i % 3 == 0);
        let mapped = buf.map_cmp(|_idx, bit| bit);

        assert_eq!(buf.len(), mapped.len());
        for i in 0..len {
            assert_eq!(buf.value(i), mapped.value(i), "Mismatch at index {}", i);
        }
    }

    #[rstest]
    #[case(5)]
    #[case(8)]
    #[case(64)]
    #[case(65)]
    #[case(100)]
    fn test_map_cmp_negate(#[case] len: usize) {
        // map_cmp negating all bits
        let buf = BitBuffer::collect_bool(len, |i| i % 2 == 0);
        let mapped = buf.map_cmp(|_idx, bit| !bit);

        assert_eq!(buf.len(), mapped.len());
        for i in 0..len {
            assert_eq!(!buf.value(i), mapped.value(i), "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_map_cmp_conditional() {
        // map_cmp with conditional logic based on index and bit value
        let len = 100;
        let buf = BitBuffer::collect_bool(len, |i| i % 2 == 0);

        // Only keep bits that are set AND at even index divisible by 4
        let mapped = buf.map_cmp(|idx, bit| bit && idx % 4 == 0);

        for i in 0..len {
            let expected = (i % 2 == 0) && (i % 4 == 0);
            assert_eq!(mapped.value(i), expected, "Mismatch at index {}", i);
        }
    }
}
