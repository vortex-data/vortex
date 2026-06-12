// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Not;

use bitvec::view::BitView;

use crate::BitBuffer;
use crate::BufferMut;
use crate::ByteBufferMut;
use crate::bit::collect_bool_words;
use crate::bit::get_bit_unchecked;
use crate::bit::ops;
use crate::bit::set_bit_unchecked;
use crate::bit::unset_bit_unchecked;
use crate::buffer_mut;

/// Sets all bits in the bit-range `[start_bit, end_bit)` of `slice` to `value`.
#[inline(always)]
pub(crate) fn fill_bits(slice: &mut [u8], start_bit: usize, end_bit: usize, value: bool) {
    if start_bit >= end_bit {
        return;
    }

    let fill_byte: u8 = if value { 0xFF } else { 0x00 };

    let start_byte = start_bit / 8;
    let start_rem = start_bit % 8;
    let end_byte = end_bit / 8;
    let end_rem = end_bit % 8;

    if start_byte == end_byte {
        // All bits are in the same byte
        let mask = ((1u8 << (end_rem - start_rem)) - 1) << start_rem;
        if value {
            slice[start_byte] |= mask;
        } else {
            slice[start_byte] &= !mask;
        }
    } else {
        // First partial byte
        if start_rem != 0 {
            let mask = !((1u8 << start_rem) - 1);
            if value {
                slice[start_byte] |= mask;
            } else {
                slice[start_byte] &= !mask;
            }
        }

        // Middle bytes
        let fill_start = if start_rem != 0 {
            start_byte + 1
        } else {
            start_byte
        };
        if fill_start < end_byte {
            slice[fill_start..end_byte].fill(fill_byte);
        }

        // Last partial byte
        if end_rem != 0 {
            let mask = (1u8 << end_rem) - 1;
            if value {
                slice[end_byte] |= mask;
            } else {
                slice[end_byte] &= !mask;
            }
        }
    }
}

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
/// See also: [`BitBuffer`].
#[derive(Debug, Clone)]
pub struct BitBufferMut {
    buffer: ByteBufferMut,
    /// Represents the offset of the bit buffer into the first byte.
    ///
    /// This is always less than 8 (for when the bit buffer is not aligned to a byte).
    offset: usize,
    len: usize,
}

impl BitBufferMut {
    /// Create new bit buffer from given byte buffer and logical bit length
    pub fn from_buffer(buffer: ByteBufferMut, offset: usize, len: usize) -> Self {
        assert!(
            len <= buffer.len() * 8,
            "Buffer len {} is too short for the given length {len}",
            buffer.len()
        );
        Self {
            buffer,
            offset,
            len,
        }
    }

    /// Creates a `BitBufferMut` from a [`BitBuffer`] by copying all of the data over.
    pub fn copy_from(bit_buffer: &BitBuffer) -> Self {
        Self {
            buffer: ByteBufferMut::copy_from(bit_buffer.inner()),
            offset: bit_buffer.offset(),
            len: bit_buffer.len(),
        }
    }

    /// Create a new empty mutable bit buffer with requested capacity (in bits).
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: BufferMut::with_capacity(capacity.div_ceil(8)),
            offset: 0,
            len: 0,
        }
    }

    /// Create a new mutable buffer with requested `len` and all bits set to `true`.
    pub fn new_set(len: usize) -> Self {
        Self {
            buffer: buffer_mut![0xFF; len.div_ceil(8)],
            offset: 0,
            len,
        }
    }

    /// Create a new mutable buffer with requested `len` and all bits set to `false`.
    pub fn new_unset(len: usize) -> Self {
        Self {
            buffer: BufferMut::zeroed(len.div_ceil(8)),
            offset: 0,
            len,
        }
    }

    /// Create a new empty `BitBufferMut`.
    #[inline(always)]
    pub fn empty() -> Self {
        Self::with_capacity(0)
    }

    /// Create a new mutable buffer with requested `len` and all bits set to `value`.
    pub fn full(value: bool, len: usize) -> Self {
        if value {
            Self::new_set(len)
        } else {
            Self::new_unset(len)
        }
    }

    /// Create a bit buffer of `len` with `indices` set as true.
    pub fn from_indices(len: usize, indices: impl IntoIterator<Item = usize>) -> BitBufferMut {
        let mut buffer = BufferMut::<u64>::zeroed(len.div_ceil(64));
        for idx in indices {
            assert!(idx < len, "index {idx} exceeds len {len}");
            buffer.as_mut_slice()[idx / 64] |= 1 << (idx % 64);
        }

        let mut buffer = buffer.into_byte_buffer();
        buffer.truncate(len.div_ceil(8));

        Self {
            buffer,
            offset: 0,
            len,
        }
    }

    /// Invokes `f` with indexes `0..len` collecting the boolean results into a new `BitBufferMut`
    #[inline]
    pub fn collect_bool<F: FnMut(usize) -> bool>(len: usize, f: F) -> Self {
        let num_words = len.div_ceil(64);
        let mut buffer: BufferMut<u64> = BufferMut::with_capacity(num_words);
        // SAFETY: `collect_bool_words` writes every word in `0..num_words` below
        // before any read; `u64` has no invalid bit patterns and the assignments
        // inside `collect_bool_words` are pure writes.
        unsafe { buffer.set_len(num_words) };
        collect_bool_words(buffer.as_mut_slice(), len, f);

        let mut bytes = buffer.into_byte_buffer();
        bytes.truncate(len.div_ceil(8));

        Self {
            buffer: bytes,
            offset: 0,
            len,
        }
    }

    /// Return the underlying byte buffer.
    pub fn inner(&self) -> &ByteBufferMut {
        &self.buffer
    }

    /// Consumes the buffer and return the underlying byte buffer.
    pub fn into_inner(self) -> ByteBufferMut {
        self.buffer
    }

    /// Get the current populated length of the buffer.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if the buffer has length 0.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the current bit offset of the buffer.
    #[inline(always)]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get the value at the requested index.
    #[inline(always)]
    pub fn value(&self, index: usize) -> bool {
        assert!(index < self.len);
        // SAFETY: checked by assertion
        unsafe { self.value_unchecked(index) }
    }

    /// Get the value at the requested index without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index` is less than the length of the buffer.
    #[inline(always)]
    pub unsafe fn value_unchecked(&self, index: usize) -> bool {
        unsafe { get_bit_unchecked(self.buffer.as_ptr(), self.offset + index) }
    }

    /// Get the bit capacity of the buffer.
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        (self.buffer.capacity() * 8) - self.offset
    }

    /// Reserve additional bit capacity for the buffer.
    pub fn reserve(&mut self, additional: usize) {
        let required_bits = self.offset + self.len + additional;
        let required_bytes = required_bits.div_ceil(8); // Rounds up.

        let additional_bytes = required_bytes.saturating_sub(self.buffer.len());
        self.buffer.reserve(additional_bytes);
    }

    /// Clears the bit buffer (but keeps any allocated memory).
    pub fn clear(&mut self) {
        // Since there are no items we need to drop, we simply set the length to 0.
        self.len = 0;
        self.offset = 0;
    }

    /// Set the bit at `index` to the given boolean value.
    ///
    /// This operation is checked so if `index` exceeds the buffer length, this will panic.
    #[inline]
    pub fn set_to(&mut self, index: usize, value: bool) {
        if value {
            self.set(index);
        } else {
            self.unset(index);
        }
    }

    /// Set the bit at `index` to the given boolean value without checking bounds.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index` does not exceed the largest bit index in the backing buffer.
    #[inline]
    pub unsafe fn set_to_unchecked(&mut self, index: usize, value: bool) {
        if value {
            // SAFETY: checked by caller
            unsafe { self.set_unchecked(index) }
        } else {
            // SAFETY: checked by caller
            unsafe { self.unset_unchecked(index) }
        }
    }

    /// Set a position to `true`.
    ///
    /// This operation is checked so if `index` exceeds the buffer length, this will panic.
    #[inline]
    pub fn set(&mut self, index: usize) {
        assert!(index < self.len, "index {index} exceeds len {}", self.len);

        // SAFETY: checked by assertion
        unsafe { self.set_unchecked(index) };
    }

    /// Set a position to `false`.
    ///
    /// This operation is checked so if `index` exceeds the buffer length, this will panic.
    #[inline]
    pub fn unset(&mut self, index: usize) {
        assert!(index < self.len, "index {index} exceeds len {}", self.len);

        // SAFETY: checked by assertion
        unsafe { self.unset_unchecked(index) };
    }

    /// Set the bit at `index` to `true` without checking bounds.
    ///
    /// Note: Do not call this in a tight loop. Prefer to use [`set_bit_unchecked`].
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index` does not exceed the largest bit index in the backing buffer.
    #[inline]
    pub unsafe fn set_unchecked(&mut self, index: usize) {
        // SAFETY: checked by caller
        unsafe { set_bit_unchecked(self.buffer.as_mut_ptr(), self.offset + index) }
    }

    /// Unset the bit at `index` without checking bounds.
    ///
    /// Note: Do not call this in a tight loop. Prefer to use [`unset_bit_unchecked`].
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index` does not exceed the largest bit index in the backing buffer.
    #[inline]
    pub unsafe fn unset_unchecked(&mut self, index: usize) {
        // SAFETY: checked by caller
        unsafe { unset_bit_unchecked(self.buffer.as_mut_ptr(), self.offset + index) }
    }

    /// Foces the length of the `BitBufferMut` to `new_len`.
    ///
    /// # Safety
    ///
    /// - `new_len` must be less than or equal to [`capacity()`](Self::capacity)
    /// - The elements at `old_len..new_len` must be initialized
    #[inline(always)]
    pub unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(
            new_len <= self.capacity(),
            "`set_len` requires that new_len <= capacity()"
        );

        // Calculate the new byte length required to hold the bits
        let bytes_len = (self.offset + new_len).div_ceil(8);
        unsafe { self.buffer.set_len(bytes_len) };

        self.len = new_len;
    }

    /// Truncate the buffer to the given length.
    ///
    /// If the given length is greater than the current length, this is a no-op.
    pub fn truncate(&mut self, len: usize) {
        if len > self.len {
            return;
        }

        let end_bit = self.offset + len;
        let new_len_bytes = end_bit.div_ceil(8);
        self.buffer.truncate(new_len_bytes);
        self.len = len;

        // Clear stale bits in the final partial byte so the "bits beyond len are zero" invariant
        // holds. `append_false` (and `append_buffer`) rely on it to avoid a read-modify-write.
        if !end_bit.is_multiple_of(8) {
            let keep = (1u8 << (end_bit % 8)) - 1;
            self.buffer.as_mut_slice()[new_len_bytes - 1] &= keep;
        }
    }

    /// Append a new boolean into the bit buffer, incrementing the length.
    #[inline]
    pub fn append(&mut self, value: bool) {
        if value {
            self.append_true()
        } else {
            self.append_false()
        }
    }

    /// Append a new true value to the buffer.
    #[inline]
    pub fn append_true(&mut self) {
        let bit_pos = self.offset + self.len;
        let byte_pos = bit_pos / 8;
        let bit_in_byte = bit_pos % 8;

        // Ensure buffer has enough bytes
        if byte_pos >= self.buffer.len() {
            self.buffer.push(0u8);
        }

        // Set the bit
        self.buffer.as_mut_slice()[byte_pos] |= 1 << bit_in_byte;
        self.len += 1;
    }

    /// Append a new false value to the buffer.
    #[inline]
    pub fn append_false(&mut self) {
        let bit_pos = self.offset + self.len;
        let byte_pos = bit_pos / 8;

        // Ensure buffer has enough bytes (pushed as 0x00, so bit is already unset).
        if byte_pos >= self.buffer.len() {
            self.buffer.push(0u8);
        }

        // The bit is guaranteed to be 0: new bytes are zero-initialized, and
        // existing bytes have this bit unset (it's beyond the current length).
        self.len += 1;
    }

    /// Append several boolean values into the bit buffer. After this operation,
    /// the length will be incremented by `n`.
    ///
    /// Panics if the buffer does not have `n` slots left.
    #[inline]
    pub fn append_n(&mut self, value: bool, n: usize) {
        if n == 0 {
            return;
        }

        let end_bit_pos = self.offset + self.len + n;
        let required_bytes = end_bit_pos.div_ceil(8);

        // Ensure buffer has enough bytes
        if required_bytes > self.buffer.len() {
            self.buffer.push_n(0x00, required_bytes - self.buffer.len());
        }

        let start = self.len;
        self.len += n;
        self.fill_range(start, self.len, value);
    }

    /// Sets all bits in the range `[start, end)` to `value`.
    ///
    /// This operates on an arbitrary range within the existing length of the buffer.
    /// Panics if `end > self.len` or `start > end`.
    #[inline(always)]
    pub fn fill_range(&mut self, start: usize, end: usize, value: bool) {
        assert!(end <= self.len, "end {end} exceeds len {}", self.len);
        assert!(start <= end, "start {start} exceeds end {end}");

        // SAFETY: assertions above guarantee start <= end <= self.len,
        // so offset + end fits within the buffer.
        unsafe { self.fill_range_unchecked(start, end, value) }
    }

    /// Sets all bits in the range `[start, end)` to `value` without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `start <= end <= self.len`.
    #[inline(always)]
    pub unsafe fn fill_range_unchecked(&mut self, start: usize, end: usize, value: bool) {
        fill_bits(
            self.buffer.as_mut_slice(),
            self.offset + start,
            self.offset + end,
            value,
        );
    }

    /// Append a [`BitBuffer`] to this [`BitBufferMut`]
    ///
    /// This efficiently copies all bits from the source buffer to the end of this buffer.
    pub fn append_buffer(&mut self, buffer: &BitBuffer) {
        let bit_len = buffer.len();
        if bit_len == 0 {
            return;
        }

        let start_bit_pos = self.offset + self.len;
        let end_bit_pos = start_bit_pos + bit_len;
        let required_bytes = end_bit_pos.div_ceil(8);

        // Ensure buffer has enough bytes, zero-initialized for OR-based writes.
        if required_bytes > self.buffer.len() {
            self.buffer.push_n(0x00, required_bytes - self.buffer.len());
        }

        let dst_bit_offset = start_bit_pos % 8;
        let src_bit_offset = buffer.offset();

        if dst_bit_offset == 0 && src_bit_offset == 0 {
            // Both byte-aligned: use memcpy for full bytes, then mask the tail.
            let dst_byte = start_bit_pos / 8;
            let src_bytes = buffer.inner().as_slice();
            let full_bytes = bit_len / 8;
            self.buffer.as_mut_slice()[dst_byte..dst_byte + full_bytes]
                .copy_from_slice(&src_bytes[..full_bytes]);
            let rem = bit_len % 8;
            if rem != 0 {
                let mask = (1u8 << rem) - 1;
                self.buffer.as_mut_slice()[dst_byte + full_bytes] |= src_bytes[full_bytes] & mask;
            }
        } else {
            // Use bitvec for unaligned bit copying.
            let self_slice = self
                .buffer
                .as_mut_slice()
                .view_bits_mut::<bitvec::prelude::Lsb0>();
            let other_slice = buffer
                .inner()
                .as_slice()
                .view_bits::<bitvec::prelude::Lsb0>();
            let source_range = src_bit_offset..src_bit_offset + bit_len;
            self_slice[start_bit_pos..end_bit_pos].copy_from_bitslice(&other_slice[source_range]);
        }

        self.len += bit_len;
    }

    /// Absorbs a mutable buffer that was previously split off.
    ///
    /// If the two buffers were previously contiguous and not mutated in a way that causes
    /// re-allocation i.e., if other was created by calling split_off on this buffer, then this is
    /// an O(1) operation that just decreases a reference count and sets a few indices.
    ///
    /// Otherwise, this method degenerates to self.append_buffer(&other).
    pub fn unsplit(&mut self, other: Self) {
        if (self.offset + self.len).is_multiple_of(8) && other.offset == 0 {
            // We are aligned and can just append the buffers
            self.buffer.unsplit(other.buffer);
            self.len += other.len;
            return;
        }

        // Otherwise, we need to append the bits one by one
        self.append_buffer(&other.freeze())
    }

    /// Freeze the buffer in its current state into an immutable `BoolBuffer`.
    pub fn freeze(self) -> BitBuffer {
        BitBuffer::new_with_offset(self.buffer.freeze(), self.len, self.offset)
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

// Mutate-in-place implementation of bitwise NOT.
impl Not for BitBufferMut {
    type Output = BitBufferMut;

    #[inline]
    fn not(mut self) -> Self::Output {
        ops::bitwise_unary_op_mut(&mut self, |b| !b);
        self
    }
}

impl From<&[bool]> for BitBufferMut {
    fn from(value: &[bool]) -> Self {
        BitBufferMut::collect_bool(value.len(), |i| value[i])
    }
}

// allow building a buffer from a set of truthy byte values.
impl From<&[u8]> for BitBufferMut {
    fn from(value: &[u8]) -> Self {
        BitBufferMut::collect_bool(value.len(), |i| value[i] > 0)
    }
}

impl From<Vec<bool>> for BitBufferMut {
    fn from(value: Vec<bool>) -> Self {
        value.as_slice().into()
    }
}

impl FromIterator<bool> for BitBufferMut {
    #[inline]
    fn from_iter<T: IntoIterator<Item = bool>>(iter: T) -> Self {
        let mut iter = iter.into_iter();

        // Since we do not know the length of the iterator, we can only guess how much memory we
        // need to reserve. Note that these hints may be inaccurate.
        let (lower_bound, _) = iter.size_hint();

        // We choose not to use the optional upper bound size hint to match the standard library.

        // Initialize all bits to 0 with the given length. By doing this, we only need to set bits
        // that are true (and this is faster from benchmarks).
        let mut buf = BitBufferMut::new_unset(lower_bound);
        assert_eq!(buf.offset, 0);

        // Directly write within our known capacity.
        let ptr = buf.buffer.as_mut_ptr();
        for i in 0..lower_bound {
            let Some(v) = iter.next() else {
                // SAFETY: We are definitely under the capacity and all values are already
                // initialized from `new_unset`.
                unsafe { buf.set_len(i) };
                return buf;
            };

            if v {
                // SAFETY: We have ensured that we are within the capacity.
                unsafe { set_bit_unchecked(ptr, i) }
            }
        }

        // Append any remaining items one at a time, as we do not know how many more there are.
        // (`append` is already a single branch + bit set, see `append_true`/`append_false`.)
        for v in iter {
            buf.append(v);
        }

        buf
    }
}

#[cfg(test)]
mod tests {
    use crate::BufferMut;
    use crate::bit::buf_mut::BitBufferMut;
    use crate::bitbuffer;
    use crate::bitbuffer_mut;
    use crate::buffer_mut;

    #[test]
    fn test_bits_mut() {
        let mut bools = bitbuffer_mut![false; 10];
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

    #[test]
    fn append_false_after_truncate_reads_back_false() {
        // `truncate` leaves stale bits in the final partial byte; a subsequent `append_false`
        // must still read back as false. Regression test for the `append_false` fast path.
        let mut bools = BitBufferMut::new_set(16);
        bools.truncate(12);
        bools.append_false();
        bools.append_true();

        let bools = bools.freeze();
        assert_eq!(bools.len(), 14);
        assert!(
            !bools.value(12),
            "appended false must read back false after truncate"
        );
        assert!(bools.value(13));
    }

    #[test]
    fn test_reserve_ensures_len_plus_additional() {
        // This test documents the fix for the bug where reserve was incorrectly
        // calculating additional bytes from capacity instead of len.

        let mut bits = BitBufferMut::with_capacity(10);
        assert_eq!(bits.len(), 0);

        bits.reserve(100);

        // Should have capacity for at least len + 100 = 0 + 100 = 100 bits.
        assert!(bits.capacity() >= 100);

        bits.append_n(true, 50);
        assert_eq!(bits.len(), 50);

        bits.reserve(100);

        // Should have capacity for at least len + 100 = 50 + 100 = 150 bits.
        assert!(bits.capacity() >= 150);
    }

    #[test]
    fn test_with_offset_zero() {
        // Test basic operations when offset is 0
        let buf = BufferMut::zeroed(2);
        let mut bit_buf = BitBufferMut::from_buffer(buf, 0, 16);

        // Set some bits
        bit_buf.set(0);
        bit_buf.set(7);
        bit_buf.set(8);
        bit_buf.set(15);

        // Verify values
        assert!(bit_buf.value(0));
        assert!(bit_buf.value(7));
        assert!(bit_buf.value(8));
        assert!(bit_buf.value(15));
        assert!(!bit_buf.value(1));
        assert!(!bit_buf.value(9));

        // Verify underlying bytes
        assert_eq!(bit_buf.as_slice()[0], 0b10000001);
        assert_eq!(bit_buf.as_slice()[1], 0b10000001);
    }

    #[test]
    fn test_with_offset_within_byte() {
        // Test operations with offset=3 (within first byte)
        let buf = buffer_mut![0b11111111, 0b00000000, 0b00000000];
        let mut bit_buf = BitBufferMut::from_buffer(buf, 3, 10);

        // Initially, bits 3-7 from first byte are set (5 bits)
        // and bits 0-4 from second byte are unset (5 bits more)
        assert!(bit_buf.value(0)); // bit 3 of byte 0
        assert!(bit_buf.value(4)); // bit 7 of byte 0
        assert!(!bit_buf.value(5)); // bit 0 of byte 1

        // Set a bit in the second byte's range
        bit_buf.set(7);
        assert!(bit_buf.value(7));

        // Unset a bit in the first byte's range
        bit_buf.unset(0);
        assert!(!bit_buf.value(0));
    }

    #[test]
    fn test_with_offset_byte_boundary() {
        // Test operations with offset=8 (exactly one byte)
        let buf = buffer_mut![0xFF, 0x00, 0xFF];
        let mut bit_buf = BitBufferMut::from_buffer(buf, 8, 16);

        // Buffer starts at byte 1, so all bits should be unset initially
        for i in 0..8 {
            assert!(!bit_buf.value(i));
        }
        // Next byte has all bits set
        for i in 8..16 {
            assert!(bit_buf.value(i));
        }

        // Set some bits
        bit_buf.set(0);
        bit_buf.set(3);
        assert!(bit_buf.value(0));
        assert!(bit_buf.value(3));
    }

    #[test]
    fn test_with_large_offset() {
        // Test with offset=13 (one byte + 5 bits)
        let buf = buffer_mut![0xFF, 0xFF, 0xFF, 0xFF];
        let mut bit_buf = BitBufferMut::from_buffer(buf, 13, 10);

        // All bits should initially be set
        for i in 0..10 {
            assert!(bit_buf.value(i));
        }

        // Unset some bits
        bit_buf.unset(0);
        bit_buf.unset(5);
        bit_buf.unset(9);

        assert!(!bit_buf.value(0));
        assert!(bit_buf.value(1));
        assert!(!bit_buf.value(5));
        assert!(!bit_buf.value(9));
    }

    #[test]
    fn test_append_with_offset() {
        // Create buffer with offset
        let buf = buffer_mut![0b11100000]; // First 3 bits unset, last 5 set
        let mut bit_buf = BitBufferMut::from_buffer(buf, 3, 0); // Start at bit 3, len=0

        // Append some bits
        bit_buf.append(false); // Should use bit 3
        bit_buf.append(true); // Should use bit 4
        bit_buf.append(true); // Should use bit 5

        assert_eq!(bit_buf.len(), 3);
        assert!(!bit_buf.value(0));
        assert!(bit_buf.value(1));
        assert!(bit_buf.value(2));
    }

    #[test]
    fn test_append_n_with_offset_crossing_boundary() {
        // Create buffer with offset that will cross byte boundary when appending
        let buf = BufferMut::zeroed(4);
        let mut bit_buf = BitBufferMut::from_buffer(buf, 5, 0);

        // Append enough bits to cross into next byte
        bit_buf.append_n(true, 10); // 5 bits left in first byte, then 5 in second

        assert_eq!(bit_buf.len(), 10);
        for i in 0..10 {
            assert!(bit_buf.value(i));
        }

        // Verify the underlying bytes
        // Bits 5-7 of byte 0 should be set (3 bits)
        // Bits 0-6 of byte 1 should be set (7 bits)
        assert_eq!(bit_buf.as_slice()[0], 0b11100000);
        assert_eq!(bit_buf.as_slice()[1], 0b01111111);
    }

    #[test]
    fn test_truncate_with_offset() {
        let buf = buffer_mut![0xFF, 0xFF];
        let mut bit_buf = BitBufferMut::from_buffer(buf, 4, 12);

        assert_eq!(bit_buf.len(), 12);

        // Truncate to 8 bits
        bit_buf.truncate(8);
        assert_eq!(bit_buf.len(), 8);

        // Truncate to 3 bits
        bit_buf.truncate(3);
        assert_eq!(bit_buf.len(), 3);

        // Truncating to larger length should be no-op
        bit_buf.truncate(10);
        assert_eq!(bit_buf.len(), 3);
    }

    #[test]
    fn test_capacity_with_offset() {
        // Use exact buffer size to test capacity calculation
        let buf = buffer_mut![0, 0, 0, 0, 0, 0, 0, 0, 0, 0]; // Exactly 10 bytes = 80 bits
        let bit_buf = BitBufferMut::from_buffer(buf, 5, 0);

        // Capacity should be at least buffer length minus offset
        // (may be more due to allocator rounding)
        assert!(bit_buf.capacity() >= 75);
        // And should account for offset
        assert_eq!(bit_buf.capacity() % 8, (80 - 5) % 8);
    }

    #[test]
    fn test_reserve_with_offset() {
        // Use exact buffer to test reserve
        let buf = buffer_mut![0, 0]; // Exactly 2 bytes = 16 bits
        let mut bit_buf = BitBufferMut::from_buffer(buf, 3, 0);

        // Current capacity should be at least 13 bits (16 - 3)
        let initial_capacity = bit_buf.capacity();
        assert!(initial_capacity >= 13);

        // Reserve 20 more bits (need total of offset 3 + len 0 + additional 20 = 23 bits)
        bit_buf.reserve(20);

        // Should now have at least 20 bits of capacity
        assert!(bit_buf.capacity() >= 20);
    }

    #[test]
    fn test_freeze_with_offset() {
        let buf = buffer_mut![0b11110000, 0b00001111];
        let mut bit_buf = BitBufferMut::from_buffer(buf, 4, 8);

        // Set some bits
        bit_buf.set(0);
        bit_buf.set(7);

        // Freeze and verify offset is preserved
        let frozen = bit_buf.freeze();
        assert_eq!(frozen.offset(), 4);
        assert_eq!(frozen.len(), 8);

        // Verify values through frozen buffer
        assert!(frozen.value(0));
        assert!(frozen.value(7));
    }

    #[cfg_attr(miri, ignore)] // bitvec crate uses a ptr cast that Miri doesn't support
    #[test]
    fn test_append_buffer_with_offsets() {
        // Create source buffer with offset
        let source = bitbuffer![false, false, true, true, false, true];

        // Create destination buffer with offset
        let buf = BufferMut::zeroed(4);
        let mut dest = BitBufferMut::from_buffer(buf, 3, 0);

        // Append 2 initial bits
        dest.append(true);
        dest.append(false);

        // Append the source buffer
        dest.append_buffer(&source);

        assert_eq!(dest.len(), 8);
        assert!(dest.value(0)); // Our first append
        assert!(!dest.value(1)); // Our second append
        assert!(!dest.value(2)); // From source[0]
        assert!(!dest.value(3)); // From source[1]
        assert!(dest.value(4)); // From source[2]
        assert!(dest.value(5)); // From source[3]
        assert!(!dest.value(6)); // From source[4]
        assert!(dest.value(7)); // From source[5]
    }

    #[test]
    fn test_set_unset_unchecked_with_offset() {
        let buf = BufferMut::zeroed(3);
        let mut bit_buf = BitBufferMut::from_buffer(buf, 7, 10);

        unsafe {
            bit_buf.set_unchecked(0);
            bit_buf.set_unchecked(5);
            bit_buf.set_unchecked(9);
        }

        assert!(bit_buf.value(0));
        assert!(bit_buf.value(5));
        assert!(bit_buf.value(9));

        unsafe {
            bit_buf.unset_unchecked(5);
        }

        assert!(!bit_buf.value(5));
    }

    #[test]
    fn test_value_unchecked_with_offset() {
        let buf = buffer_mut![0b11110000, 0b00001111];
        let bit_buf = BitBufferMut::from_buffer(buf, 4, 8);

        unsafe {
            // First 4 bits of logical buffer come from bits 4-7 of first byte (all 1s)
            assert!(bit_buf.value_unchecked(0));
            assert!(bit_buf.value_unchecked(3));

            // Next 4 bits come from bits 0-3 of second byte (all 1s)
            assert!(bit_buf.value_unchecked(4));
            assert!(bit_buf.value_unchecked(7));
        }
    }

    #[test]
    fn test_append_alternating_with_offset() {
        let buf = BufferMut::zeroed(4);
        let mut bit_buf = BitBufferMut::from_buffer(buf, 2, 0);

        // Append alternating pattern across byte boundaries
        for i in 0..20 {
            bit_buf.append(i % 2 == 0);
        }

        assert_eq!(bit_buf.len(), 20);
        for i in 0..20 {
            assert_eq!(bit_buf.value(i), i % 2 == 0);
        }
    }

    #[test]
    fn test_new_set_new_unset() {
        let set_buf = bitbuffer_mut![true; 10];
        let unset_buf = bitbuffer_mut![false; 10];

        for i in 0..10 {
            assert!(set_buf.value(i));
            assert!(!unset_buf.value(i));
        }

        assert_eq!(set_buf.len(), 10);
        assert_eq!(unset_buf.len(), 10);
    }

    #[test]
    fn test_append_n_false_with_offset() {
        let buf = BufferMut::zeroed(4);
        let mut bit_buf = BitBufferMut::from_buffer(buf, 5, 0);

        bit_buf.append_n(false, 15);

        assert_eq!(bit_buf.len(), 15);
        for i in 0..15 {
            assert!(!bit_buf.value(i));
        }
    }

    #[test]
    fn test_append_n_true_with_offset() {
        let buf = BufferMut::zeroed(4);
        let mut bit_buf = BitBufferMut::from_buffer(buf, 5, 0);

        bit_buf.append_n(true, 15);

        assert_eq!(bit_buf.len(), 15);
        for i in 0..15 {
            assert!(bit_buf.value(i));
        }
    }

    #[test]
    fn test_mixed_operations_with_offset() {
        // Complex test combining multiple operations with offset
        let buf = BufferMut::zeroed(5);
        let mut bit_buf = BitBufferMut::from_buffer(buf, 3, 0);

        // Append some bits
        bit_buf.append_n(true, 5);
        bit_buf.append_n(false, 3);
        bit_buf.append(true);

        assert_eq!(bit_buf.len(), 9);

        // Set and unset
        bit_buf.set(6); // Was false, now true
        bit_buf.unset(2); // Was true, now false

        // Verify
        assert!(bit_buf.value(0));
        assert!(bit_buf.value(1));
        assert!(!bit_buf.value(2)); // Unset
        assert!(bit_buf.value(3));
        assert!(bit_buf.value(4));
        assert!(!bit_buf.value(5));
        assert!(bit_buf.value(6)); // Set
        assert!(!bit_buf.value(7));
        assert!(bit_buf.value(8));

        // Truncate
        bit_buf.truncate(6);
        assert_eq!(bit_buf.len(), 6);

        // Freeze and verify offset preserved
        let frozen = bit_buf.freeze();
        assert_eq!(frozen.offset(), 3);
        assert_eq!(frozen.len(), 6);
    }

    #[test]
    fn test_from_iterator_with_incorrect_size_hint() {
        // This test catches a bug where FromIterator assumed the upper bound
        // from size_hint was accurate. The iterator contract allows the actual
        // count to exceed the upper bound, which could cause UB if we used
        // append_unchecked beyond the allocated capacity.

        // Custom iterator that lies about its size hint.
        struct LyingIterator {
            values: Vec<bool>,
            index: usize,
        }

        impl Iterator for LyingIterator {
            type Item = bool;

            fn next(&mut self) -> Option<Self::Item> {
                (self.index < self.values.len()).then(|| {
                    let val = self.values[self.index];
                    self.index += 1;
                    val
                })
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                // Deliberately return an incorrect upper bound that's smaller
                // than the actual number of elements we'll yield.
                let remaining = self.values.len() - self.index;
                let lower = remaining.min(5); // Correct lower bound (but capped).
                let upper = Some(5); // Incorrect upper bound - we actually have more!
                (lower, upper)
            }
        }

        // Create an iterator that claims to have at most 5 elements but actually has 10.
        let lying_iter = LyingIterator {
            values: vec![
                true, false, true, false, true, false, true, false, true, false,
            ],
            index: 0,
        };

        // Collect the iterator. This would cause UB in the old implementation
        // if it trusted the upper bound and used append_unchecked beyond capacity.
        let bit_buf: BitBufferMut = lying_iter.collect();

        // Verify all 10 elements were collected correctly.
        assert_eq!(bit_buf.len(), 10);
        for i in 0..10 {
            assert_eq!(bit_buf.value(i), i % 2 == 0);
        }
    }
}
