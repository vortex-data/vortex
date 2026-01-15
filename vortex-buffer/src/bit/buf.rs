// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;
use std::ops::BitOr;
use std::ops::BitXor;
use std::ops::Bound;
use std::ops::Not;
use std::ops::RangeBounds;

use vortex_error::vortex_panic;

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
use crate::bit::get_bit_unchecked;
use crate::bit::ops::bitwise_binary_op;
use crate::bit::ops::bitwise_unary_op;
use crate::buffer;

// =============================================================================
// Select-in-word helpers
// =============================================================================

/// Lookup table for select within a byte.
/// SELECT_IN_BYTE_TABLE[byte][n] = position of nth set bit in byte (0-indexed).
/// Invalid entries (n >= popcount) are set to 8 (will cause incorrect results if used).
#[allow(clippy::cast_possible_truncation)]
static SELECT_IN_BYTE_TABLE: [[u8; 8]; 256] = {
    let mut table = [[8u8; 8]; 256];
    let mut byte = 0usize;
    while byte < 256 {
        let mut bit_pos = 0usize;
        let mut rank = 0usize;
        while bit_pos < 8 {
            if (byte >> bit_pos) & 1 == 1 {
                // SAFETY: bit_pos is always < 8, so fits in u8
                table[byte][rank] = bit_pos as u8;
                rank += 1;
            }
            bit_pos += 1;
        }
        byte += 1;
    }
    table
};

/// Find the position of the nth set bit within a u64 word (0-indexed).
///
/// This is the "select" operation within a single word.
/// Uses BMI2 pdep instruction when available (single instruction),
/// otherwise falls back to a hybrid approach.
#[inline]
fn select_in_word(word: u64, n: usize) -> usize {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    {
        // BMI2 pdep: O(1) - single instruction!
        // Deposits a 1-bit at position n into the set-bit positions of word,
        // then trailing_zeros finds where it landed.
        use std::arch::x86_64::_pdep_u64;
        unsafe { _pdep_u64(1u64 << n, word).trailing_zeros() as usize }
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    {
        // Hybrid approach: loop is faster for small n, binary search for larger n.
        // Crossover point is around n=3 based on benchmarks.
        if n <= 2 {
            select_in_word_loop(word, n)
        } else {
            select_in_word_binary_search(word, n)
        }
    }
}

/// Loop-based select: O(n) - fastest for small n (0-2).
#[inline]
fn select_in_word_loop(mut word: u64, mut n: usize) -> usize {
    loop {
        let tz = word.trailing_zeros() as usize;
        if n == 0 {
            return tz;
        }
        word &= word - 1; // Clear the lowest set bit
        n -= 1;
    }
}

/// Binary search select: O(log 64) = max 3 comparisons + table lookup.
/// Better for larger n values (3+).
#[allow(clippy::cast_possible_truncation)]
#[inline]
fn select_in_word_binary_search(word: u64, mut n: usize) -> usize {
    let mut word = word;
    let mut pos = 0usize;

    // Check lower 32 bits
    let lower_count = (word as u32).count_ones() as usize;
    if n >= lower_count {
        n -= lower_count;
        word >>= 32;
        pos += 32;
    }

    // Check lower 16 bits of remaining
    let lower_count = (word as u16).count_ones() as usize;
    if n >= lower_count {
        n -= lower_count;
        word >>= 16;
        pos += 16;
    }

    // Check lower 8 bits of remaining
    let lower_count = (word as u8).count_ones() as usize;
    if n >= lower_count {
        n -= lower_count;
        word >>= 8;
        pos += 8;
    }

    // Final 8 bits - use lookup table
    pos + SELECT_IN_BYTE_TABLE[(word as u8) as usize][n] as usize
}

/// Find the position of the nth set bit from the end of a word (0-indexed from end).
///
/// For example, if word has set bits at positions [0, 2, 4, 62], then:
/// - select_in_word_reverse(word, 0) = 62 (last set bit)
/// - select_in_word_reverse(word, 1) = 4 (second-to-last)
/// - select_in_word_reverse(word, 2) = 2
/// - select_in_word_reverse(word, 3) = 0 (first set bit)
#[inline]
fn select_in_word_reverse(word: u64, n: usize) -> usize {
    // Reverse the word so the last bit becomes the first
    // Then find the nth bit from the start in the reversed word
    // Convert back: position in reversed = 63 - position in original
    63 - select_in_word(word.reverse_bits(), n)
}

// =============================================================================
// Block8 processing macros for optimized select
// =============================================================================

/// Process 8 chunks in forward direction, returning if target found.
macro_rules! block8_forward {
    ($remaining:ident, $c:expr, $base:expr) => {{
        let [c0, c1, c2, c3, c4, c5, c6, c7] = $c;

        // 8 independent popcounts - CPU can execute in parallel
        let pop0 = c0.count_ones() as usize;
        let pop1 = c1.count_ones() as usize;
        let pop2 = c2.count_ones() as usize;
        let pop3 = c3.count_ones() as usize;
        let pop4 = c4.count_ones() as usize;
        let pop5 = c5.count_ones() as usize;
        let pop6 = c6.count_ones() as usize;
        let pop7 = c7.count_ones() as usize;

        let block_pop = pop0 + pop1 + pop2 + pop3 + pop4 + pop5 + pop6 + pop7;

        if $remaining < block_pop {
            // Narrow down within block
            if $remaining < pop0 {
                return $base + select_in_word(c0, $remaining);
            }
            $remaining -= pop0;
            if $remaining < pop1 {
                return $base + 64 + select_in_word(c1, $remaining);
            }
            $remaining -= pop1;
            if $remaining < pop2 {
                return $base + 128 + select_in_word(c2, $remaining);
            }
            $remaining -= pop2;
            if $remaining < pop3 {
                return $base + 192 + select_in_word(c3, $remaining);
            }
            $remaining -= pop3;
            if $remaining < pop4 {
                return $base + 256 + select_in_word(c4, $remaining);
            }
            $remaining -= pop4;
            if $remaining < pop5 {
                return $base + 320 + select_in_word(c5, $remaining);
            }
            $remaining -= pop5;
            if $remaining < pop6 {
                return $base + 384 + select_in_word(c6, $remaining);
            }
            $remaining -= pop6;
            return $base + 448 + select_in_word(c7, $remaining);
        }
        $remaining -= block_pop;
    }};
}

/// Process 8 chunks in reverse direction, returning if target found.
macro_rules! block8_reverse {
    ($remaining:ident, $c:expr, $base:expr) => {{
        let [c0, c1, c2, c3, c4, c5, c6, c7] = $c;

        let pop0 = c0.count_ones() as usize;
        let pop1 = c1.count_ones() as usize;
        let pop2 = c2.count_ones() as usize;
        let pop3 = c3.count_ones() as usize;
        let pop4 = c4.count_ones() as usize;
        let pop5 = c5.count_ones() as usize;
        let pop6 = c6.count_ones() as usize;
        let pop7 = c7.count_ones() as usize;

        let block_pop = pop0 + pop1 + pop2 + pop3 + pop4 + pop5 + pop6 + pop7;

        if $remaining < block_pop {
            if $remaining < pop0 {
                return $base + 448 + select_in_word_reverse(c0, $remaining);
            }
            $remaining -= pop0;
            if $remaining < pop1 {
                return $base + 384 + select_in_word_reverse(c1, $remaining);
            }
            $remaining -= pop1;
            if $remaining < pop2 {
                return $base + 320 + select_in_word_reverse(c2, $remaining);
            }
            $remaining -= pop2;
            if $remaining < pop3 {
                return $base + 256 + select_in_word_reverse(c3, $remaining);
            }
            $remaining -= pop3;
            if $remaining < pop4 {
                return $base + 192 + select_in_word_reverse(c4, $remaining);
            }
            $remaining -= pop4;
            if $remaining < pop5 {
                return $base + 128 + select_in_word_reverse(c5, $remaining);
            }
            $remaining -= pop5;
            if $remaining < pop6 {
                return $base + 64 + select_in_word_reverse(c6, $remaining);
            }
            $remaining -= pop6;
            return $base + select_in_word_reverse(c7, $remaining);
        }
        $remaining -= block_pop;
    }};
}

/// Load remainder bits into a u64 (for the partial chunk at the end).
#[inline]
fn load_remainder(bytes: &[u8], num_chunks: usize, rem_bits: usize) -> u64 {
    let start = num_chunks * 8;
    let mut buf8 = [0u8; 8];
    let avail = (bytes.len() - start).min(8);
    buf8[..avail].copy_from_slice(&bytes[start..start + avail]);
    u64::from_le_bytes(buf8) & ((1u64 << rem_bits) - 1)
}

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
        let buffer = buffer.slice(byte_offset..);

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

    /// Invokes `f` with indexes `0..len` collecting the boolean results into a new [`BitBuffer`].
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
        self.unaligned_chunks().count_ones()
    }

    /// Get the number of unset bits in the buffer.
    pub fn false_count(&self) -> usize {
        self.len - self.true_count()
    }

    /// Returns the index of the nth set bit (0-indexed).
    ///
    /// This is also known as the "select" operation in succinct data structures.
    /// Uses bidirectional search (from start or end) based on target position,
    /// providing up to 60x speedup for high percentiles.
    ///
    /// # Panics
    ///
    /// Panics if `n >= true_count()`.
    ///
    /// # Example
    ///
    /// ```
    /// use vortex_buffer::BitBuffer;
    ///
    /// let buf = BitBuffer::from_iter([false, true, false, true, true]);
    /// assert_eq!(buf.select(0), 1); // 1st set bit is at index 1
    /// assert_eq!(buf.select(1), 3); // 2nd set bit is at index 3
    /// assert_eq!(buf.select(2), 4); // 3rd set bit is at index 4
    /// ```
    #[inline]
    pub fn select(&self, n: usize) -> usize {
        let true_count = self.true_count();
        self.select_with_true_count(n, true_count)
    }

    /// Returns the index of the nth set bit (0-indexed), using a pre-computed true_count.
    ///
    /// This is more efficient when the true_count is already known (e.g., cached in Mask).
    /// Uses bidirectional search (from start or end) based on target position,
    /// providing up to 60x speedup for high percentiles.
    ///
    /// # Panics
    ///
    /// Panics if `n >= true_count` or if `true_count` doesn't match the actual count.
    #[allow(clippy::collapsible_if)]
    #[inline]
    pub fn select_with_true_count(&self, n: usize, true_count: usize) -> usize {
        if n >= true_count {
            vortex_panic!("select({n}) out of bounds: buffer has only {true_count} set bits");
        }

        // Bidirectional: search from start or end based on target position
        if n < true_count / 2 {
            self.select_forward(n)
        } else {
            self.select_reverse(true_count - 1 - n)
        }
    }

    /// Forward select using block8 processing for instruction-level parallelism.
    // TODO(perf): The unaligned path is ~2x slower than aligned due to UnalignedBitChunk
    // abstraction overhead. Could optimize by using direct pointer access similar to aligned case.
    #[allow(clippy::collapsible_if, clippy::many_single_char_names)]
    fn select_forward(&self, n: usize) -> usize {
        let mut remaining = n;

        if self.offset == 0 {
            // Aligned: direct pointer access
            let bytes = self.buffer.as_slice();
            let len = self.len;
            let ptr = bytes.as_ptr() as *const u64;
            let num_chunks = len / 64;
            let mut i = 0;

            // Process 8 chunks at a time for ILP
            while i + 8 <= num_chunks {
                let chunks = [
                    unsafe { ptr.add(i).read_unaligned() },
                    unsafe { ptr.add(i + 1).read_unaligned() },
                    unsafe { ptr.add(i + 2).read_unaligned() },
                    unsafe { ptr.add(i + 3).read_unaligned() },
                    unsafe { ptr.add(i + 4).read_unaligned() },
                    unsafe { ptr.add(i + 5).read_unaligned() },
                    unsafe { ptr.add(i + 6).read_unaligned() },
                    unsafe { ptr.add(i + 7).read_unaligned() },
                ];
                block8_forward!(remaining, chunks, i * 64);
                i += 8;
            }

            // Handle remaining chunks one at a time
            while i < num_chunks {
                let chunk = unsafe { ptr.add(i).read_unaligned() };
                let pop = chunk.count_ones() as usize;
                if remaining < pop {
                    return i * 64 + select_in_word(chunk, remaining);
                }
                remaining -= pop;
                i += 1;
            }

            // Handle remainder bits
            let rem_bits = len % 64;
            if rem_bits > 0 {
                let rem = load_remainder(bytes, num_chunks, rem_bits);
                if remaining < rem.count_ones() as usize {
                    return num_chunks * 64 + select_in_word(rem, remaining);
                }
            }
        } else {
            // Unaligned: use UnalignedBitChunk for aligned middle section
            let unaligned = self.unaligned_chunks();
            let lead_padding = unaligned.lead_padding();
            let mut bit_idx = 0usize;

            // Handle prefix
            if let Some(prefix) = unaligned.prefix() {
                let pop = prefix.count_ones() as usize;
                if remaining < pop {
                    return select_in_word(prefix, remaining) - lead_padding;
                }
                remaining -= pop;
                bit_idx = 64 - lead_padding;
            }

            // Process aligned middle chunks with block8
            let chunks = unaligned.chunks();
            let num_chunks = chunks.len();
            let mut i = 0;

            while i + 8 <= num_chunks {
                let c = [
                    chunks[i],
                    chunks[i + 1],
                    chunks[i + 2],
                    chunks[i + 3],
                    chunks[i + 4],
                    chunks[i + 5],
                    chunks[i + 6],
                    chunks[i + 7],
                ];
                block8_forward!(remaining, c, bit_idx);
                bit_idx += 512;
                i += 8;
            }

            while i < num_chunks {
                let chunk = chunks[i];
                let pop = chunk.count_ones() as usize;
                if remaining < pop {
                    return bit_idx + select_in_word(chunk, remaining);
                }
                remaining -= pop;
                bit_idx += 64;
                i += 1;
            }

            // Handle suffix
            if let Some(suffix) = unaligned.suffix() {
                if remaining < suffix.count_ones() as usize {
                    return bit_idx + select_in_word(suffix, remaining);
                }
            }
        }

        // Should be unreachable if bounds check was done
        vortex_panic!("select_forward: n out of bounds");
    }

    /// Reverse select using block8 processing (search from end).
    // TODO(perf): Same as select_forward - unaligned path could use direct pointer access.
    #[allow(clippy::collapsible_if, clippy::many_single_char_names)]
    fn select_reverse(&self, n_from_end: usize) -> usize {
        let mut remaining = n_from_end;

        if self.offset == 0 {
            // Aligned: direct pointer access
            let bytes = self.buffer.as_slice();
            let len = self.len;
            let ptr = bytes.as_ptr() as *const u64;
            let num_chunks = len / 64;
            let rem_bits = len % 64;

            // Handle remainder bits first (they're at the end)
            if rem_bits > 0 {
                let rem = load_remainder(bytes, num_chunks, rem_bits);
                let pop = rem.count_ones() as usize;
                if remaining < pop {
                    return num_chunks * 64 + select_in_word_reverse(rem, remaining);
                }
                remaining -= pop;
            }

            // Process chunks in reverse, 8 at a time
            let mut i = num_chunks;
            while i >= 8 {
                let chunks = [
                    unsafe { ptr.add(i - 1).read_unaligned() },
                    unsafe { ptr.add(i - 2).read_unaligned() },
                    unsafe { ptr.add(i - 3).read_unaligned() },
                    unsafe { ptr.add(i - 4).read_unaligned() },
                    unsafe { ptr.add(i - 5).read_unaligned() },
                    unsafe { ptr.add(i - 6).read_unaligned() },
                    unsafe { ptr.add(i - 7).read_unaligned() },
                    unsafe { ptr.add(i - 8).read_unaligned() },
                ];
                block8_reverse!(remaining, chunks, (i - 8) * 64);
                i -= 8;
            }

            // Handle remaining chunks
            while i > 0 {
                i -= 1;
                let chunk = unsafe { ptr.add(i).read_unaligned() };
                let pop = chunk.count_ones() as usize;
                if remaining < pop {
                    return i * 64 + select_in_word_reverse(chunk, remaining);
                }
                remaining -= pop;
            }
        } else {
            // Unaligned: use UnalignedBitChunk
            let unaligned = self.unaligned_chunks();
            let lead_padding = unaligned.lead_padding();
            let chunks = unaligned.chunks();
            let num_chunks = chunks.len();

            let prefix_bits = if unaligned.prefix().is_some() {
                64 - lead_padding
            } else {
                0
            };
            let suffix_start = prefix_bits + num_chunks * 64;

            // Handle suffix first
            if let Some(suffix) = unaligned.suffix() {
                let pop = suffix.count_ones() as usize;
                if remaining < pop {
                    return suffix_start + select_in_word_reverse(suffix, remaining);
                }
                remaining -= pop;
            }

            // Process middle chunks in reverse with block8
            let mut i = num_chunks;
            while i >= 8 {
                let c = [
                    chunks[i - 1],
                    chunks[i - 2],
                    chunks[i - 3],
                    chunks[i - 4],
                    chunks[i - 5],
                    chunks[i - 6],
                    chunks[i - 7],
                    chunks[i - 8],
                ];
                block8_reverse!(remaining, c, prefix_bits + (i - 8) * 64);
                i -= 8;
            }

            while i > 0 {
                i -= 1;
                let chunk = chunks[i];
                let pop = chunk.count_ones() as usize;
                if remaining < pop {
                    return prefix_bits + i * 64 + select_in_word_reverse(chunk, remaining);
                }
                remaining -= pop;
            }

            // Handle prefix last
            if let Some(prefix) = unaligned.prefix() {
                if remaining < prefix.count_ones() as usize {
                    return select_in_word_reverse(prefix, remaining) - lead_padding;
                }
            }
        }

        // Should be unreachable if bounds check was done
        vortex_panic!("select_reverse: n out of bounds");
    }

    /// Returns the index of the nth set bit using the set_slices iterator.
    ///
    /// This is an alternative implementation optimized for data with long runs
    /// of consecutive 1s or 0s (correlated data). It skips entire runs at a time.
    ///
    /// # Panics
    ///
    /// Panics if `n >= true_count()`.
    pub fn select_via_slices(&self, n: usize) -> usize {
        let mut remaining = n;

        // set_slices returns (start, end) pairs representing [start, end) ranges
        for (start, end) in self.set_slices() {
            let slice_len = end - start;
            if remaining < slice_len {
                return start + remaining;
            }
            remaining -= slice_len;
        }

        vortex_panic!(
            "select_via_slices({n}) out of bounds: buffer has only {} set bits",
            self.true_count()
        );
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
        bitwise_unary_op(self, |a| a)
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

    /// Get a mutable version of this `BitBuffer` along with bit offset in the first byte.
    ///
    /// If the caller doesn't hold only reference to the underlying buffer, a copy is created.
    /// The second value of the tuple is a bit_offset of the first value in the first byte
    pub fn into_mut(self) -> BitBufferMut {
        let (offset, len, inner) = self.into_inner();
        // TODO(robert): if we are copying here we could strip offset bits
        BitBufferMut::from_buffer(inner.into_mut(), offset, len)
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

impl BitAnd<BitBuffer> for &BitBuffer {
    type Output = BitBuffer;

    fn bitand(self, rhs: BitBuffer) -> Self::Output {
        self.bitand(&rhs)
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
    fn test_select_basic(#[case] len: usize) {
        // Create a buffer with alternating bits
        let buf = BitBuffer::collect_bool(len, |i| i % 2 == 0);

        // Verify select returns correct indices
        let true_count = buf.true_count();
        for n in 0..true_count {
            let selected_idx = buf.select(n);
            // The nth set bit should be at position n * 2 (every other bit is set)
            assert_eq!(selected_idx, n * 2, "select({n}) should be {}", n * 2);
        }
    }

    #[test]
    fn test_select_sparse() {
        // Create a buffer with only a few bits set
        let buf = BitBuffer::collect_bool(1000, |i| i == 10 || i == 500 || i == 999);

        assert_eq!(buf.select(0), 10);
        assert_eq!(buf.select(1), 500);
        assert_eq!(buf.select(2), 999);
    }

    #[test]
    fn test_select_dense() {
        // Create a buffer with all bits set
        let buf = BitBuffer::new_set(100);

        for n in 0..100 {
            assert_eq!(buf.select(n), n);
        }
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_select_out_of_bounds() {
        let buf = BitBuffer::collect_bool(100, |i| i % 10 == 0); // 10 set bits
        buf.select(10); // Should panic - only 10 bits (indices 0-9)
    }

    #[rstest]
    #[case(100)]
    #[case(1000)]
    #[case(10000)]
    fn test_select_via_slices_matches_select(#[case] len: usize) {
        // Test with runs of 1s and 0s (correlated data)
        let buf = BitBuffer::collect_bool(len, |i| (i / 64) % 2 == 0);

        let true_count = buf.true_count();
        for n in 0..true_count {
            let expected = buf.select(n);
            let actual = buf.select_via_slices(n);
            assert_eq!(actual, expected, "select_via_slices({n}) mismatch");
        }
    }

    #[test]
    fn test_select_via_slices_all_ones() {
        let buf = BitBuffer::new_set(1000);
        for n in 0..1000 {
            assert_eq!(buf.select_via_slices(n), n);
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

    #[rstest]
    #[case(100)]
    #[case(1000)]
    #[case(10000)]
    fn test_select_with_true_count(#[case] len: usize) {
        let buf = BitBuffer::collect_bool(len, |i| i % 10 == 0);
        let true_count = buf.true_count();

        for n in 0..true_count {
            let expected = buf.select(n);
            let actual = buf.select_with_true_count(n, true_count);
            assert_eq!(actual, expected, "select_with_true_count({n}) mismatch");
        }
    }

    #[rstest]
    #[case(1, 1000)] // offset of 1
    #[case(3, 1000)] // offset of 3
    #[case(7, 1000)] // offset of 7
    #[case(13, 1000)] // offset spanning multiple bytes
    #[case(63, 1000)] // offset close to chunk boundary
    fn test_select_unaligned(#[case] offset: usize, #[case] len: usize) {
        let total_bits = offset + len;
        let buf = BitBuffer::collect_bool(total_bits, |i| i % 10 == 0);
        let sliced = buf.slice(offset..offset + len);

        // Verify the sliced buffer has correct offset
        assert!(sliced.offset() != 0 || offset % 8 == 0);

        let true_count = sliced.true_count();
        for n in 0..true_count {
            let selected_idx = sliced.select(n);
            // Verify the selected index is actually a set bit
            assert!(
                sliced.value(selected_idx),
                "select({n}) = {} but bit is not set",
                selected_idx
            );

            // Count the number of set bits before selected_idx
            let count_before: usize = (0..selected_idx).filter(|&i| sliced.value(i)).count();
            assert_eq!(
                count_before, n,
                "select({n}) = {} has wrong rank",
                selected_idx
            );
        }
    }

    #[rstest]
    #[case(100_000, 2)] // Low percentile - forward search
    #[case(100_000, 50)] // Middle - either direction
    #[case(100_000, 98)] // High percentile - reverse search
    fn test_select_bidirectional(#[case] len: usize, #[case] pct: usize) {
        let buf = BitBuffer::collect_bool(len, |i| i % 10 == 0);
        let true_count = buf.true_count();
        let target = true_count * pct / 100;

        // Compute expected value using set_indices iterator
        let expected_idx = buf.set_indices().nth(target).unwrap();
        let actual = buf.select(target);
        assert_eq!(actual, expected_idx, "select at {}% mismatch", pct);
    }

    #[rstest]
    #[case(1, 100_000)] // 1-bit offset
    #[case(7, 100_000)] // 7-bit offset (almost full byte)
    #[case(33, 100_000)] // Multi-byte offset
    fn test_select_unaligned_bidirectional(#[case] offset: usize, #[case] len: usize) {
        let total_bits = offset + len;
        let buf = BitBuffer::collect_bool(total_bits, |i| i % 10 == 0);
        let sliced = buf.slice(offset..offset + len);

        let true_count = sliced.true_count();

        // Test low percentile (forward)
        let target_low = true_count * 2 / 100;
        let expected_low = sliced.set_indices().nth(target_low).unwrap();
        assert_eq!(sliced.select(target_low), expected_low);

        // Test high percentile (reverse)
        let target_high = true_count * 98 / 100;
        let expected_high = sliced.set_indices().nth(target_high).unwrap();
        assert_eq!(sliced.select(target_high), expected_high);
    }

    #[test]
    fn test_select_large_buffer_block8() {
        // Test a buffer large enough to exercise block8 processing (>512 chunks = >32768 bits)
        let len = 100_000;
        let buf = BitBuffer::collect_bool(len, |i| i % 7 == 0);
        let true_count = buf.true_count();

        // Test various positions across the buffer
        for pct in [0, 10, 25, 50, 75, 90, 100] {
            let target = if pct == 100 {
                true_count - 1
            } else {
                true_count * pct / 100
            };
            let expected = buf.set_indices().nth(target).unwrap();
            let actual = buf.select(target);
            assert_eq!(actual, expected, "select at {}% mismatch", pct);
        }
    }

    // Additional comprehensive tests for select_in_word_reverse
    #[test]
    fn test_select_in_word_reverse_basic() {
        // Word with bits at 0, 2, 4, 6 (4 set bits)
        let word: u64 = 0b01010101;
        // From end: 0=6, 1=4, 2=2, 3=0
        assert_eq!(super::select_in_word_reverse(word, 0), 6);
        assert_eq!(super::select_in_word_reverse(word, 1), 4);
        assert_eq!(super::select_in_word_reverse(word, 2), 2);
        assert_eq!(super::select_in_word_reverse(word, 3), 0);
    }

    #[test]
    fn test_select_in_word_reverse_single_bit() {
        // Single bit at position 32
        let word: u64 = 1u64 << 32;
        assert_eq!(super::select_in_word_reverse(word, 0), 32);
    }

    #[test]
    fn test_select_in_word_reverse_all_ones() {
        let word: u64 = u64::MAX;
        // Last bit is at 63, first is at 0
        assert_eq!(super::select_in_word_reverse(word, 0), 63);
        assert_eq!(super::select_in_word_reverse(word, 63), 0);
        assert_eq!(super::select_in_word_reverse(word, 31), 32);
    }

    #[test]
    fn test_select_in_word_reverse_high_bits() {
        // Bits set at 60, 61, 62, 63
        let word: u64 = 0xF000_0000_0000_0000;
        assert_eq!(super::select_in_word_reverse(word, 0), 63);
        assert_eq!(super::select_in_word_reverse(word, 1), 62);
        assert_eq!(super::select_in_word_reverse(word, 2), 61);
        assert_eq!(super::select_in_word_reverse(word, 3), 60);
    }

    #[rstest]
    #[case(1)] // Single set bit
    #[case(10)] // Sparse
    #[case(32)] // Half
    #[case(63)] // Almost all
    #[case(64)] // All
    fn test_select_consistency_forward_reverse(#[case] density_percent: usize) {
        let len = 10000;
        let buf = BitBuffer::collect_bool(len, |i| i % (100 / density_percent.max(1)) == 0);
        let true_count = buf.true_count();

        if true_count == 0 {
            return;
        }

        // Test that forward and reverse select produce consistent results
        for n in 0..true_count.min(100) {
            let forward_result = buf.select(n);
            let reverse_n = true_count - 1 - n;
            let reverse_result = buf.select(reverse_n);

            // Both should be valid set bits
            assert!(buf.value(forward_result), "forward select({n}) invalid");
            assert!(
                buf.value(reverse_result),
                "reverse select({reverse_n}) invalid"
            );
        }
    }

    #[rstest]
    #[case(511)] // Just under 8 chunks
    #[case(512)] // Exactly 8 chunks
    #[case(513)] // Just over 8 chunks
    #[case(1023)] // Just under 16 chunks
    #[case(1024)] // Exactly 16 chunks
    fn test_select_block8_boundary(#[case] num_chunks: usize) {
        let len = num_chunks * 64;
        let buf = BitBuffer::collect_bool(len, |i| i % 5 == 0);
        let true_count = buf.true_count();

        // Test all positions to catch boundary errors
        for n in 0..true_count {
            let result = buf.select(n);
            assert!(buf.value(result), "select({n}) = {} not a set bit", result);
            let rank: usize = (0..result).filter(|&i| buf.value(i)).count();
            assert_eq!(rank, n, "select({n}) = {} has wrong rank {}", result, rank);
        }
    }

    #[rstest]
    #[case(0)] // No offset
    #[case(1)] // 1-bit offset
    #[case(7)] // 7-bit offset (max within byte)
    #[case(8)] // Byte-aligned offset
    #[case(63)] // Almost chunk-aligned
    #[case(64)] // Chunk-aligned
    #[case(65)] // Just past chunk boundary
    fn test_select_various_offsets_comprehensive(#[case] offset: usize) {
        let total = offset + 1000;
        let buf = BitBuffer::collect_bool(total, |i| i % 3 == 0);
        let sliced = buf.slice(offset..total);

        let true_count = sliced.true_count();
        for n in 0..true_count {
            let result = sliced.select(n);
            assert!(
                sliced.value(result),
                "offset={}, select({n}) = {} not set",
                offset,
                result
            );
        }
    }

    #[test]
    fn test_select_first_and_last() {
        let len = 10000;
        let buf = BitBuffer::collect_bool(len, |i| i % 100 == 0);
        let true_count = buf.true_count();

        // First set bit
        assert_eq!(buf.select(0), 0);

        // Last set bit
        let last_idx = buf.select(true_count - 1);
        assert_eq!(last_idx, 9900);
    }

    #[test]
    fn test_select_single_bit_buffer() {
        // Buffer with exactly one bit set
        let buf = BitBuffer::collect_bool(1000, |i| i == 500);
        assert_eq!(buf.true_count(), 1);
        assert_eq!(buf.select(0), 500);
    }

    #[test]
    fn test_select_two_bits_far_apart() {
        // Two bits set far apart
        let buf = BitBuffer::collect_bool(10000, |i| i == 0 || i == 9999);
        assert_eq!(buf.true_count(), 2);
        assert_eq!(buf.select(0), 0);
        assert_eq!(buf.select(1), 9999);
    }

    #[test]
    fn test_select_run_patterns() {
        // Runs of 64 ones followed by 64 zeros
        let buf = BitBuffer::collect_bool(1024, |i| (i / 64) % 2 == 0);
        let true_count = buf.true_count();

        // Verify all selects are correct
        let expected: Vec<usize> = buf.set_indices().collect();
        for (n, &expected_idx) in expected.iter().enumerate() {
            assert_eq!(buf.select(n), expected_idx, "select({n}) mismatch");
        }
        assert_eq!(expected.len(), true_count);
    }

    #[test]
    fn test_select_with_true_count_matches_select() {
        let buf = BitBuffer::collect_bool(5000, |i| i % 11 == 0);
        let true_count = buf.true_count();

        for n in 0..true_count {
            let via_select = buf.select(n);
            let via_with_count = buf.select_with_true_count(n, true_count);
            assert_eq!(via_select, via_with_count, "mismatch at n={n}");
        }
    }

    #[rstest]
    #[case(100, 10)]
    #[case(1000, 100)]
    #[case(10000, 1000)]
    fn test_select_exhaustive_small_buffers(#[case] len: usize, #[case] density: usize) {
        let buf = BitBuffer::collect_bool(len, |i| i % density == 0);
        let expected: Vec<usize> = buf.set_indices().collect();

        // Exhaustively test all positions
        for (n, &expected_idx) in expected.iter().enumerate() {
            let actual = buf.select(n);
            assert_eq!(
                actual, expected_idx,
                "len={}, density={}, select({n}) expected {} got {}",
                len, density, expected_idx, actual
            );
        }
    }

    #[test]
    fn test_select_remainder_bits() {
        // Buffer that doesn't align to 64-bit boundary
        for len in [65, 100, 127, 128, 129, 255, 256, 257] {
            let buf = BitBuffer::collect_bool(len, |i| i % 4 == 0);
            let expected: Vec<usize> = buf.set_indices().collect();

            for (n, &expected_idx) in expected.iter().enumerate() {
                let actual = buf.select(n);
                assert_eq!(actual, expected_idx, "len={}, select({n}) mismatch", len);
            }
        }
    }

    #[test]
    fn test_select_alternating_bytes() {
        // Pattern that alternates by byte to catch byte-level bugs
        let buf = BitBuffer::collect_bool(1024, |i| (i / 8) % 2 == 0);
        let expected: Vec<usize> = buf.set_indices().collect();

        for (n, &expected_idx) in expected.iter().enumerate() {
            assert_eq!(buf.select(n), expected_idx, "select({n}) mismatch");
        }
    }
}
