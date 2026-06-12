// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Packed bitmaps that can be used to store boolean values.
//!
//! This module provides a wrapper on top of the `Buffer` type to store mutable and immutable
//! bitsets. The bitsets are stored in little-endian order, meaning that the least significant bit
//! of the first byte is the first bit in the bitset.
#[cfg(feature = "arrow")]
mod arrow;
mod buf;
mod buf_mut;
mod count_ones;
mod macros;
mod meta;
mod ops;
mod select;
mod view;

pub use arrow_buffer::bit_chunk_iterator::BitChunkIterator;
pub use arrow_buffer::bit_chunk_iterator::BitChunks;
pub use arrow_buffer::bit_chunk_iterator::UnalignedBitChunk;
pub use arrow_buffer::bit_chunk_iterator::UnalignedBitChunkIterator;
pub use arrow_buffer::bit_iterator::BitIndexIterator;
pub use arrow_buffer::bit_iterator::BitIterator;
pub use arrow_buffer::bit_iterator::BitSliceIterator;
pub use buf::*;
pub use buf_mut::*;
pub use meta::*;
pub use view::*;

/// Packs up to 64 boolean values into a little-endian `u64` word.
#[inline]
pub fn collect_bool_word<F>(len: usize, mut f: F) -> u64
where
    F: FnMut(usize) -> bool,
{
    assert!(len <= 64, "cannot pack {len} bits into a u64 word");

    let mut packed = 0;
    for bit_idx in 0..len {
        packed |= (f(bit_idx) as u64) << bit_idx;
    }
    packed
}

/// Pack `len` boolean values returned by `f` into the prefix of `words`, LSB-first,
/// 64 bits per `u64`. `words` must have capacity for at least `len.div_ceil(64)` entries.
///
/// Writes via `=` (not `|=`), so the destination need not be zero-initialised.
#[inline]
pub fn collect_bool_words<F>(words: &mut [u64], len: usize, mut f: F)
where
    F: FnMut(usize) -> bool,
{
    let num_words = len.div_ceil(64);
    assert!(
        words.len() >= num_words,
        "words slice has {} entries, need at least {num_words}",
        words.len(),
    );

    let full = len / 64;
    let remainder = len % 64;

    for word_idx in 0..full {
        let offset = word_idx * 64;
        words[word_idx] = collect_bool_word(64, |bit_idx| f(offset + bit_idx));
    }

    if remainder != 0 {
        let offset = full * 64;
        words[full] = collect_bool_word(remainder, |bit_idx| f(offset + bit_idx));
    }
}

/// Splice a packed word `w` (whose bits above the highest valid bit are zero) into
/// `words` at the given bit position.
///
/// The destination word at `bit_offset / 64` is OR'd, preserving any bits below
/// `bit_offset % 64`. When `w` has high bits that spill into the next word, those
/// bits are *assigned* (not OR'd) — so callers must ensure that next slot is zero
/// (e.g. via `BufferMut::zeroed`).
///
/// `words.len()` need only cover the slots `w` actually writes to: skipping the
/// spillover when its bits are all zero means a tail that fits entirely in the
/// leading word never touches `words[dest_word + 1]`.
#[inline]
pub fn splice_word_at_bit(words: &mut [u64], bit_offset: usize, word: u64) {
    let dest_word = bit_offset / 64;
    let bit_in_word = bit_offset % 64;
    words[dest_word] |= word << bit_in_word;
    if bit_in_word != 0 {
        let high = word >> (64 - bit_in_word);
        if high != 0 {
            words[dest_word + 1] = high;
        }
    }
}

/// Pack `len` boolean values returned by `f` into `words` starting at bit position
/// `bit_offset`, LSB-first.
///
/// Composes [`collect_bool_word`] (pack up to 64 bools into a u64) with
/// [`splice_word_at_bit`] (merge the packed word into the destination via shift-OR).
///
/// `words` must have at least `(bit_offset + len).div_ceil(64)` entries; see
/// [`splice_word_at_bit`] for zero-init requirements on words above the cursor.
#[inline]
pub fn pack_bools_into_words<F>(words: &mut [u64], bit_offset: usize, len: usize, mut f: F)
where
    F: FnMut(usize) -> bool,
{
    if len == 0 {
        return;
    }
    let num_words = (bit_offset + len).div_ceil(64);
    assert!(
        words.len() >= num_words,
        "words slice has {} entries, need at least {num_words}",
        words.len(),
    );

    let mut done = 0;
    while len - done >= 64 {
        let word = collect_bool_word(64, |bit| f(done + bit));
        splice_word_at_bit(words, bit_offset + done, word);
        done += 64;
    }
    let tail = len - done;
    if tail > 0 {
        let word = collect_bool_word(tail, |bit| f(done + bit));
        splice_word_at_bit(words, bit_offset + done, word);
    }
}

/// Get the bit value at `index` out of `buf`.
///
/// # Panics
///
/// Panics if `index` is not between 0 and length of `buf * 8`.
#[inline(always)]
pub fn get_bit(buf: &[u8], index: usize) -> bool {
    buf[index / 8] & (1 << (index % 8)) != 0
}

/// Get the bit value at `index` out of `buf` without bounds checking.
///
/// # Safety
///
/// `index` must be between 0 and length of `buf * 8`.
#[inline(always)]
pub unsafe fn get_bit_unchecked(buf: *const u8, index: usize) -> bool {
    (unsafe { *buf.add(index / 8) } & (1 << (index % 8))) != 0
}

/// Set the bit value at `index` in `buf` without bounds checking.
///
/// # Safety
///
/// `index` must be between 0 and length of `buf * 8`.
#[inline(always)]
pub unsafe fn set_bit_unchecked(buf: *mut u8, index: usize) {
    unsafe { *buf.add(index / 8) |= 1 << (index % 8) };
}

/// Unset the bit value at `index` in `buf` without bounds checking.
///
/// # Safety
///
/// `index` must be between 0 and length of `buf * 8`.
#[inline(always)]
pub unsafe fn unset_bit_unchecked(buf: *mut u8, index: usize) {
    unsafe { *buf.add(index / 8) &= !(1 << (index % 8)) };
}

#[cfg(test)]
mod tests {
    use super::collect_bool_word;
    use super::pack_bools_into_words;

    #[test]
    fn collect_bool_word_packs_lsb_first() {
        let word = collect_bool_word(5, |idx| idx.is_multiple_of(2));
        assert_eq!(word, 0b10101);
    }

    #[test]
    fn collect_bool_word_empty() {
        assert_eq!(collect_bool_word(0, |_| true), 0);
    }

    #[test]
    #[should_panic(expected = "cannot pack 65 bits into a u64 word")]
    fn collect_bool_word_rejects_too_many_bits() {
        let _ = collect_bool_word(65, |_| true);
    }

    fn pack(bit_offset: usize, len: usize, f: impl Fn(usize) -> bool) -> Vec<bool> {
        let num_words = (bit_offset + len).div_ceil(64);
        let mut words = vec![0u64; num_words];
        pack_bools_into_words(&mut words, bit_offset, len, &f);
        (0..bit_offset + len)
            .map(|i| (words[i / 64] >> (i % 64)) & 1 == 1)
            .collect()
    }

    #[test]
    fn pack_bools_aligned_multi_word_with_tail() {
        let bits = pack(0, 130, |i| i.is_multiple_of(3));
        for i in 0..130 {
            assert_eq!(bits[i], i.is_multiple_of(3), "bit {i}");
        }
    }

    #[test]
    fn pack_bools_unaligned_crossing_words() {
        let bits = pack(40, 200, |i| i.is_multiple_of(7));
        assert!(bits[..40].iter().all(|&b| !b));
        for i in 0..200 {
            assert_eq!(bits[40 + i], i.is_multiple_of(7), "bit {}", 40 + i);
        }
    }

    #[test]
    fn pack_bools_preserves_low_bits_of_leading_word() {
        let mut words = vec![0u64; 2];
        words[0] = 0b11111;
        pack_bools_into_words(&mut words, 5, 70, |_| true);
        for i in 0..5 {
            assert_eq!((words[0] >> i) & 1, 1, "preserved bit {i}");
        }
        for i in 5..75 {
            assert_eq!((words[i / 64] >> (i % 64)) & 1, 1, "extended bit {i}");
        }
    }
}
