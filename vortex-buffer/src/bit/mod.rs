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
mod ops;
mod select;

pub use arrow_buffer::bit_chunk_iterator::BitChunkIterator;
pub use arrow_buffer::bit_chunk_iterator::BitChunks;
pub use arrow_buffer::bit_chunk_iterator::UnalignedBitChunk;
pub use arrow_buffer::bit_chunk_iterator::UnalignedBitChunkIterator;
pub use arrow_buffer::bit_iterator::BitIndexIterator;
pub use arrow_buffer::bit_iterator::BitIterator;
pub use arrow_buffer::bit_iterator::BitSliceIterator;
pub use buf::*;
pub use buf_mut::*;

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
}
