// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Packed bitmaps that can be used to store boolean values.
//!
//! This module provides a wrapper on top of the `Buffer` type to store mutable and immutable
//! bitsets. The bitsets are stored in little-endian order, meaning that the least significant bit
//! of the first byte is the first bit in the bitset.
mod buf;
mod buf_mut;
mod chunk_iterator;
mod count_ones;
mod iterator;
mod macros;
mod ops;

pub use buf::*;
pub use buf_mut::*;
pub use chunk_iterator::BitChunkIterator;
pub use chunk_iterator::BitChunks;
pub use chunk_iterator::UnalignedBitChunk;
pub use chunk_iterator::UnalignedBitChunkIterator;
pub use iterator::BitIndexIterator;
pub use iterator::BitIterator;
pub use iterator::BitSliceIterator;

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
