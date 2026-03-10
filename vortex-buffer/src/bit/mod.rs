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
mod macros;
mod ops;

pub use arrow_buffer::bit_chunk_iterator::BitChunkIterator;
pub use arrow_buffer::bit_chunk_iterator::BitChunks;
pub use arrow_buffer::bit_chunk_iterator::UnalignedBitChunk;
pub use arrow_buffer::bit_chunk_iterator::UnalignedBitChunkIterator;
pub use arrow_buffer::bit_iterator::BitIndexIterator;
pub use arrow_buffer::bit_iterator::BitIterator;
pub use arrow_buffer::bit_iterator::BitSliceIterator;
pub use buf::*;
pub use buf_mut::*;

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

/// Sets all bits in the bit-range `[start_bit, end_bit)` of `slice` to `value`.
///
/// This operates directly on a byte slice where bits are stored in little-endian order.
/// The caller must ensure that the slice is large enough to hold bits up to `end_bit`.
///
/// # Panics
///
/// Panics if `start_bit > end_bit` or if the slice is too small.
#[inline(always)]
pub fn fill_bits(slice: &mut [u8], start_bit: usize, end_bit: usize, value: bool) {
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
