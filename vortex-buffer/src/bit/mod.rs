// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Packed bitmaps that can be used to store boolean values.
//!
//! This module provides a wrapper on top of the `Buffer` type to store mutable and immutable
//! bitsets. The bitsets are stored in little-endian order, meaning that the least significant bit
//! of the first byte is the first bit in the bitset.
mod aligned;
#[cfg(feature = "arrow")]
mod arrow;
mod buf;
mod buf_mut;
mod ops;
mod unaligned;

pub use aligned::BitChunks;
pub use buf::*;
pub use buf_mut::*;

/// Get bit value at `index` out of `buf`
#[inline]
fn get_bit(buf: &[u8], index: usize) -> bool {
    buf[index / 8] & (1 << (index % 8)) != 0
}

/// Get bit value at `index` out of `buf` without bounds checking
///
/// # Safety
/// `index` must be between 0 and length of `buf`
#[inline]
unsafe fn get_bit_unchecked(buf: &[u8], index: usize) -> bool {
    let byte = unsafe { buf.get_unchecked(index / 8) };
    byte & (1 << (index % 8)) != 0
}
