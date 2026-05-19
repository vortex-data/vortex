// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The **composed** path: two kernels, joined by a temp buffer.
//!
//! This is the analog of what vx does today: an unpack kernel writes 1024
//! values into a stack buffer, then a compare kernel reads from it.
//!
//! Each kernel is `#[inline(never)]` so they're truly separate units of code
//! that cannot see each other's bodies — the same way real dispatched kernels
//! work behind a vtable. Without these annotations, LLVM would inline both
//! into the caller and effectively fuse them for free, hiding the cost we're
//! trying to measure.

use crate::CHUNK_SIZE;
use crate::MASK_WORDS;
use crate::pack::unpack_one;

/// Kernel 1: unpack a packed chunk into 1024 `u32`s.
///
/// Contract (the "left half" of the boundary):
/// - `packed` holds enough words to encode 1024 values at `bit_width`.
/// - On return, every element of `dst` is initialised.
#[inline(never)]
pub fn unpack_kernel(packed: &[u32], bit_width: u32, dst: &mut [u32; CHUNK_SIZE]) {
    for i in 0..CHUNK_SIZE {
        dst[i] = unpack_one(packed, i, bit_width);
    }
}

/// Kernel 2: compare each element against `k` and write a bitmap.
///
/// Contract (the "right half" of the boundary):
/// - Reads exactly `CHUNK_SIZE` elements from `src`.
/// - Writes a 1024-bit mask packed LSB-first into 16 `u64` words.
#[inline(never)]
pub fn compare_kernel(src: &[u32; CHUNK_SIZE], k: u32, dst: &mut [u64; MASK_WORDS]) {
    for word in 0..MASK_WORDS {
        let mut bits: u64 = 0;
        for bit in 0..64 {
            let i = word * 64 + bit;
            bits |= u64::from(src[i] > k) << bit;
        }
        dst[word] = bits;
    }
}

/// Public entry point: cross the boundary explicitly.
///
/// Notice the named `tmp` — that 4 KiB stack buffer **is** the boundary.
pub fn unpack_then_compare(packed: &[u32], bit_width: u32, k: u32, mask: &mut [u64; MASK_WORDS]) {
    let mut tmp = [0u32; CHUNK_SIZE];
    unpack_kernel(packed, bit_width, &mut tmp);
    compare_kernel(&tmp, k, mask);
}
