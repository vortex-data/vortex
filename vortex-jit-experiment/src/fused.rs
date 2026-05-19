// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The **fused** path: one function, no temp buffer.
//!
//! This is what a perfect JIT would emit for the specific
//! `(bit_width, threshold)` pair. We hand-write it in Rust here so we have
//! a known-good reference to compare the JIT output against.
//!
//! The crucial observation: there is no intermediate `[u32; 1024]`. Each
//! value is unpacked, immediately compared, and only the result bit is
//! stored. The boundary is gone — the two kernels' bodies are interleaved
//! in a single loop nest.

use crate::MASK_WORDS;
use crate::pack::unpack_one;

/// Fused unpack + compare > k → bitmap.
///
/// `#[inline(never)]` to keep it a fair fight with `composed::unpack_then_compare`
/// when we measure code size and timing.
#[inline(never)]
pub fn unpack_compare_fused(packed: &[u32], bit_width: u32, k: u32, mask: &mut [u64; MASK_WORDS]) {
    for word in 0..MASK_WORDS {
        let mut bits: u64 = 0;
        for bit in 0..64 {
            let i = word * 64 + bit;
            let v = unpack_one(packed, i, bit_width);
            bits |= u64::from(v > k) << bit;
        }
        mask[word] = bits;
    }
}
