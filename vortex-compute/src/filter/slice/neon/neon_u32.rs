// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(unsafe_op_in_unsafe_fn)]

use crate::filter::slice::neon::neon_u8::SHUFFLE_MASKS;
use std::arch::aarch64::*;
use std::ptr;
use vortex_buffer::BitView;

/// For u32 values we can only look at 4 values at a time (128 bits).
/// Therefore, we have a very manageable 16 possible bitmask combinations (0..15) and therefore
/// avoid the need for large lookup tables.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(super) unsafe fn filter_neon_u32<const NB: usize>(data: *mut u32, mask: &BitView<NB>) {
    let mut read_ptr = data.cast_const();
    let mut write_ptr = data;

    for word in mask.iter_words() {
        match word {
            0usize => {
                // Skip empty chunks
                read_ptr = read_ptr.add(usize::BITS as usize);
                continue;
            }
            usize::MAX => {
                // All bits set - fast path
                ptr::copy(read_ptr, write_ptr, usize::BITS as usize);
                read_ptr = read_ptr.add(usize::BITS as usize);
                write_ptr = write_ptr.add(usize::BITS as usize);
                continue;
            }
            _ => {
                // Otherwise, loop over the bytes of the word
                let word_le_bytes = word.to_le_bytes();
                for &byte in &word_le_bytes {
                    // Finally, use the lookup table to compress selected elements
                    // Load uint8x8 values and compress them using the lookup table.
                    // Load 4 interleaved u8 vectors
                    let values = vld4_u8(read_ptr.cast());
                    let count = byte.count_ones() as usize;
                    let shuffle_vec = vld1_u8(SHUFFLE_MASKS[byte as usize].as_ptr());

                    // Shuffle all four byte vectors separately.
                    let compressed = uint8x8x4_t {
                        0: vtbl1_u8(values.0, shuffle_vec),
                        1: vtbl1_u8(values.1, shuffle_vec),
                        2: vtbl1_u8(values.2, shuffle_vec),
                        3: vtbl1_u8(values.3, shuffle_vec),
                    };

                    // Store all compressed values, and only increment write_ptr by count.
                    vst4_u8(write_ptr.cast(), compressed);
                    write_ptr = write_ptr.add(count);
                }
            }
        }
    }
}
