// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(unsafe_op_in_unsafe_fn)]

use std::arch::aarch64::*;
use std::ptr;
use vortex_buffer::BitView;

/// For u8 types, we use NEON's tbl lookup instruction to perform a shuffle based on a pre-computed
/// lookup table (LUT).
///
/// In theory, we could use vqtbl1q_u8 for 16-byte vectors, but that would require a 65KB LUT
/// (256 entries × 16 bytes each), which is too large for practical use as it thrashes
/// the CPU cache.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(super) unsafe fn filter_neon_u8<const NB: usize>(data: *mut u8, mask: &BitView<NB>) {
    let mut read_ptr = data;
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
                    match byte {
                        0u8 => {
                            // Skip empty chunks
                        }
                        0xFF => {
                            // All bits set - fast path
                            ptr::copy(read_ptr, write_ptr, 8);
                            write_ptr = write_ptr.add(8);
                        }
                        _ => {
                            // Finally, use the lookup table to compress selected elements
                            // Load uint8x8 values and compress them using the lookup table.
                            let values = vld1_u8(read_ptr);
                            let count = byte.count_ones() as usize;
                            let shuffle_vec = vld1_u8(SHUFFLE_MASKS[byte as usize].as_ptr());
                            let compressed = vtbl1_u8(values, shuffle_vec);

                            // Store all compressed values, and only increment write_ptr by count.
                            vst1_u8(write_ptr, compressed);
                            write_ptr = write_ptr.add(count);
                        }
                    }
                    read_ptr = read_ptr.add(8);
                }
            }
        }
    }
}

/// Pre-computed shuffle masks for an 8-bit selection mask.
/// 256 entries × 8 bytes = 2KB lookup table
pub(super) static SHUFFLE_MASKS: [[u8; 8]; 256] = generate_u8_lut();

/// Generate 256-entry lookup table at compile time
const fn generate_u8_lut() -> [[u8; 8]; 256] {
    let mut lut = [[0u8; 8]; 256];
    let mut mask = 0;

    while mask < 256 {
        let mut write_pos = 0;
        let mut bit = 0;

        while bit < 8 {
            if (mask >> bit) & 1 != 0 {
                lut[mask][write_pos] = bit;
                write_pos += 1;
            }
            bit += 1;
        }

        // Fill rest with 0
        while write_pos < 8 {
            lut[mask][write_pos] = 0;
            write_pos += 1;
        }

        mask += 1;
    }

    lut
}
