// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(unsafe_op_in_unsafe_fn)]

use crate::filter::slice::neon::neon_u8::SHUFFLE_MASKS;
use std::arch::aarch64::*;
use std::ptr;
use vortex_buffer::BitView;

/// For u16 types, we perform a similar strategy to u8 with a few key differences.
///
/// When it comes to shuffling u16 elements, we load u16x8 values into a uint8x8x2 vector. This
/// is represented internally as two uint8x8 vectors, where the first vector contains the lower
/// bytes of each u16 element, and the second vector contains the higher bytes.
///
/// Since the interleaved load is done in a single instruction, we then use the same u8 shuffle
/// masks to shuffle both the lower and higher byte vectors separately. Finally, we use an
/// interleaved store to write the compressed u16 elements back to memory.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(super) unsafe fn filter_neon_u16<const NB: usize>(data: *mut u16, mask: &BitView<NB>) {
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
                            let values = vld2_u8(read_ptr.cast());
                            let count = byte.count_ones() as usize;
                            let shuffle_vec = vld1_u8(SHUFFLE_MASKS[byte as usize].as_ptr());
                            // Shuffle both lower and higher byte vectors separately.
                            let compressed = uint8x8x2_t {
                                0: vtbl1_u8(values.0, shuffle_vec),
                                1: vtbl1_u8(values.1, shuffle_vec),
                            };

                            // Store all compressed values, and only increment write_ptr by count.
                            vst2_u8(write_ptr.cast(), compressed);
                            write_ptr = write_ptr.add(count);
                        }
                    }
                    read_ptr = read_ptr.add(8);
                }
            }
        }
    }
}
