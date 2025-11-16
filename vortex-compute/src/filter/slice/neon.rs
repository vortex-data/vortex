// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// There's simply too many intrinsics to bother with this!
#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
use std::arch::is_aarch64_feature_detected;
use vortex_buffer::BitView;
use vortex_error::vortex_panic;

/// Benchmark wrapper for [`filter_neon`].
#[doc(hidden)]
#[cfg(feature = "bench")]
#[cfg(target_arch = "aarch64")]
#[inline(never)]
pub fn bench_filter_neon<const NB: usize, T: Copy>(bit_view: &BitView<NB>, slice: &mut [T]) {
    if !is_aarch64_feature_detected!("neon") {
        vortex_panic!("NEON not detected on this CPU");
    }
    unsafe { filter_neon(slice, bit_view) }
}

/// Filters the given slice of items in place according to the provided BitView using neon
/// (non-SIMD) code.
///
/// The caller *should* handle where the BitView has zero or full true counts to avoid unnecessary
/// work.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(super) unsafe fn filter_neon<const NB: usize, T: Copy>(slice: &mut [T], mask: &BitView<NB>) {
    assert_eq!(
        slice.len(),
        BitView::<NB>::N,
        "Slice length must match BitView length"
    );
    match size_of::<T>() {
        1 => filter_neon_u82(slice.as_mut_ptr() as *mut u8, mask),
        2 => filter_neon_u16(slice.as_mut_ptr() as *mut u16, mask),
        4 => filter_neon_u32(slice.as_mut_ptr() as *mut u32, mask),
        8 => filter_neon_u64(slice.as_mut_ptr() as *mut u64, mask),
        _ => {
            // Fallback to scalar for non-standard sizes
            super::scalar::filter_scalar(slice, mask)
        }
    }
}

/// Most optimized NEON compress for u8
///
/// Key insights:
/// - Process 8 u8s at a time (one u64 word = 8 bytes)
/// - Each u64 in the mask gives us 8 bytes worth of bits
/// - Use vtbl1 for efficient byte-level shuffling
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn filter_neon_u82<const NB: usize>(data: *mut u8, mask: &BitView<NB>) {
    let mut write_ptr = data;
    let mut read_ptr = data.cast_const();

    // Iterate over u16 (16 bits = 16xu8 = 128 bit SIMD)
    for word in mask.iter_sized::<u16>() {
        match word {
            0u16 => {
                // All bits clear - skip
            }
            0xFFFFu16 => {
                // All bits set - fast path: copy entire vector
                let values = vld1q_u8(read_ptr.cast());
                vst1q_u8(write_ptr, values);
                write_ptr = write_ptr.add(16);
            }
            _ => {
                // Generate shuffle mask on the fly and compress
                let count = word.count_ones() as usize;

                if count == 1 {
                    // Single element - ultra fast path
                    let bit_pos = word.trailing_zeros() as usize;
                    write_ptr.write(read_ptr.add(bit_pos).read());
                    write_ptr = write_ptr.add(1);
                } else {
                    // Use dynamic shuffle generation
                    for i in 0..1 {
                        let lower = word.to_le_bytes()[i];
                        let shuffle_indices = generate_u8_shuffle_mask(lower);
                        let shuffle_vec = vld1_u8(shuffle_indices.as_ptr());

                        let values = vld1_u8(read_ptr.add(8 * i));
                        let compressed = vtbl1_u8(values, shuffle_vec);

                        vst1_u8(write_ptr, compressed);
                        write_ptr = write_ptr.add(lower.count_ones() as usize);
                    }
                }
            }
        }
        read_ptr = read_ptr.add(16);
    }
}
/// Generate shuffle mask for u8 compress using branchless bit manipulation
///
/// This generates the indices on-the-fly rather than using a lookup table.
/// For an 8-bit mask, this is faster than fetching from a 256-entry table
/// due to cache pressure and branch prediction.
#[inline(always)]
fn generate_u8_shuffle_mask(mask: u8) -> [u8; 8] {
    let mut shuffle = [0u8; 8];
    let mut write_pos = 0;

    // Unrolled loop for maximum performance
    // The compiler should optimize this to branchless code
    if mask & 0x01 != 0 {
        shuffle[write_pos] = 0;
        write_pos += 1;
    }
    if mask & 0x02 != 0 {
        shuffle[write_pos] = 1;
        write_pos += 1;
    }
    if mask & 0x04 != 0 {
        shuffle[write_pos] = 2;
        write_pos += 1;
    }
    if mask & 0x08 != 0 {
        shuffle[write_pos] = 3;
        write_pos += 1;
    }
    if mask & 0x10 != 0 {
        shuffle[write_pos] = 4;
        write_pos += 1;
    }
    if mask & 0x20 != 0 {
        shuffle[write_pos] = 5;
        write_pos += 1;
    }
    if mask & 0x40 != 0 {
        shuffle[write_pos] = 6;
        write_pos += 1;
    }
    if mask & 0x80 != 0 {
        shuffle[write_pos] = 7;
        write_pos += 1;
    }

    // Fill remaining with 0 (safe, won't be read)
    while write_pos < 8 {
        shuffle[write_pos] = 0;
        write_pos += 1;
    }

    shuffle
}

/// Alternative: Use a small 256-entry LUT (2KB total)
/// This might be faster on some microarchitectures with good L1 cache
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn filter_neon_u8_lut<const NB: usize>(data: *mut u8, mask: &BitView<NB>) -> usize {
    let mut write_idx = 0;

    // 256 entries × 8 bytes = 2KB lookup table
    static SHUFFLE_MASKS: [[u8; 8]; 256] = generate_u8_lut();

    for (word_idx, word) in mask.iter_words().enumerate() {
        if word == 0 {
            continue;
        }

        let base_idx = word_idx * 8;

        for byte_in_word in 0..8 {
            let chunk_start = base_idx * 8 + byte_in_word * 8;
            let chunk_mask = ((word >> (byte_in_word * 8)) & 0xFF) as u8;

            if chunk_mask == 0 {
                continue;
            }

            let values = vld1_u8(data.add(chunk_start));

            if chunk_mask == 0xFF {
                vst1_u8(data.add(write_idx), values);
                write_idx += 8;
            } else {
                let count = chunk_mask.count_ones() as usize;
                let shuffle_vec = vld1_u8(SHUFFLE_MASKS[chunk_mask as usize].as_ptr());
                let compressed = vtbl1_u8(values, shuffle_vec);

                if count >= 4 {
                    vst1_u8(data.add(write_idx), compressed);
                } else {
                    // Store individual bytes for sparse case
                    let mut temp = [0u8; 8];
                    vst1_u8(temp.as_mut_ptr(), compressed);
                    for i in 0..count {
                        *data.add(write_idx + i) = temp[i];
                    }
                }
                write_idx += count;
            }
        }
    }

    write_idx
}

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

/// Hybrid approach: use LUT for common patterns, generate for rare ones
/// This gives best of both worlds for realistic selectivity patterns
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn filter_neon_u8_hybrid<const NB: usize>(data: *mut u8, mask: &BitView<NB>) -> usize {
    let mut write_idx = 0;

    // Only cache the 16 most common patterns (all combinations of lower 4 bits)
    static SHUFFLE_MASKS_COMMON: [[u8; 8]; 16] = generate_u8_lut_4bit();

    for (word_idx, word) in mask.iter_words().enumerate() {
        if word == 0 {
            continue;
        }

        let base_idx = word_idx * 8;

        for byte_in_word in 0..8 {
            let chunk_start = base_idx * 8 + byte_in_word * 8;
            let chunk_mask = ((word >> (byte_in_word * 8)) & 0xFF) as u8;

            if chunk_mask == 0 {
                continue;
            }

            let values = vld1_u8(data.add(chunk_start));

            if chunk_mask == 0xFF {
                vst1_u8(data.add(write_idx), values);
                write_idx += 8;
            } else {
                let count = chunk_mask.count_ones() as usize;

                // Use LUT for patterns with only lower 4 bits set
                let shuffle_vec = if chunk_mask <= 0x0F {
                    vld1_u8(SHUFFLE_MASKS_COMMON[chunk_mask as usize].as_ptr())
                } else {
                    // Generate on-the-fly for less common patterns
                    let shuffle_indices = generate_u8_shuffle_mask(chunk_mask);
                    vld1_u8(shuffle_indices.as_ptr())
                };

                let compressed = vtbl1_u8(values, shuffle_vec);

                if count >= 4 {
                    vst1_u8(data.add(write_idx), compressed);
                } else {
                    let mut temp = [0u8; 8];
                    vst1_u8(temp.as_mut_ptr(), compressed);
                    for i in 0..count {
                        *data.add(write_idx + i) = temp[i];
                    }
                }
                write_idx += count;
            }
        }
    }

    write_idx
}

const fn generate_u8_lut_4bit() -> [[u8; 8]; 16] {
    let mut lut = [[0u8; 8]; 16];
    let mut mask = 0;

    while mask < 16 {
        let mut write_pos = 0;
        let mut bit = 0;

        while bit < 8 {
            if (mask >> bit) & 1 != 0 {
                lut[mask][write_pos] = bit;
                write_pos += 1;
            }
            bit += 1;
        }

        while write_pos < 8 {
            lut[mask][write_pos] = 0;
            write_pos += 1;
        }

        mask += 1;
    }

    lut
}

/// NEON in-place filter for u8 elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn filter_neon_u8<const NB: usize>(data: *mut u8, mask: &BitView<NB>) {
    let mut write_idx = 0;

    for (word_idx, word) in mask.iter_words().enumerate() {
        if word == 0 {
            continue;
        }

        let base_idx = word_idx * (usize::BITS as usize);

        // Process in chunks of 16 bytes (128 bits)
        for chunk in 0..(usize::BITS as usize / 16) {
            let chunk_start = base_idx + chunk * 16;
            let chunk_mask = (word >> (chunk * 16)) & 0xFFFF;

            if chunk_mask == 0 {
                continue;
            }

            // Load 16 u8 values
            let values = vld1q_u8(data.add(chunk_start));

            if chunk_mask == 0xFFFF {
                // All bits set - fast path: copy entire vector
                vst1q_u8(data.add(write_idx), values);
                write_idx += 16;
            } else {
                // Selective copy based on mask bits
                let mut temp = [0u8; 16];
                vst1q_u8(temp.as_mut_ptr(), values);

                for i in 0..16 {
                    if (chunk_mask >> i) & 1 != 0 {
                        *data.add(write_idx) = temp[i];
                        write_idx += 1;
                    }
                }
            }
        }
    }
}

/// NEON in-place filter for u16 elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn filter_neon_u16<const NB: usize>(data: *mut u16, mask: &BitView<NB>) {
    let mut write_idx = 0;

    for (word_idx, word) in mask.iter_words().enumerate() {
        if word == 0 {
            continue;
        }

        let base_idx = word_idx * (usize::BITS as usize);

        // Process in chunks of 8 u16s (128 bits)
        for chunk in 0..(usize::BITS as usize / 8) {
            let chunk_start = base_idx + chunk * 8;
            let chunk_mask = (word >> (chunk * 8)) & 0xFF;

            if chunk_mask == 0 {
                continue;
            }

            // Load 8 u16 values
            let values = vld1q_u16(data.add(chunk_start));

            if chunk_mask == 0xFF {
                // All bits set - fast path
                vst1q_u16(data.add(write_idx), values);
                write_idx += 8;
            } else {
                // Extract selected elements
                let mut temp = [0u16; 8];
                vst1q_u16(temp.as_mut_ptr(), values);

                for i in 0..8 {
                    if (chunk_mask >> i) & 1 != 0 {
                        *data.add(write_idx) = temp[i];
                        write_idx += 1;
                    }
                }
            }
        }
    }
}

/// NEON in-place filter for u32 elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn filter_neon_u32<const NB: usize>(data: *mut u32, mask: &BitView<NB>) {
    let mut write_idx = 0;

    for (word_idx, word) in mask.iter_words().enumerate() {
        if word == 0 {
            continue;
        }

        let base_idx = word_idx * (usize::BITS as usize);

        // Process in chunks of 4 u32s (128 bits)
        for chunk in 0..(usize::BITS as usize / 4) {
            let chunk_start = base_idx + chunk * 4;
            let chunk_mask = (word >> (chunk * 4)) & 0xF;

            if chunk_mask == 0 {
                continue;
            }

            // Load 4 u32 values
            let values = vld1q_u32(data.add(chunk_start));

            if chunk_mask == 0xF {
                // All bits set - fast path
                vst1q_u32(data.add(write_idx), values);
                write_idx += 4;
            } else {
                // Extract lane by lane based on mask
                if chunk_mask & 0x1 != 0 {
                    *data.add(write_idx) = vgetq_lane_u32::<0>(values);
                    write_idx += 1;
                }
                if chunk_mask & 0x2 != 0 {
                    *data.add(write_idx) = vgetq_lane_u32::<1>(values);
                    write_idx += 1;
                }
                if chunk_mask & 0x4 != 0 {
                    *data.add(write_idx) = vgetq_lane_u32::<2>(values);
                    write_idx += 1;
                }
                if chunk_mask & 0x8 != 0 {
                    *data.add(write_idx) = vgetq_lane_u32::<3>(values);
                    write_idx += 1;
                }
            }
        }
    }
}

/// NEON in-place filter for u64 elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn filter_neon_u64<const NB: usize>(data: *mut u64, mask: &BitView<NB>) {
    let mut write_idx = 0;

    for (word_idx, word) in mask.iter_words().enumerate() {
        if word == 0 {
            continue;
        }

        let base_idx = word_idx * (usize::BITS as usize);

        // Process in chunks of 2 u64s (128 bits)
        for chunk in 0..(usize::BITS as usize / 2) {
            let chunk_start = base_idx + chunk * 2;
            let chunk_mask = (word >> (chunk * 2)) & 0x3;

            if chunk_mask == 0 {
                continue;
            }

            // Load 2 u64 values
            let values = vld1q_u64(data.add(chunk_start));

            if chunk_mask == 0x3 {
                // Both bits set - fast path
                vst1q_u64(data.add(write_idx), values);
                write_idx += 2;
            } else {
                // Extract lane by lane
                if chunk_mask & 0x1 != 0 {
                    *data.add(write_idx) = vgetq_lane_u64::<0>(values);
                    write_idx += 1;
                }
                if chunk_mask & 0x2 != 0 {
                    *data.add(write_idx) = vgetq_lane_u64::<1>(values);
                    write_idx += 1;
                }
            }
        }
    }
}

/// Optimized NEON in-place filter using TBL for u32
/// Uses table lookup to rearrange elements efficiently
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn filter_neon_u32_tbl<const NB: usize>(data: *mut u32, mask: &BitView<NB>) {
    let mut write_idx = 0;

    // Pre-computed shuffle masks for all 16 possible 4-bit patterns
    static SHUFFLE_MASKS: [[u8; 16]; 16] = generate_shuffle_masks();

    for (word_idx, word) in mask.iter_words().enumerate() {
        if word == 0 {
            continue;
        }

        let base_idx = word_idx * (usize::BITS as usize);

        // Process chunks of 4 u32s (128 bits = 16 bytes)
        for chunk in 0..(usize::BITS as usize / 4) {
            let chunk_start = base_idx + chunk * 4;
            let chunk_mask = ((word >> (chunk * 4)) & 0xF) as usize;

            if chunk_mask == 0 {
                continue;
            }

            if chunk_mask == 0xF {
                // All selected - fast path: just copy
                let values = vld1q_u32(data.add(chunk_start));
                vst1q_u32(data.add(write_idx), values);
                write_idx += 4;
            } else {
                // Load 16 bytes (4 u32s) as bytes for table lookup
                let values_bytes = vld1q_u8(data.add(chunk_start) as *const u8);

                // Load pre-computed shuffle mask
                let shuffle_mask = vld1q_u8(SHUFFLE_MASKS[chunk_mask].as_ptr());

                // Perform table lookup to rearrange bytes
                let values_low = vget_low_u8(values_bytes);
                let values_high = vget_high_u8(values_bytes);

                let shuffled_low = vtbl2_u8(
                    uint8x8x2_t(values_low, values_high),
                    vget_low_u8(shuffle_mask),
                );
                let shuffled_high = vtbl2_u8(
                    uint8x8x2_t(values_low, values_high),
                    vget_high_u8(shuffle_mask),
                );

                let shuffled = vcombine_u8(shuffled_low, shuffled_high);

                // Store compacted result
                let count = chunk_mask.count_ones() as usize;
                vst1q_u8(data.add(write_idx) as *mut u8, shuffled);
                write_idx += count;
            }
        }
    }
}

/// Helper function to generate shuffle masks at compile time
const fn generate_shuffle_masks() -> [[u8; 16]; 16] {
    let mut masks = [[0u8; 16]; 16];
    let mut pattern = 0;
    while pattern < 16 {
        let mut write_pos = 0;
        let mut bit = 0;
        while bit < 4 {
            if (pattern >> bit) & 1 != 0 {
                // Each u32 is 4 bytes
                let src_offset = bit * 4;
                masks[pattern][write_pos * 4 + 0] = src_offset + 0;
                masks[pattern][write_pos * 4 + 1] = src_offset + 1;
                masks[pattern][write_pos * 4 + 2] = src_offset + 2;
                masks[pattern][write_pos * 4 + 3] = src_offset + 3;
                write_pos += 1;
            }
            bit += 1;
        }
        // Fill remaining with 0xff (out of bounds reads return 0)
        while write_pos < 4 {
            masks[pattern][write_pos * 4 + 0] = 0xff;
            masks[pattern][write_pos * 4 + 1] = 0xff;
            masks[pattern][write_pos * 4 + 2] = 0xff;
            masks[pattern][write_pos * 4 + 3] = 0xff;
            write_pos += 1;
        }
        pattern += 1;
    }
    masks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[cfg(target_feature = "neon")]
    fn test_filter_in_place_u32() {
        const NB: usize = 128; // 1024 bits
        const N: usize = NB * 8;

        let view = BitView::<NB>::with_prefix(512);
        let mut data: Vec<u32> = (0..N).map(|i| i as u32).collect();

        unsafe { filter_neon(&mut data, &view) };

        assert_eq!(
            &data[..view.true_count()],
            &(0..512).collect::<Vec<u32>>()[..]
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[cfg(target_feature = "neon")]
    fn test_filter_in_place_sparse() {
        const NB: usize = 256; // 2048 bits
        const N: usize = NB * 8;

        let mut bits = [0u8; NB];
        // Set every 16th bit
        for i in (0..N).step_by(16) {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            bits[byte_idx] |= 1 << bit_idx;
        }
        let view = BitView::<NB>::new(&bits);

        let mut data: Vec<u64> = (0..N).map(|i| i as u64).collect();
        unsafe { filter_neon(&mut data, &view) };

        assert_eq!(
            &data[..view.true_count()],
            &(0..N as u64).step_by(16).collect::<Vec<u64>>()[..]
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[cfg(target_feature = "neon")]
    fn test_filter_in_place_all_true() {
        const NB: usize = 64; // 512 bits
        const N: usize = NB * 8;

        let view = BitView::<NB>::all_true();
        let mut data: Vec<u16> = (0..N).map(|i| i as u16).collect();
        let original = data.clone();

        unsafe { filter_neon(&mut data, &view) };

        assert_eq!(&data[..view.true_count()], &original[..]);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[cfg(target_feature = "neon")]
    fn test_filter_in_place_all_false() {
        const NB: usize = 64; // 512 bits
        const N: usize = NB * 8;

        let view = BitView::<NB>::all_false();
        let mut data: Vec<u8> = (0..N).map(|i| i as u8).collect();

        unsafe { filter_neon(&mut data, &view) };
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[cfg(target_feature = "neon")]
    fn test_filter_slice_wrapper() {
        const NB: usize = 128;
        const N: usize = NB * 8;

        let view = BitView::<NB>::with_prefix(256);
        let mut data: Vec<u32> = (0..N).map(|i| i as u32).collect();

        unsafe { filter_neon(&mut data, &view) };

        assert_eq!(&data[..256], &(0..256).collect::<Vec<u32>>()[..]);
    }
}
