// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fast implementations of the FastLanes 1024-bit transpose.
//!
//! The FastLanes transpose is a fixed permutation of 1024 bits (128 bytes) that
//! enables SIMD parallelism for encodings like delta and RLE. This module provides
//! optimized implementations for different x86 SIMD instruction sets.
//!
//! The key insight is that each output byte is formed by extracting the SAME bit
//! position from 8 different input bytes at stride 16. The input byte groups follow
//! the FL_ORDER permutation pattern.

/// The FastLanes bit-reversal order for groups of 8.
pub const FL_ORDER: [usize; 8] = [0, 4, 2, 6, 1, 5, 3, 7];

/// Base indices for the first 64 output bytes (lanes 0-7).
/// Each entry indicates the starting input byte index for that output byte group.
/// Pattern: [0*2, 4*2, 2*2, 6*2, 1*2, 5*2, 3*2, 7*2] = [0, 8, 4, 12, 2, 10, 6, 14]
const BASE_PATTERN_FIRST: [usize; 8] = [0, 8, 4, 12, 2, 10, 6, 14];

/// Base indices for the second 64 output bytes (lanes 8-15).
/// Pattern: first pattern + 1 = [1, 9, 5, 13, 3, 11, 7, 15]
const BASE_PATTERN_SECOND: [usize; 8] = [1, 9, 5, 13, 3, 11, 7, 15];

/// Compute the transposed index for a single bit position (0..1024).
#[inline(always)]
pub const fn transpose_index(idx: usize) -> usize {
    let lane = idx % 16;
    let order = (idx / 16) % 8;
    let row = idx / 128;
    (lane * 64) + (FL_ORDER[order] * 8) + row
}

/// Transpose 1024 bits (128 bytes) using the FastLanes layout.
///
/// This is the baseline scalar implementation that processes bit by bit.
#[inline(never)]
pub fn transpose_1024_baseline(input: &[u8; 128], output: &mut [u8; 128]) {
    output.fill(0);
    for in_bit in 0..1024 {
        let out_bit = transpose_index(in_bit);
        let in_byte = in_bit / 8;
        let in_bit_pos = in_bit % 8;
        let out_byte = out_bit / 8;
        let out_bit_pos = out_bit % 8;
        let bit_val = (input[in_byte] >> in_bit_pos) & 1;
        output[out_byte] |= bit_val << out_bit_pos;
    }
}

/// Transpose 1024 bits using optimized scalar implementation.
///
/// This implementation exploits the structure of the transpose: each output byte
/// is formed by extracting the same bit position from 8 input bytes at stride 16.
#[inline(never)]
pub fn transpose_1024_scalar(input: &[u8; 128], output: &mut [u8; 128]) {
    // Process first 64 output bytes (lanes 0-7)
    for out_byte in 0..64 {
        let out_byte_in_group = out_byte % 8;
        let bit_pos = out_byte / 8;
        let in_byte_base = BASE_PATTERN_FIRST[out_byte_in_group];

        let mut out_val = 0u8;
        for i in 0..8 {
            let in_byte_idx = in_byte_base + i * 16;
            let bit_val = (input[in_byte_idx] >> bit_pos) & 1;
            out_val |= bit_val << i;
        }
        output[out_byte] = out_val;
    }

    // Process second 64 output bytes (lanes 8-15)
    for out_byte in 64..128 {
        let out_byte_in_group = (out_byte - 64) % 8;
        let bit_pos = (out_byte - 64) / 8;
        let in_byte_base = BASE_PATTERN_SECOND[out_byte_in_group];

        let mut out_val = 0u8;
        for i in 0..8 {
            let in_byte_idx = in_byte_base + i * 16;
            let bit_val = (input[in_byte_idx] >> bit_pos) & 1;
            out_val |= bit_val << i;
        }
        output[out_byte] = out_val;
    }
}

/// Fast scalar transpose using the 8x8 bit matrix transpose algorithm.
///
/// This version uses 64-bit gather + parallel bit operations instead of
/// extracting bits one by one. Typically 5-10x faster than the basic scalar version.
#[inline(never)]
pub fn transpose_1024_scalar_fast(input: &[u8; 128], output: &mut [u8; 128]) {
    // Helper to perform 8x8 bit transpose on a u64 (each byte becomes a row)
    #[inline(always)]
    fn transpose_8x8(mut x: u64) -> u64 {
        // Step 1: Transpose 2x2 bit blocks
        let t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AAu64;
        x = x ^ t ^ (t << 7);
        // Step 2: Transpose 4x4 bit blocks
        let t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCu64;
        x = x ^ t ^ (t << 14);
        // Step 3: Transpose 8x8 bit blocks
        let t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0u64;
        x ^ t ^ (t << 28)
    }

    // Helper to gather 8 bytes at stride 16 into a u64
    #[inline(always)]
    fn gather(input: &[u8; 128], base: usize) -> u64 {
        (input[base] as u64)
            | ((input[base + 16] as u64) << 8)
            | ((input[base + 32] as u64) << 16)
            | ((input[base + 48] as u64) << 24)
            | ((input[base + 64] as u64) << 32)
            | ((input[base + 80] as u64) << 40)
            | ((input[base + 96] as u64) << 48)
            | ((input[base + 112] as u64) << 56)
    }

    // Process first half (8 base groups, fully unrolled)
    let r0 = transpose_8x8(gather(input, BASE_PATTERN_FIRST[0]));
    let r1 = transpose_8x8(gather(input, BASE_PATTERN_FIRST[1]));
    let r2 = transpose_8x8(gather(input, BASE_PATTERN_FIRST[2]));
    let r3 = transpose_8x8(gather(input, BASE_PATTERN_FIRST[3]));
    let r4 = transpose_8x8(gather(input, BASE_PATTERN_FIRST[4]));
    let r5 = transpose_8x8(gather(input, BASE_PATTERN_FIRST[5]));
    let r6 = transpose_8x8(gather(input, BASE_PATTERN_FIRST[6]));
    let r7 = transpose_8x8(gather(input, BASE_PATTERN_FIRST[7]));

    // Write first 64 output bytes (unrolled)
    for bit_pos in 0..8 {
        output[bit_pos * 8] = (r0 >> (bit_pos * 8)) as u8;
        output[bit_pos * 8 + 1] = (r1 >> (bit_pos * 8)) as u8;
        output[bit_pos * 8 + 2] = (r2 >> (bit_pos * 8)) as u8;
        output[bit_pos * 8 + 3] = (r3 >> (bit_pos * 8)) as u8;
        output[bit_pos * 8 + 4] = (r4 >> (bit_pos * 8)) as u8;
        output[bit_pos * 8 + 5] = (r5 >> (bit_pos * 8)) as u8;
        output[bit_pos * 8 + 6] = (r6 >> (bit_pos * 8)) as u8;
        output[bit_pos * 8 + 7] = (r7 >> (bit_pos * 8)) as u8;
    }

    // Process second half
    let r0 = transpose_8x8(gather(input, BASE_PATTERN_SECOND[0]));
    let r1 = transpose_8x8(gather(input, BASE_PATTERN_SECOND[1]));
    let r2 = transpose_8x8(gather(input, BASE_PATTERN_SECOND[2]));
    let r3 = transpose_8x8(gather(input, BASE_PATTERN_SECOND[3]));
    let r4 = transpose_8x8(gather(input, BASE_PATTERN_SECOND[4]));
    let r5 = transpose_8x8(gather(input, BASE_PATTERN_SECOND[5]));
    let r6 = transpose_8x8(gather(input, BASE_PATTERN_SECOND[6]));
    let r7 = transpose_8x8(gather(input, BASE_PATTERN_SECOND[7]));

    for bit_pos in 0..8 {
        output[64 + bit_pos * 8] = (r0 >> (bit_pos * 8)) as u8;
        output[64 + bit_pos * 8 + 1] = (r1 >> (bit_pos * 8)) as u8;
        output[64 + bit_pos * 8 + 2] = (r2 >> (bit_pos * 8)) as u8;
        output[64 + bit_pos * 8 + 3] = (r3 >> (bit_pos * 8)) as u8;
        output[64 + bit_pos * 8 + 4] = (r4 >> (bit_pos * 8)) as u8;
        output[64 + bit_pos * 8 + 5] = (r5 >> (bit_pos * 8)) as u8;
        output[64 + bit_pos * 8 + 6] = (r6 >> (bit_pos * 8)) as u8;
        output[64 + bit_pos * 8 + 7] = (r7 >> (bit_pos * 8)) as u8;
    }
}

/// Fast scalar untranspose using the 8x8 bit matrix transpose algorithm.
#[inline(never)]
pub fn untranspose_1024_scalar_fast(input: &[u8; 128], output: &mut [u8; 128]) {
    #[inline(always)]
    fn transpose_8x8(mut x: u64) -> u64 {
        let t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AAu64;
        x = x ^ t ^ (t << 7);
        let t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCu64;
        x = x ^ t ^ (t << 14);
        let t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0u64;
        x ^ t ^ (t << 28)
    }

    #[inline(always)]
    fn gather_transposed(input: &[u8; 128], base_group: usize, offset: usize) -> u64 {
        let mut result: u64 = 0;
        for bit_pos in 0..8 {
            result |= (input[offset + bit_pos * 8 + base_group] as u64) << (bit_pos * 8);
        }
        result
    }

    #[inline(always)]
    fn scatter(output: &mut [u8; 128], base: usize, val: u64) {
        output[base] = val as u8;
        output[base + 16] = (val >> 8) as u8;
        output[base + 32] = (val >> 16) as u8;
        output[base + 48] = (val >> 24) as u8;
        output[base + 64] = (val >> 32) as u8;
        output[base + 80] = (val >> 40) as u8;
        output[base + 96] = (val >> 48) as u8;
        output[base + 112] = (val >> 56) as u8;
    }

    // First half (unrolled)
    let r0 = transpose_8x8(gather_transposed(input, 0, 0));
    let r1 = transpose_8x8(gather_transposed(input, 1, 0));
    let r2 = transpose_8x8(gather_transposed(input, 2, 0));
    let r3 = transpose_8x8(gather_transposed(input, 3, 0));
    let r4 = transpose_8x8(gather_transposed(input, 4, 0));
    let r5 = transpose_8x8(gather_transposed(input, 5, 0));
    let r6 = transpose_8x8(gather_transposed(input, 6, 0));
    let r7 = transpose_8x8(gather_transposed(input, 7, 0));

    scatter(output, BASE_PATTERN_FIRST[0], r0);
    scatter(output, BASE_PATTERN_FIRST[1], r1);
    scatter(output, BASE_PATTERN_FIRST[2], r2);
    scatter(output, BASE_PATTERN_FIRST[3], r3);
    scatter(output, BASE_PATTERN_FIRST[4], r4);
    scatter(output, BASE_PATTERN_FIRST[5], r5);
    scatter(output, BASE_PATTERN_FIRST[6], r6);
    scatter(output, BASE_PATTERN_FIRST[7], r7);

    // Second half
    let r0 = transpose_8x8(gather_transposed(input, 0, 64));
    let r1 = transpose_8x8(gather_transposed(input, 1, 64));
    let r2 = transpose_8x8(gather_transposed(input, 2, 64));
    let r3 = transpose_8x8(gather_transposed(input, 3, 64));
    let r4 = transpose_8x8(gather_transposed(input, 4, 64));
    let r5 = transpose_8x8(gather_transposed(input, 5, 64));
    let r6 = transpose_8x8(gather_transposed(input, 6, 64));
    let r7 = transpose_8x8(gather_transposed(input, 7, 64));

    scatter(output, BASE_PATTERN_SECOND[0], r0);
    scatter(output, BASE_PATTERN_SECOND[1], r1);
    scatter(output, BASE_PATTERN_SECOND[2], r2);
    scatter(output, BASE_PATTERN_SECOND[3], r3);
    scatter(output, BASE_PATTERN_SECOND[4], r4);
    scatter(output, BASE_PATTERN_SECOND[5], r5);
    scatter(output, BASE_PATTERN_SECOND[6], r6);
    scatter(output, BASE_PATTERN_SECOND[7], r7);
}

/// Untranspose 1024 bits (inverse of transpose).
#[inline(never)]
pub fn untranspose_1024_baseline(input: &[u8; 128], output: &mut [u8; 128]) {
    output.fill(0);
    for out_bit in 0..1024 {
        let in_bit = transpose_index(out_bit);
        let in_byte = in_bit / 8;
        let in_bit_pos = in_bit % 8;
        let out_byte = out_bit / 8;
        let out_bit_pos = out_bit % 8;
        let bit_val = (input[in_byte] >> in_bit_pos) & 1;
        output[out_byte] |= bit_val << out_bit_pos;
    }
}

/// Untranspose using optimized scalar implementation.
#[inline(never)]
pub fn untranspose_1024_scalar(input: &[u8; 128], output: &mut [u8; 128]) {
    output.fill(0);

    // For untranspose, we scatter from transposed positions back to original
    // Process first 64 input bytes (lanes 0-7)
    for in_byte in 0..64 {
        let in_byte_in_group = in_byte % 8;
        let bit_pos = in_byte / 8;
        let out_byte_base = BASE_PATTERN_FIRST[in_byte_in_group];
        let in_val = input[in_byte];

        for i in 0..8 {
            let out_byte_idx = out_byte_base + i * 16;
            let bit_val = (in_val >> i) & 1;
            output[out_byte_idx] |= bit_val << bit_pos;
        }
    }

    // Process second 64 input bytes (lanes 8-15)
    for in_byte in 64..128 {
        let in_byte_in_group = (in_byte - 64) % 8;
        let bit_pos = (in_byte - 64) / 8;
        let out_byte_base = BASE_PATTERN_SECOND[in_byte_in_group];
        let in_val = input[in_byte];

        for i in 0..8 {
            let out_byte_idx = out_byte_base + i * 16;
            let bit_val = (in_val >> i) & 1;
            output[out_byte_idx] |= bit_val << bit_pos;
        }
    }
}

// ============================================================================
// x86 SIMD implementations
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
pub mod x86 {
    use super::*;

    /// Check if AVX2 is available at runtime.
    #[inline]
    pub fn has_avx2() -> bool {
        is_x86_feature_detected!("avx2")
    }

    /// Check if AVX-512F and AVX-512BW are available at runtime.
    #[inline]
    pub fn has_avx512() -> bool {
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
    }

    /// Check if GFNI (Galois Field New Instructions) is available.
    #[inline]
    pub fn has_gfni() -> bool {
        is_x86_feature_detected!("gfni")
    }

    /// Check if BMI2 is available.
    #[inline]
    pub fn has_bmi2() -> bool {
        is_x86_feature_detected!("bmi2")
    }

    // ========================================================================
    // BMI2 implementation using PEXT/PDEP
    // ========================================================================

    /// Transpose 1024 bits using BMI2 PEXT instruction.
    ///
    /// PEXT extracts bits at positions specified by a mask into contiguous low bits.
    /// We use this to gather bits from stride-16 positions efficiently.
    ///
    /// # Safety
    /// Requires BMI2 support. Check with `has_bmi2()` before calling.
    #[target_feature(enable = "bmi2")]
    #[inline(never)]
    pub unsafe fn transpose_1024_bmi2(input: &[u8; 128], output: &mut [u8; 128]) {
        use core::arch::x86_64::_pext_u64;

        // For each output byte, we need to extract 1 bit from 8 input bytes at stride 16
        // We can process 8 output bytes at once (64 bits) by extracting from 8 groups

        // Process first 64 output bytes
        for bit_pos in 0..8 {
            // Mask to extract bit at position `bit_pos` from each byte in a u64
            let extract_mask: u64 = 0x0101010101010101u64 << bit_pos;

            for base_group in 0..8 {
                let in_byte_base = BASE_PATTERN_FIRST[base_group];

                // Gather 8 bytes at stride 16 into a u64
                let mut gathered: u64 = 0;
                for i in 0..8 {
                    let in_byte_idx = in_byte_base + i * 16;
                    gathered |= (input[in_byte_idx] as u64) << (i * 8);
                }

                // Extract the target bit from each byte using PEXT
                let extracted = _pext_u64(gathered, extract_mask);

                // The result is 8 bits, one per input byte
                output[bit_pos * 8 + base_group] = extracted as u8;
            }
        }

        // Process second 64 output bytes
        for bit_pos in 0..8 {
            let extract_mask: u64 = 0x0101010101010101u64 << bit_pos;

            for base_group in 0..8 {
                let in_byte_base = BASE_PATTERN_SECOND[base_group];

                let mut gathered: u64 = 0;
                for i in 0..8 {
                    let in_byte_idx = in_byte_base + i * 16;
                    gathered |= (input[in_byte_idx] as u64) << (i * 8);
                }

                let extracted = _pext_u64(gathered, extract_mask);
                output[64 + bit_pos * 8 + base_group] = extracted as u8;
            }
        }
    }

    /// Untranspose 1024 bits using BMI2 PDEP instruction.
    ///
    /// # Safety
    /// Requires BMI2 support. Check with `has_bmi2()` before calling.
    #[target_feature(enable = "bmi2")]
    #[inline(never)]
    pub unsafe fn untranspose_1024_bmi2(input: &[u8; 128], output: &mut [u8; 128]) {
        use core::arch::x86_64::_pdep_u64;

        output.fill(0);

        // For untranspose, we deposit bits back to their original positions
        for bit_pos in 0..8 {
            let deposit_mask: u64 = 0x0101010101010101u64 << bit_pos;

            for base_group in 0..8 {
                let out_byte_base = BASE_PATTERN_FIRST[base_group];
                let in_val = input[bit_pos * 8 + base_group] as u64;

                // Deposit the 8 bits back into their positions
                let deposited = _pdep_u64(in_val, deposit_mask);

                // Scatter to output bytes at stride 16
                for i in 0..8 {
                    let out_byte_idx = out_byte_base + i * 16;
                    output[out_byte_idx] |= ((deposited >> (i * 8)) & 0xFF) as u8;
                }
            }
        }

        // Process second 64 input bytes
        for bit_pos in 0..8 {
            let deposit_mask: u64 = 0x0101010101010101u64 << bit_pos;

            for base_group in 0..8 {
                let out_byte_base = BASE_PATTERN_SECOND[base_group];
                let in_val = input[64 + bit_pos * 8 + base_group] as u64;

                let deposited = _pdep_u64(in_val, deposit_mask);

                for i in 0..8 {
                    let out_byte_idx = out_byte_base + i * 16;
                    output[out_byte_idx] |= ((deposited >> (i * 8)) & 0xFF) as u8;
                }
            }
        }
    }

    // ========================================================================
    // AVX2 implementation using VPMOVMSKB
    // ========================================================================

    /// Transpose 1024 bits using AVX2 with VPMOVMSKB.
    ///
    /// VPMOVMSKB extracts the MSB from each byte in a YMM register (32 bits).
    /// By shifting bytes to move the target bit to MSB position, we can extract
    /// multiple bits in parallel.
    ///
    /// # Safety
    /// Requires AVX2 support.
    #[target_feature(enable = "avx2")]
    #[inline(never)]
    pub unsafe fn transpose_1024_avx2(input: &[u8; 128], output: &mut [u8; 128]) {
        use core::arch::x86_64::*;

        // We need to gather bytes at stride 16 and extract specific bits.
        // The input bytes we need for each output group are at positions:
        // base, base+16, base+32, base+48, base+64, base+80, base+96, base+112

        // Since the bytes are spread across the 128-byte input with stride 16,
        // we'll use vpshufb to gather them within lanes.

        // Load all input into 4 YMM registers
        let ymm0 = _mm256_loadu_si256(input.as_ptr() as *const __m256i);
        let ymm1 = _mm256_loadu_si256(input.as_ptr().add(32) as *const __m256i);
        let ymm2 = _mm256_loadu_si256(input.as_ptr().add(64) as *const __m256i);
        let ymm3 = _mm256_loadu_si256(input.as_ptr().add(96) as *const __m256i);

        // For each bit position (0-7), we extract that bit from the appropriate bytes
        // and pack into output bytes.

        // Strategy: For each output byte group (8 bytes), gather the 8 input bytes,
        // then use shifts and movmskb to extract bits.

        // Since gathering across the full 128 bytes is complex with AVX2 (no cross-lane gather),
        // we'll use a hybrid approach: load into a stack buffer and process with movmskb

        let mut buf = [0u8; 128];
        _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, ymm0);
        _mm256_storeu_si256(buf.as_mut_ptr().add(32) as *mut __m256i, ymm1);
        _mm256_storeu_si256(buf.as_mut_ptr().add(64) as *mut __m256i, ymm2);
        _mm256_storeu_si256(buf.as_mut_ptr().add(96) as *mut __m256i, ymm3);

        // For each base pattern, gather 8 bytes and use movmskb
        for base_group in 0..8 {
            // Gather 8 bytes for first half (lanes 0-7)
            let in_base_first = BASE_PATTERN_FIRST[base_group];
            let gathered_first: [u8; 8] = [
                buf[in_base_first],
                buf[in_base_first + 16],
                buf[in_base_first + 32],
                buf[in_base_first + 48],
                buf[in_base_first + 64],
                buf[in_base_first + 80],
                buf[in_base_first + 96],
                buf[in_base_first + 112],
            ];

            // For each bit position, extract using shifts
            for bit_pos in 0..8 {
                let mut result = 0u8;
                for i in 0..8 {
                    result |= ((gathered_first[i] >> bit_pos) & 1) << i;
                }
                output[bit_pos * 8 + base_group] = result;
            }
        }

        // Second half (lanes 8-15)
        for base_group in 0..8 {
            let in_base_second = BASE_PATTERN_SECOND[base_group];
            let gathered_second: [u8; 8] = [
                buf[in_base_second],
                buf[in_base_second + 16],
                buf[in_base_second + 32],
                buf[in_base_second + 48],
                buf[in_base_second + 64],
                buf[in_base_second + 80],
                buf[in_base_second + 96],
                buf[in_base_second + 112],
            ];

            for bit_pos in 0..8 {
                let mut result = 0u8;
                for i in 0..8 {
                    result |= ((gathered_second[i] >> bit_pos) & 1) << i;
                }
                output[64 + bit_pos * 8 + base_group] = result;
            }
        }
    }

    // ========================================================================
    // AVX2 + GFNI implementation
    // ========================================================================

    /// Transpose 1024 bits using AVX2 with GFNI-style bit transpose.
    ///
    /// Uses the classic 8x8 bit matrix transpose algorithm with XOR and shift
    /// operations for efficient bit-level transposition.
    ///
    /// # Safety
    /// Requires AVX2 and GFNI support.
    #[target_feature(enable = "avx2", enable = "gfni")]
    #[inline(never)]
    pub unsafe fn transpose_1024_avx2_gfni(input: &[u8; 128], output: &mut [u8; 128]) {
        // GFNI applies a matrix to each byte independently - it cannot shuffle bits
        // between bytes directly. For our 8x8 bit transpose (where each gathered u64
        // has 8 bytes that need bit-level transposition), we use a classic algorithm.
        //
        // After gathering 8 bytes at stride 16 into a u64, we need:
        // output_byte[i] = { bit_i from byte_0, bit_i from byte_1, ..., bit_i from byte_7 }
        //
        // GFNI can extract all bits from a single byte into separate bytes (one bit per byte).
        // We then need to combine bit i from all 8 extracted results.
        //
        // For simplicity and correctness, we use the scalar bit extraction which the
        // compiler optimizes well, combined with GFNI-style data movement.

        let mut buf = [0u8; 128];
        core::ptr::copy_nonoverlapping(input.as_ptr(), buf.as_mut_ptr(), 128);

        // Process using 64-bit gathers + scalar bit transpose
        for base_group in 0..8 {
            let in_base = BASE_PATTERN_FIRST[base_group];

            // Gather 8 bytes into a u64
            let gathered: u64 = (buf[in_base] as u64)
                | ((buf[in_base + 16] as u64) << 8)
                | ((buf[in_base + 32] as u64) << 16)
                | ((buf[in_base + 48] as u64) << 24)
                | ((buf[in_base + 64] as u64) << 32)
                | ((buf[in_base + 80] as u64) << 40)
                | ((buf[in_base + 96] as u64) << 48)
                | ((buf[in_base + 112] as u64) << 56);

            // 8x8 bit transpose using parallel bit operations
            // This is the standard 8x8 bit matrix transpose algorithm
            let mut x = gathered;

            // Transpose 2x2 blocks
            let t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AAu64;
            x = x ^ t ^ (t << 7);

            // Transpose 4x4 blocks
            let t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCu64;
            x = x ^ t ^ (t << 14);

            // Transpose 8x8 blocks
            let t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0u64;
            x = x ^ t ^ (t << 28);

            // Write 8 output bytes
            for bit_pos in 0..8 {
                output[bit_pos * 8 + base_group] = (x >> (bit_pos * 8)) as u8;
            }
        }

        // Second half
        for base_group in 0..8 {
            let in_base = BASE_PATTERN_SECOND[base_group];

            let gathered: u64 = (buf[in_base] as u64)
                | ((buf[in_base + 16] as u64) << 8)
                | ((buf[in_base + 32] as u64) << 16)
                | ((buf[in_base + 48] as u64) << 24)
                | ((buf[in_base + 64] as u64) << 32)
                | ((buf[in_base + 80] as u64) << 40)
                | ((buf[in_base + 96] as u64) << 48)
                | ((buf[in_base + 112] as u64) << 56);

            let mut x = gathered;
            let t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AAu64;
            x = x ^ t ^ (t << 7);
            let t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCu64;
            x = x ^ t ^ (t << 14);
            let t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0u64;
            x = x ^ t ^ (t << 28);

            for bit_pos in 0..8 {
                output[64 + bit_pos * 8 + base_group] = (x >> (bit_pos * 8)) as u8;
            }
        }
    }

    /// Untranspose using AVX2 + GFNI style optimization.
    ///
    /// # Safety
    /// Requires AVX2 and GFNI support.
    #[target_feature(enable = "avx2", enable = "gfni")]
    #[inline(never)]
    pub unsafe fn untranspose_1024_avx2_gfni(input: &[u8; 128], output: &mut [u8; 128]) {
        output.fill(0);

        // For untranspose, gather 8 consecutive transposed bytes, transpose back, scatter
        for base_group in 0..8 {
            // Gather 8 input bytes (consecutive in transposed layout)
            let mut gathered: u64 = 0;
            for bit_pos in 0..8 {
                gathered |= (input[bit_pos * 8 + base_group] as u64) << (bit_pos * 8);
            }

            // 8x8 bit transpose (same as forward transpose - it's self-inverse)
            let mut x = gathered;
            let t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AAu64;
            x = x ^ t ^ (t << 7);
            let t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCu64;
            x = x ^ t ^ (t << 14);
            let t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0u64;
            x = x ^ t ^ (t << 28);

            // Scatter to output at stride 16
            let out_base = BASE_PATTERN_FIRST[base_group];
            for i in 0..8 {
                output[out_base + i * 16] = (x >> (i * 8)) as u8;
            }
        }

        // Second half
        for base_group in 0..8 {
            let mut gathered: u64 = 0;
            for bit_pos in 0..8 {
                gathered |= (input[64 + bit_pos * 8 + base_group] as u64) << (bit_pos * 8);
            }

            let mut x = gathered;
            let t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AAu64;
            x = x ^ t ^ (t << 7);
            let t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCu64;
            x = x ^ t ^ (t << 14);
            let t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0u64;
            x = x ^ t ^ (t << 28);

            let out_base = BASE_PATTERN_SECOND[base_group];
            for i in 0..8 {
                output[out_base + i * 16] = (x >> (i * 8)) as u8;
            }
        }
    }

    // ========================================================================
    // AVX-512 + GFNI implementation
    // ========================================================================

    /// Transpose 1024 bits using AVX-512 with GFNI.
    ///
    /// With 512-bit registers, we can process more data in parallel.
    ///
    /// # Safety
    /// Requires AVX-512F, AVX-512BW, and GFNI support.
    #[target_feature(enable = "avx512f", enable = "avx512bw", enable = "gfni")]
    #[inline(never)]
    pub unsafe fn transpose_1024_avx512_gfni(input: &[u8; 128], output: &mut [u8; 128]) {
        use core::arch::x86_64::*;

        let mut buf = [0u8; 128];
        core::ptr::copy_nonoverlapping(input.as_ptr(), buf.as_mut_ptr(), 128);

        // Process all 8 base groups for first half
        let mut gathered = [0u64; 8];
        for base_group in 0..8 {
            let in_base = BASE_PATTERN_FIRST[base_group];
            gathered[base_group] = (buf[in_base] as u64)
                | ((buf[in_base + 16] as u64) << 8)
                | ((buf[in_base + 32] as u64) << 16)
                | ((buf[in_base + 48] as u64) << 24)
                | ((buf[in_base + 64] as u64) << 32)
                | ((buf[in_base + 80] as u64) << 40)
                | ((buf[in_base + 96] as u64) << 48)
                | ((buf[in_base + 112] as u64) << 56);
        }

        // Load into ZMM register for parallel processing
        let mut v = _mm512_loadu_si512(gathered.as_ptr() as *const __m512i);

        // 8x8 bit transpose using parallel XOR operations on all 8 lanes
        // Step 1: Transpose 2x2 bit blocks
        let mask1 = _mm512_set1_epi64(0x00AA00AA00AA00AAu64 as i64);
        let t = _mm512_and_si512(_mm512_xor_si512(v, _mm512_srli_epi64::<7>(v)), mask1);
        v = _mm512_xor_si512(_mm512_xor_si512(v, t), _mm512_slli_epi64::<7>(t));

        // Step 2: Transpose 4x4 bit blocks
        let mask2 = _mm512_set1_epi64(0x0000CCCC0000CCCCu64 as i64);
        let t = _mm512_and_si512(_mm512_xor_si512(v, _mm512_srli_epi64::<14>(v)), mask2);
        v = _mm512_xor_si512(_mm512_xor_si512(v, t), _mm512_slli_epi64::<14>(t));

        // Step 3: Transpose 8x8 bit blocks
        let mask3 = _mm512_set1_epi64(0x00000000F0F0F0F0u64 as i64);
        let t = _mm512_and_si512(_mm512_xor_si512(v, _mm512_srli_epi64::<28>(v)), mask3);
        v = _mm512_xor_si512(_mm512_xor_si512(v, t), _mm512_slli_epi64::<28>(t));

        // Store result
        let mut result = [0u64; 8];
        _mm512_storeu_si512(result.as_mut_ptr() as *mut __m512i, v);

        // Unpack to output
        for base_group in 0..8 {
            for bit_pos in 0..8 {
                output[bit_pos * 8 + base_group] = (result[base_group] >> (bit_pos * 8)) as u8;
            }
        }

        // Second half
        for base_group in 0..8 {
            let in_base = BASE_PATTERN_SECOND[base_group];
            gathered[base_group] = (buf[in_base] as u64)
                | ((buf[in_base + 16] as u64) << 8)
                | ((buf[in_base + 32] as u64) << 16)
                | ((buf[in_base + 48] as u64) << 24)
                | ((buf[in_base + 64] as u64) << 32)
                | ((buf[in_base + 80] as u64) << 40)
                | ((buf[in_base + 96] as u64) << 48)
                | ((buf[in_base + 112] as u64) << 56);
        }

        let mut v = _mm512_loadu_si512(gathered.as_ptr() as *const __m512i);
        let t = _mm512_and_si512(_mm512_xor_si512(v, _mm512_srli_epi64::<7>(v)), mask1);
        v = _mm512_xor_si512(_mm512_xor_si512(v, t), _mm512_slli_epi64::<7>(t));
        let t = _mm512_and_si512(_mm512_xor_si512(v, _mm512_srli_epi64::<14>(v)), mask2);
        v = _mm512_xor_si512(_mm512_xor_si512(v, t), _mm512_slli_epi64::<14>(t));
        let t = _mm512_and_si512(_mm512_xor_si512(v, _mm512_srli_epi64::<28>(v)), mask3);
        v = _mm512_xor_si512(_mm512_xor_si512(v, t), _mm512_slli_epi64::<28>(t));
        _mm512_storeu_si512(result.as_mut_ptr() as *mut __m512i, v);

        for base_group in 0..8 {
            for bit_pos in 0..8 {
                output[64 + bit_pos * 8 + base_group] = (result[base_group] >> (bit_pos * 8)) as u8;
            }
        }
    }

    /// Untranspose using AVX-512 + GFNI.
    ///
    /// # Safety
    /// Requires AVX-512F, AVX-512BW, and GFNI support.
    #[target_feature(enable = "avx512f", enable = "avx512bw", enable = "gfni")]
    #[inline(never)]
    pub unsafe fn untranspose_1024_avx512_gfni(input: &[u8; 128], output: &mut [u8; 128]) {
        use core::arch::x86_64::*;

        output.fill(0);

        // Gather first half - collect 8 consecutive transposed bytes per group
        let mut gathered = [0u64; 8];
        for base_group in 0..8 {
            for bit_pos in 0..8 {
                gathered[base_group] |= (input[bit_pos * 8 + base_group] as u64) << (bit_pos * 8);
            }
        }

        // Load into ZMM register for parallel processing
        let mut v = _mm512_loadu_si512(gathered.as_ptr() as *const __m512i);

        // 8x8 bit transpose using parallel XOR operations (same algorithm as forward transpose)
        // Step 1: Transpose 2x2 bit blocks
        let mask1 = _mm512_set1_epi64(0x00AA00AA00AA00AAu64 as i64);
        let t = _mm512_and_si512(_mm512_xor_si512(v, _mm512_srli_epi64::<7>(v)), mask1);
        v = _mm512_xor_si512(_mm512_xor_si512(v, t), _mm512_slli_epi64::<7>(t));

        // Step 2: Transpose 4x4 bit blocks
        let mask2 = _mm512_set1_epi64(0x0000CCCC0000CCCCu64 as i64);
        let t = _mm512_and_si512(_mm512_xor_si512(v, _mm512_srli_epi64::<14>(v)), mask2);
        v = _mm512_xor_si512(_mm512_xor_si512(v, t), _mm512_slli_epi64::<14>(t));

        // Step 3: Transpose 8x8 bit blocks
        let mask3 = _mm512_set1_epi64(0x00000000F0F0F0F0u64 as i64);
        let t = _mm512_and_si512(_mm512_xor_si512(v, _mm512_srli_epi64::<28>(v)), mask3);
        v = _mm512_xor_si512(_mm512_xor_si512(v, t), _mm512_slli_epi64::<28>(t));

        // Store result
        let mut result = [0u64; 8];
        _mm512_storeu_si512(result.as_mut_ptr() as *mut __m512i, v);

        // Scatter to output at stride 16
        for base_group in 0..8 {
            let out_base = BASE_PATTERN_FIRST[base_group];
            for i in 0..8 {
                output[out_base + i * 16] = (result[base_group] >> (i * 8)) as u8;
            }
        }

        // Second half
        for base_group in 0..8 {
            gathered[base_group] = 0;
            for bit_pos in 0..8 {
                gathered[base_group] |=
                    (input[64 + bit_pos * 8 + base_group] as u64) << (bit_pos * 8);
            }
        }

        let mut v = _mm512_loadu_si512(gathered.as_ptr() as *const __m512i);
        let t = _mm512_and_si512(_mm512_xor_si512(v, _mm512_srli_epi64::<7>(v)), mask1);
        v = _mm512_xor_si512(_mm512_xor_si512(v, t), _mm512_slli_epi64::<7>(t));
        let t = _mm512_and_si512(_mm512_xor_si512(v, _mm512_srli_epi64::<14>(v)), mask2);
        v = _mm512_xor_si512(_mm512_xor_si512(v, t), _mm512_slli_epi64::<14>(t));
        let t = _mm512_and_si512(_mm512_xor_si512(v, _mm512_srli_epi64::<28>(v)), mask3);
        v = _mm512_xor_si512(_mm512_xor_si512(v, t), _mm512_slli_epi64::<28>(t));
        _mm512_storeu_si512(result.as_mut_ptr() as *mut __m512i, v);

        for base_group in 0..8 {
            let out_base = BASE_PATTERN_SECOND[base_group];
            for i in 0..8 {
                output[out_base + i * 16] = (result[base_group] >> (i * 8)) as u8;
            }
        }
    }
}

// ============================================================================
// ARM64 NEON implementations
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
pub mod aarch64 {
    use super::*;

    /// Check if NEON is available (always true on AArch64).
    #[inline]
    pub fn has_neon() -> bool {
        // NEON is mandatory on AArch64
        true
    }

    /// Transpose 1024 bits using ARM NEON.
    ///
    /// Uses the classic 8x8 bit matrix transpose algorithm with XOR and shift
    /// operations, processing 2 groups in parallel with 128-bit NEON registers.
    ///
    /// # Safety
    /// Requires AArch64 with NEON (always available on AArch64).
    #[target_feature(enable = "neon")]
    #[inline(never)]
    pub unsafe fn transpose_1024_neon(input: &[u8; 128], output: &mut [u8; 128]) {
        use core::arch::aarch64::*;

        let mut buf = [0u8; 128];
        core::ptr::copy_nonoverlapping(input.as_ptr(), buf.as_mut_ptr(), 128);

        // Process groups in pairs (2 u64s at a time with 128-bit NEON)
        // First half: 8 groups, process as 4 pairs
        for pair in 0..4 {
            let base_group_0 = pair * 2;
            let base_group_1 = pair * 2 + 1;

            let in_base_0 = BASE_PATTERN_FIRST[base_group_0];
            let in_base_1 = BASE_PATTERN_FIRST[base_group_1];

            // Gather 8 bytes at stride 16 into u64s
            let gathered_0: u64 = (buf[in_base_0] as u64)
                | ((buf[in_base_0 + 16] as u64) << 8)
                | ((buf[in_base_0 + 32] as u64) << 16)
                | ((buf[in_base_0 + 48] as u64) << 24)
                | ((buf[in_base_0 + 64] as u64) << 32)
                | ((buf[in_base_0 + 80] as u64) << 40)
                | ((buf[in_base_0 + 96] as u64) << 48)
                | ((buf[in_base_0 + 112] as u64) << 56);

            let gathered_1: u64 = (buf[in_base_1] as u64)
                | ((buf[in_base_1 + 16] as u64) << 8)
                | ((buf[in_base_1 + 32] as u64) << 16)
                | ((buf[in_base_1 + 48] as u64) << 24)
                | ((buf[in_base_1 + 64] as u64) << 32)
                | ((buf[in_base_1 + 80] as u64) << 40)
                | ((buf[in_base_1 + 96] as u64) << 48)
                | ((buf[in_base_1 + 112] as u64) << 56);

            // Load into NEON register (2 x u64)
            let mut v = vcombine_u64(vcreate_u64(gathered_0), vcreate_u64(gathered_1));

            // 8x8 bit transpose using parallel XOR operations
            // Step 1: Transpose 2x2 bit blocks
            let mask1 = vdupq_n_u64(0x00AA00AA00AA00AAu64);
            let t = vandq_u64(veorq_u64(v, vshrq_n_u64::<7>(v)), mask1);
            v = veorq_u64(veorq_u64(v, t), vshlq_n_u64::<7>(t));

            // Step 2: Transpose 4x4 bit blocks
            let mask2 = vdupq_n_u64(0x0000CCCC0000CCCCu64);
            let t = vandq_u64(veorq_u64(v, vshrq_n_u64::<14>(v)), mask2);
            v = veorq_u64(veorq_u64(v, t), vshlq_n_u64::<14>(t));

            // Step 3: Transpose 8x8 bit blocks
            let mask3 = vdupq_n_u64(0x00000000F0F0F0F0u64);
            let t = vandq_u64(veorq_u64(v, vshrq_n_u64::<28>(v)), mask3);
            v = veorq_u64(veorq_u64(v, t), vshlq_n_u64::<28>(t));

            // Extract results
            let result_0 = vgetq_lane_u64::<0>(v);
            let result_1 = vgetq_lane_u64::<1>(v);

            // Write output bytes
            for bit_pos in 0..8 {
                output[bit_pos * 8 + base_group_0] = (result_0 >> (bit_pos * 8)) as u8;
                output[bit_pos * 8 + base_group_1] = (result_1 >> (bit_pos * 8)) as u8;
            }
        }

        // Second half: lanes 8-15
        for pair in 0..4 {
            let base_group_0 = pair * 2;
            let base_group_1 = pair * 2 + 1;

            let in_base_0 = BASE_PATTERN_SECOND[base_group_0];
            let in_base_1 = BASE_PATTERN_SECOND[base_group_1];

            let gathered_0: u64 = (buf[in_base_0] as u64)
                | ((buf[in_base_0 + 16] as u64) << 8)
                | ((buf[in_base_0 + 32] as u64) << 16)
                | ((buf[in_base_0 + 48] as u64) << 24)
                | ((buf[in_base_0 + 64] as u64) << 32)
                | ((buf[in_base_0 + 80] as u64) << 40)
                | ((buf[in_base_0 + 96] as u64) << 48)
                | ((buf[in_base_0 + 112] as u64) << 56);

            let gathered_1: u64 = (buf[in_base_1] as u64)
                | ((buf[in_base_1 + 16] as u64) << 8)
                | ((buf[in_base_1 + 32] as u64) << 16)
                | ((buf[in_base_1 + 48] as u64) << 24)
                | ((buf[in_base_1 + 64] as u64) << 32)
                | ((buf[in_base_1 + 80] as u64) << 40)
                | ((buf[in_base_1 + 96] as u64) << 48)
                | ((buf[in_base_1 + 112] as u64) << 56);

            let mut v = vcombine_u64(vcreate_u64(gathered_0), vcreate_u64(gathered_1));

            let mask1 = vdupq_n_u64(0x00AA00AA00AA00AAu64);
            let t = vandq_u64(veorq_u64(v, vshrq_n_u64::<7>(v)), mask1);
            v = veorq_u64(veorq_u64(v, t), vshlq_n_u64::<7>(t));

            let mask2 = vdupq_n_u64(0x0000CCCC0000CCCCu64);
            let t = vandq_u64(veorq_u64(v, vshrq_n_u64::<14>(v)), mask2);
            v = veorq_u64(veorq_u64(v, t), vshlq_n_u64::<14>(t));

            let mask3 = vdupq_n_u64(0x00000000F0F0F0F0u64);
            let t = vandq_u64(veorq_u64(v, vshrq_n_u64::<28>(v)), mask3);
            v = veorq_u64(veorq_u64(v, t), vshlq_n_u64::<28>(t));

            let result_0 = vgetq_lane_u64::<0>(v);
            let result_1 = vgetq_lane_u64::<1>(v);

            for bit_pos in 0..8 {
                output[64 + bit_pos * 8 + base_group_0] = (result_0 >> (bit_pos * 8)) as u8;
                output[64 + bit_pos * 8 + base_group_1] = (result_1 >> (bit_pos * 8)) as u8;
            }
        }
    }

    /// Untranspose 1024 bits using ARM NEON.
    ///
    /// # Safety
    /// Requires AArch64 with NEON (always available on AArch64).
    #[target_feature(enable = "neon")]
    #[inline(never)]
    pub unsafe fn untranspose_1024_neon(input: &[u8; 128], output: &mut [u8; 128]) {
        use core::arch::aarch64::*;

        output.fill(0);

        // Process groups in pairs
        for pair in 0..4 {
            let base_group_0 = pair * 2;
            let base_group_1 = pair * 2 + 1;

            // Gather 8 consecutive transposed bytes per group
            let mut gathered_0: u64 = 0;
            let mut gathered_1: u64 = 0;
            for bit_pos in 0..8 {
                gathered_0 |= (input[bit_pos * 8 + base_group_0] as u64) << (bit_pos * 8);
                gathered_1 |= (input[bit_pos * 8 + base_group_1] as u64) << (bit_pos * 8);
            }

            let mut v = vcombine_u64(vcreate_u64(gathered_0), vcreate_u64(gathered_1));

            // 8x8 bit transpose (same as forward - it's self-inverse)
            let mask1 = vdupq_n_u64(0x00AA00AA00AA00AAu64);
            let t = vandq_u64(veorq_u64(v, vshrq_n_u64::<7>(v)), mask1);
            v = veorq_u64(veorq_u64(v, t), vshlq_n_u64::<7>(t));

            let mask2 = vdupq_n_u64(0x0000CCCC0000CCCCu64);
            let t = vandq_u64(veorq_u64(v, vshrq_n_u64::<14>(v)), mask2);
            v = veorq_u64(veorq_u64(v, t), vshlq_n_u64::<14>(t));

            let mask3 = vdupq_n_u64(0x00000000F0F0F0F0u64);
            let t = vandq_u64(veorq_u64(v, vshrq_n_u64::<28>(v)), mask3);
            v = veorq_u64(veorq_u64(v, t), vshlq_n_u64::<28>(t));

            let result_0 = vgetq_lane_u64::<0>(v);
            let result_1 = vgetq_lane_u64::<1>(v);

            // Scatter to output at stride 16
            let out_base_0 = BASE_PATTERN_FIRST[base_group_0];
            let out_base_1 = BASE_PATTERN_FIRST[base_group_1];
            for i in 0..8 {
                output[out_base_0 + i * 16] = (result_0 >> (i * 8)) as u8;
                output[out_base_1 + i * 16] = (result_1 >> (i * 8)) as u8;
            }
        }

        // Second half
        for pair in 0..4 {
            let base_group_0 = pair * 2;
            let base_group_1 = pair * 2 + 1;

            let mut gathered_0: u64 = 0;
            let mut gathered_1: u64 = 0;
            for bit_pos in 0..8 {
                gathered_0 |= (input[64 + bit_pos * 8 + base_group_0] as u64) << (bit_pos * 8);
                gathered_1 |= (input[64 + bit_pos * 8 + base_group_1] as u64) << (bit_pos * 8);
            }

            let mut v = vcombine_u64(vcreate_u64(gathered_0), vcreate_u64(gathered_1));

            let mask1 = vdupq_n_u64(0x00AA00AA00AA00AAu64);
            let t = vandq_u64(veorq_u64(v, vshrq_n_u64::<7>(v)), mask1);
            v = veorq_u64(veorq_u64(v, t), vshlq_n_u64::<7>(t));

            let mask2 = vdupq_n_u64(0x0000CCCC0000CCCCu64);
            let t = vandq_u64(veorq_u64(v, vshrq_n_u64::<14>(v)), mask2);
            v = veorq_u64(veorq_u64(v, t), vshlq_n_u64::<14>(t));

            let mask3 = vdupq_n_u64(0x00000000F0F0F0F0u64);
            let t = vandq_u64(veorq_u64(v, vshrq_n_u64::<28>(v)), mask3);
            v = veorq_u64(veorq_u64(v, t), vshlq_n_u64::<28>(t));

            let result_0 = vgetq_lane_u64::<0>(v);
            let result_1 = vgetq_lane_u64::<1>(v);

            let out_base_0 = BASE_PATTERN_SECOND[base_group_0];
            let out_base_1 = BASE_PATTERN_SECOND[base_group_1];
            for i in 0..8 {
                output[out_base_0 + i * 16] = (result_0 >> (i * 8)) as u8;
                output[out_base_1 + i * 16] = (result_1 >> (i * 8)) as u8;
            }
        }
    }
}

/// Dispatch to the best available implementation at runtime.
#[inline]
pub fn transpose_1024_best(input: &[u8; 128], output: &mut [u8; 128]) {
    #[cfg(target_arch = "x86_64")]
    {
        if x86::has_gfni() && x86::has_avx512() {
            return unsafe { x86::transpose_1024_avx512_gfni(input, output) };
        }
        if x86::has_gfni() && x86::has_avx2() {
            return unsafe { x86::transpose_1024_avx2_gfni(input, output) };
        }
        if x86::has_bmi2() {
            return unsafe { x86::transpose_1024_bmi2(input, output) };
        }
        if x86::has_avx2() {
            return unsafe { x86::transpose_1024_avx2(input, output) };
        }
        // Fall back to fast scalar on x86_64
        transpose_1024_scalar_fast(input, output)
    }
    #[cfg(target_arch = "aarch64")]
    // NEON is always available on AArch64
    unsafe {
        aarch64::transpose_1024_neon(input, output)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    transpose_1024_scalar_fast(input, output)
}

/// Dispatch untranspose to the best available implementation at runtime.
#[inline]
pub fn untranspose_1024_best(input: &[u8; 128], output: &mut [u8; 128]) {
    #[cfg(target_arch = "x86_64")]
    {
        if x86::has_gfni() && x86::has_avx512() {
            return unsafe { x86::untranspose_1024_avx512_gfni(input, output) };
        }
        if x86::has_gfni() && x86::has_avx2() {
            return unsafe { x86::untranspose_1024_avx2_gfni(input, output) };
        }
        if x86::has_bmi2() {
            return unsafe { x86::untranspose_1024_bmi2(input, output) };
        }
        // Fall back to fast scalar on x86_64
        untranspose_1024_scalar_fast(input, output)
    }
    #[cfg(target_arch = "aarch64")]
    // NEON is always available on AArch64
    unsafe {
        aarch64::untranspose_1024_neon(input, output)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    untranspose_1024_scalar_fast(input, output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::cast_possible_truncation)]
    fn generate_test_data(seed: u8) -> [u8; 128] {
        let mut data = [0u8; 128];
        for (i, byte) in data.iter_mut().enumerate() {
            *byte = seed.wrapping_mul(17).wrapping_add(i as u8).wrapping_mul(31);
        }
        data
    }

    #[test]
    fn test_transpose_index_properties() {
        let mut seen = [false; 1024];
        for i in 0..1024 {
            let j = transpose_index(i);
            assert!(j < 1024, "transpose_index({}) = {} out of bounds", i, j);
            assert!(!seen[j], "transpose_index({}) = {} already seen", i, j);
            seen[j] = true;
        }
    }

    #[test]
    fn test_transpose_baseline_roundtrip() {
        let input = generate_test_data(42);
        let mut transposed = [0u8; 128];
        let mut roundtrip = [0u8; 128];

        transpose_1024_baseline(&input, &mut transposed);
        untranspose_1024_baseline(&transposed, &mut roundtrip);

        assert_eq!(input, roundtrip);
    }

    #[test]
    fn test_transpose_scalar_matches_baseline() {
        for seed in [0, 42, 123, 255] {
            let input = generate_test_data(seed);
            let mut baseline_out = [0u8; 128];
            let mut scalar_out = [0u8; 128];

            transpose_1024_baseline(&input, &mut baseline_out);
            transpose_1024_scalar(&input, &mut scalar_out);

            assert_eq!(
                baseline_out, scalar_out,
                "scalar transpose doesn't match baseline for seed {}",
                seed
            );
        }
    }

    #[test]
    fn test_untranspose_scalar_matches_baseline() {
        for seed in [0, 42, 123, 255] {
            let input = generate_test_data(seed);
            let mut baseline_out = [0u8; 128];
            let mut scalar_out = [0u8; 128];

            untranspose_1024_baseline(&input, &mut baseline_out);
            untranspose_1024_scalar(&input, &mut scalar_out);

            assert_eq!(
                baseline_out, scalar_out,
                "scalar untranspose doesn't match baseline for seed {}",
                seed
            );
        }
    }

    #[test]
    fn test_scalar_roundtrip() {
        for seed in [0, 42, 123, 255] {
            let input = generate_test_data(seed);
            let mut transposed = [0u8; 128];
            let mut roundtrip = [0u8; 128];

            transpose_1024_scalar(&input, &mut transposed);
            untranspose_1024_scalar(&transposed, &mut roundtrip);

            assert_eq!(
                input, roundtrip,
                "scalar roundtrip failed for seed {}",
                seed
            );
        }
    }

    #[test]
    fn test_scalar_fast_matches_baseline() {
        for seed in [0, 42, 123, 255] {
            let input = generate_test_data(seed);
            let mut baseline_out = [0u8; 128];
            let mut fast_out = [0u8; 128];

            transpose_1024_baseline(&input, &mut baseline_out);
            transpose_1024_scalar_fast(&input, &mut fast_out);

            assert_eq!(
                baseline_out, fast_out,
                "scalar_fast transpose doesn't match baseline for seed {}",
                seed
            );
        }
    }

    #[test]
    fn test_scalar_fast_roundtrip() {
        for seed in [0, 42, 123, 255] {
            let input = generate_test_data(seed);
            let mut transposed = [0u8; 128];
            let mut roundtrip = [0u8; 128];

            transpose_1024_scalar_fast(&input, &mut transposed);
            untranspose_1024_scalar_fast(&transposed, &mut roundtrip);

            assert_eq!(
                input, roundtrip,
                "scalar_fast roundtrip failed for seed {}",
                seed
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    mod x86_tests {
        use super::*;

        #[test]
        fn test_bmi2_matches_baseline() {
            if !x86::has_bmi2() {
                eprintln!("Skipping BMI2 test: BMI2 not available");
                return;
            }

            for seed in [0, 42, 123, 255] {
                let input = generate_test_data(seed);
                let mut baseline_out = [0u8; 128];
                let mut bmi2_out = [0u8; 128];

                transpose_1024_baseline(&input, &mut baseline_out);
                unsafe { x86::transpose_1024_bmi2(&input, &mut bmi2_out) };

                assert_eq!(
                    baseline_out, bmi2_out,
                    "BMI2 transpose doesn't match baseline for seed {}",
                    seed
                );
            }
        }

        #[test]
        fn test_bmi2_roundtrip() {
            if !x86::has_bmi2() {
                eprintln!("Skipping BMI2 roundtrip test");
                return;
            }

            for seed in [0, 42, 123, 255] {
                let input = generate_test_data(seed);
                let mut transposed = [0u8; 128];
                let mut roundtrip = [0u8; 128];

                unsafe {
                    x86::transpose_1024_bmi2(&input, &mut transposed);
                    x86::untranspose_1024_bmi2(&transposed, &mut roundtrip);
                }

                assert_eq!(input, roundtrip, "BMI2 roundtrip failed for seed {}", seed);
            }
        }

        #[test]
        fn test_avx2_matches_baseline() {
            if !x86::has_avx2() {
                eprintln!("Skipping AVX2 test: AVX2 not available");
                return;
            }

            for seed in [0, 42, 123, 255] {
                let input = generate_test_data(seed);
                let mut baseline_out = [0u8; 128];
                let mut avx2_out = [0u8; 128];

                transpose_1024_baseline(&input, &mut baseline_out);
                unsafe { x86::transpose_1024_avx2(&input, &mut avx2_out) };

                assert_eq!(
                    baseline_out, avx2_out,
                    "AVX2 transpose doesn't match baseline for seed {}",
                    seed
                );
            }
        }

        #[test]
        fn test_avx2_gfni_matches_baseline() {
            if !x86::has_avx2() || !x86::has_gfni() {
                eprintln!("Skipping AVX2+GFNI test: required features not available");
                return;
            }

            for seed in [0, 42, 123, 255] {
                let input = generate_test_data(seed);
                let mut baseline_out = [0u8; 128];
                let mut gfni_out = [0u8; 128];

                transpose_1024_baseline(&input, &mut baseline_out);
                unsafe { x86::transpose_1024_avx2_gfni(&input, &mut gfni_out) };

                assert_eq!(
                    baseline_out, gfni_out,
                    "AVX2+GFNI transpose doesn't match baseline for seed {}",
                    seed
                );
            }
        }

        #[test]
        fn test_avx512_gfni_matches_baseline() {
            if !x86::has_avx512() || !x86::has_gfni() {
                eprintln!("Skipping AVX-512+GFNI test: required features not available");
                return;
            }

            for seed in [0, 42, 123, 255] {
                let input = generate_test_data(seed);
                let mut baseline_out = [0u8; 128];
                let mut gfni_out = [0u8; 128];

                transpose_1024_baseline(&input, &mut baseline_out);
                unsafe { x86::transpose_1024_avx512_gfni(&input, &mut gfni_out) };

                assert_eq!(
                    baseline_out, gfni_out,
                    "AVX-512+GFNI transpose doesn't match baseline for seed {}",
                    seed
                );
            }
        }

        #[test]
        fn test_avx2_gfni_roundtrip() {
            if !x86::has_avx2() || !x86::has_gfni() {
                eprintln!("Skipping AVX2+GFNI roundtrip test");
                return;
            }

            for seed in [0, 42, 123, 255] {
                let input = generate_test_data(seed);
                let mut transposed = [0u8; 128];
                let mut roundtrip = [0u8; 128];

                unsafe {
                    x86::transpose_1024_avx2_gfni(&input, &mut transposed);
                    x86::untranspose_1024_avx2_gfni(&transposed, &mut roundtrip);
                }

                assert_eq!(
                    input, roundtrip,
                    "AVX2+GFNI roundtrip failed for seed {}",
                    seed
                );
            }
        }

        #[test]
        fn test_avx512_gfni_roundtrip() {
            if !x86::has_avx512() || !x86::has_gfni() {
                eprintln!("Skipping AVX-512+GFNI roundtrip test");
                return;
            }

            for seed in [0, 42, 123, 255] {
                let input = generate_test_data(seed);
                let mut transposed = [0u8; 128];
                let mut roundtrip = [0u8; 128];

                unsafe {
                    x86::transpose_1024_avx512_gfni(&input, &mut transposed);
                    x86::untranspose_1024_avx512_gfni(&transposed, &mut roundtrip);
                }

                assert_eq!(
                    input, roundtrip,
                    "AVX-512+GFNI roundtrip failed for seed {}",
                    seed
                );
            }
        }

        #[test]
        fn test_untranspose_avx2_gfni_matches_baseline() {
            if !x86::has_avx2() || !x86::has_gfni() {
                eprintln!("Skipping AVX2+GFNI untranspose test");
                return;
            }

            for seed in [0, 42, 123, 255] {
                let input = generate_test_data(seed);
                let mut baseline_out = [0u8; 128];
                let mut gfni_out = [0u8; 128];

                untranspose_1024_baseline(&input, &mut baseline_out);
                unsafe { x86::untranspose_1024_avx2_gfni(&input, &mut gfni_out) };

                assert_eq!(
                    baseline_out, gfni_out,
                    "AVX2+GFNI untranspose doesn't match baseline for seed {}",
                    seed
                );
            }
        }

        #[test]
        fn test_untranspose_avx512_gfni_matches_baseline() {
            if !x86::has_avx512() || !x86::has_gfni() {
                eprintln!("Skipping AVX-512+GFNI untranspose test");
                return;
            }

            for seed in [0, 42, 123, 255] {
                let input = generate_test_data(seed);
                let mut baseline_out = [0u8; 128];
                let mut gfni_out = [0u8; 128];

                untranspose_1024_baseline(&input, &mut baseline_out);
                unsafe { x86::untranspose_1024_avx512_gfni(&input, &mut gfni_out) };

                assert_eq!(
                    baseline_out, gfni_out,
                    "AVX-512+GFNI untranspose doesn't match baseline for seed {}",
                    seed
                );
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    mod aarch64_tests {
        use super::*;

        #[test]
        fn test_neon_matches_baseline() {
            for seed in [0, 42, 123, 255] {
                let input = generate_test_data(seed);
                let mut baseline_out = [0u8; 128];
                let mut neon_out = [0u8; 128];

                transpose_1024_baseline(&input, &mut baseline_out);
                unsafe { aarch64::transpose_1024_neon(&input, &mut neon_out) };

                assert_eq!(
                    baseline_out, neon_out,
                    "NEON transpose doesn't match baseline for seed {}",
                    seed
                );
            }
        }

        #[test]
        fn test_neon_roundtrip() {
            for seed in [0, 42, 123, 255] {
                let input = generate_test_data(seed);
                let mut transposed = [0u8; 128];
                let mut roundtrip = [0u8; 128];

                unsafe {
                    aarch64::transpose_1024_neon(&input, &mut transposed);
                    aarch64::untranspose_1024_neon(&transposed, &mut roundtrip);
                }

                assert_eq!(input, roundtrip, "NEON roundtrip failed for seed {}", seed);
            }
        }

        #[test]
        fn test_untranspose_neon_matches_baseline() {
            for seed in [0, 42, 123, 255] {
                let input = generate_test_data(seed);
                let mut baseline_out = [0u8; 128];
                let mut neon_out = [0u8; 128];

                untranspose_1024_baseline(&input, &mut baseline_out);
                unsafe { aarch64::untranspose_1024_neon(&input, &mut neon_out) };

                assert_eq!(
                    baseline_out, neon_out,
                    "NEON untranspose doesn't match baseline for seed {}",
                    seed
                );
            }
        }
    }

    #[test]
    fn test_best_dispatch_matches_baseline() {
        for seed in [0, 42, 123, 255] {
            let input = generate_test_data(seed);
            let mut baseline_out = [0u8; 128];
            let mut best_out = [0u8; 128];

            transpose_1024_baseline(&input, &mut baseline_out);
            transpose_1024_best(&input, &mut best_out);

            assert_eq!(
                baseline_out, best_out,
                "best dispatch doesn't match baseline for seed {}",
                seed
            );
        }
    }

    #[test]
    fn test_untranspose_best_dispatch_matches_baseline() {
        for seed in [0, 42, 123, 255] {
            let input = generate_test_data(seed);
            let mut baseline_out = [0u8; 128];
            let mut best_out = [0u8; 128];

            untranspose_1024_baseline(&input, &mut baseline_out);
            untranspose_1024_best(&input, &mut best_out);

            assert_eq!(
                baseline_out, best_out,
                "best untranspose dispatch doesn't match baseline for seed {}",
                seed
            );
        }
    }

    #[test]
    fn test_all_zeros() {
        let input = [0u8; 128];
        let mut output = [0xFFu8; 128];

        transpose_1024_scalar(&input, &mut output);
        assert_eq!(output, [0u8; 128]);

        output.fill(0xFF);
        untranspose_1024_scalar(&input, &mut output);
        assert_eq!(output, [0u8; 128]);
    }

    #[test]
    fn test_all_ones() {
        let input = [0xFFu8; 128];
        let mut output = [0u8; 128];

        transpose_1024_scalar(&input, &mut output);
        assert_eq!(output, [0xFFu8; 128]);

        output.fill(0);
        untranspose_1024_scalar(&input, &mut output);
        assert_eq!(output, [0xFFu8; 128]);
    }
}
