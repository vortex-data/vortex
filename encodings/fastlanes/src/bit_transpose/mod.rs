// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fast implementations of the `FastLanes` 1024-bit transpose.
//!
//! The `FastLanes` transpose is a fixed permutation of 1024 bits (128 bytes) that
//! enables SIMD parallelism for encodings like delta and RLE. This module provides
//! optimized implementations for different x86 SIMD instruction sets.
//!
//! The key insight is that each output byte is formed by extracting the SAME bit
//! position from 8 different input bytes at stride 16. The input byte groups follow
//! the `FL_ORDER` permutation pattern.

#[cfg(feature = "_test-harness")]
pub mod aarch64;
#[cfg(feature = "_test-harness")]
pub mod x86;

#[cfg(not(feature = "_test-harness"))]
mod aarch64;
#[cfg(not(feature = "_test-harness"))]
mod x86;

/// Base indices for the first 64 output bytes (lanes 0-7).
/// Each entry indicates the starting input byte index for that output byte group.
/// Pattern: [0*2, 4*2, 2*2, 6*2, 1*2, 5*2, 3*2, 7*2] = [0, 8, 4, 12, 2, 10, 6, 14]
const BASE_PATTERN_FIRST: [usize; 8] = [0, 8, 4, 12, 2, 10, 6, 14];

/// Base indices for the second 64 output bytes (lanes 8-15).
/// Pattern: first pattern + 1 = [1, 9, 5, 13, 3, 11, 7, 15]
const BASE_PATTERN_SECOND: [usize; 8] = [1, 9, 5, 13, 3, 11, 7, 15];

/// Masks for transposing 8x8 bit blocks.
const TRANSPOSE_2X2: u64 = 0x00AA_00AA_00AA_00AA;
const TRANSPOSE_4X4: u64 = 0x0000_CCCC_0000_CCCC;
const TRANSPOSE_8X8: u64 = 0x0000_0000_F0F0_F0F0;

/// Fast scalar transpose using the 8x8 bit matrix transpose algorithm.
///
/// This version uses 64-bit gather + parallel bit operations instead of
/// extracting bits one by one. Typically 5-10x faster than the basic scalar version.
#[inline(never)]
pub fn transpose_bits_scalar(input: &[u8; 128], output: &mut [u8; 128]) {
    // Helper to perform 8x8 bit transpose on a u64 (each byte becomes a row)
    #[inline]
    fn transpose_8x8(mut x: u64) -> u64 {
        // Step 1: Transpose 2x2 bit blocks
        let t = (x ^ (x >> 7)) & TRANSPOSE_2X2;
        x = x ^ t ^ (t << 7);
        // Step 2: Transpose 4x4 bit blocks
        let t = (x ^ (x >> 14)) & TRANSPOSE_4X4;
        x = x ^ t ^ (t << 14);
        // Step 3: Transpose 8x8 bit blocks
        let t = (x ^ (x >> 28)) & TRANSPOSE_8X8;
        x ^ t ^ (t << 28)
    }

    // Helper to gather 8 bytes at stride 16 into a u64
    #[inline]
    fn gather(input: &[u8; 128], base: usize) -> u64 {
        u64::from(input[base])
            | (u64::from(input[base + 16]) << 8)
            | (u64::from(input[base + 32]) << 16)
            | (u64::from(input[base + 48]) << 24)
            | (u64::from(input[base + 64]) << 32)
            | (u64::from(input[base + 80]) << 40)
            | (u64::from(input[base + 96]) << 48)
            | (u64::from(input[base + 112]) << 56)
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
pub fn untranspose_bits_scalar(input: &[u8; 128], output: &mut [u8; 128]) {
    #[inline]
    fn transpose_8x8(mut x: u64) -> u64 {
        let t = (x ^ (x >> 7)) & TRANSPOSE_2X2;
        x = x ^ t ^ (t << 7);
        let t = (x ^ (x >> 14)) & TRANSPOSE_4X4;
        x = x ^ t ^ (t << 14);
        let t = (x ^ (x >> 28)) & TRANSPOSE_8X8;
        x ^ t ^ (t << 28)
    }

    #[inline]
    fn gather_transposed(input: &[u8; 128], base_group: usize, offset: usize) -> u64 {
        let mut result: u64 = 0;
        for bit_pos in 0..8 {
            result |= u64::from(input[offset + bit_pos * 8 + base_group]) << (bit_pos * 8);
        }
        result
    }

    #[inline]
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

/// Dispatch to the best available implementation at runtime.
#[inline]
pub fn transpose_bits(input: &[u8; 128], output: &mut [u8; 128]) {
    #[cfg(target_arch = "x86_64")]
    {
        // VBMI is fastest
        if x86::has_vbmi() {
            unsafe { x86::transpose_bits_vbmi(input, output) };
            return;
        }
        if x86::has_bmi2() {
            unsafe { x86::transpose_bits_bmi2(input, output) };
            return;
        }
        // Fall back to scalar
        transpose_bits_scalar(input, output);
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { aarch64::transpose_bits_neon(input, output) };
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    transpose_bits_scalar(input, output);
}

/// Dispatch untranspose to the best available implementation at runtime.
#[inline]
pub fn untranspose_bits(input: &[u8; 128], output: &mut [u8; 128]) {
    #[cfg(target_arch = "x86_64")]
    {
        // VBMI is fastest
        if x86::has_vbmi() {
            unsafe { x86::untranspose_bits_vbmi(input, output) };
            return;
        }
        if x86::has_bmi2() {
            unsafe { x86::untranspose_bits_bmi2(input, output) };
            return;
        }
        // Fall back to scalar
        untranspose_bits_scalar(input, output);
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { aarch64::untranspose_bits_neon(input, output) };
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    untranspose_bits_scalar(input, output);
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
fn generate_test_data(seed: u8) -> [u8; 128] {
    let mut data = [0u8; 128];
    for (i, byte) in data.iter_mut().enumerate() {
        *byte = seed.wrapping_mul(17).wrapping_add(i as u8).wrapping_mul(31);
    }
    data
}

#[cfg(test)]
pub fn transpose_bits_baseline(input: &[u8; 128], output: &mut [u8; 128]) {
    for in_bit in 0..1024 {
        let out_bit = fastlanes::transpose(in_bit);
        let in_byte = in_bit / 8;
        let in_bit_pos = in_bit % 8;
        let out_byte = out_bit / 8;
        let out_bit_pos = out_bit % 8;
        let bit_val = (input[in_byte] >> in_bit_pos) & 1;
        output[out_byte] |= bit_val << out_bit_pos;
    }
}

#[cfg(test)]
pub fn untranspose_bits_baseline(input: &[u8; 128], output: &mut [u8; 128]) {
    for out_bit in 0..1024 {
        let in_bit = fastlanes::transpose(out_bit);
        let in_byte = in_bit / 8;
        let in_bit_pos = in_bit % 8;
        let out_byte = out_bit / 8;
        let out_bit_pos = out_bit % 8;
        let bit_val = (input[in_byte] >> in_bit_pos) & 1;
        output[out_byte] |= bit_val << out_bit_pos;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_baseline_roundtrip() {
        let input = generate_test_data(42);
        let mut transposed = [0u8; 128];
        let mut roundtrip = [0u8; 128];

        transpose_bits_baseline(&input, &mut transposed);
        untranspose_bits_baseline(&transposed, &mut roundtrip);

        assert_eq!(input, roundtrip);
    }

    #[test]
    fn test_scalar_matches_baseline() {
        for seed in [0, 42, 123, 255] {
            let input = generate_test_data(seed);
            let mut baseline_out = [0u8; 128];
            let mut fast_out = [0u8; 128];

            transpose_bits_baseline(&input, &mut baseline_out);
            transpose_bits_scalar(&input, &mut fast_out);

            assert_eq!(
                baseline_out, fast_out,
                "scalar_fast transpose doesn't match baseline for seed {seed}"
            );
        }
    }

    #[test]
    fn test_scalar_roundtrip() {
        for seed in [0, 42, 123, 255] {
            let input = generate_test_data(seed);
            let mut transposed = [0u8; 128];
            let mut roundtrip = [0u8; 128];

            transpose_bits_scalar(&input, &mut transposed);
            untranspose_bits_scalar(&transposed, &mut roundtrip);

            assert_eq!(
                input, roundtrip,
                "scalar_fast roundtrip failed for seed {seed}"
            );
        }
    }

    #[test]
    fn test_dispatch_matches_baseline() {
        for seed in [0, 42, 123, 255] {
            let input = generate_test_data(seed);
            let mut baseline_out = [0u8; 128];
            let mut out = [0u8; 128];

            transpose_bits_baseline(&input, &mut baseline_out);
            transpose_bits(&input, &mut out);

            assert_eq!(
                baseline_out, out,
                "best dispatch doesn't match baseline for seed {seed}"
            );
        }
    }

    #[test]
    fn test_untranspose_dispatch_matches_baseline() {
        for seed in [0, 42, 123, 255] {
            let input = generate_test_data(seed);
            let mut baseline_out = [0u8; 128];
            let mut out = [0u8; 128];

            untranspose_bits_baseline(&input, &mut baseline_out);
            untranspose_bits(&input, &mut out);

            assert_eq!(
                baseline_out, out,
                "best untranspose dispatch doesn't match baseline for seed {seed}"
            );
        }
    }

    #[test]
    fn test_all_zeros() {
        let input = [0u8; 128];
        let mut output = [0xFFu8; 128];

        transpose_bits_scalar(&input, &mut output);
        assert_eq!(output, [0u8; 128]);

        untranspose_bits_scalar(&input, &mut output);
        assert_eq!(output, [0u8; 128]);
    }

    #[test]
    fn test_all_ones() {
        let input = [0xFFu8; 128];
        let mut output = [0u8; 128];

        transpose_bits_scalar(&input, &mut output);
        assert_eq!(output, [0xFFu8; 128]);

        untranspose_bits_scalar(&input, &mut output);
        assert_eq!(output, [0xFFu8; 128]);
    }
}
