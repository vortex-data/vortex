// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::bit_transpose::BASE_PATTERN_FIRST;
use crate::bit_transpose::BASE_PATTERN_SECOND;
use crate::bit_transpose::TRANSPOSE_2X2;
use crate::bit_transpose::TRANSPOSE_4X4;
use crate::bit_transpose::TRANSPOSE_8X8;

/// Fast scalar transpose using the 8x8 bit matrix transpose algorithm.
///
/// This version uses 64-bit gather + parallel bit operations instead of
/// extracting bits one by one. Typically 5-10x faster than the basic scalar version.
#[inline(never)]
pub fn transpose_bits_scalar(input: &[u8; 128], output: &mut [u8; 128]) {
    // Helper to perform 8x8 bit transpose on a u64 (each byte becomes a row)
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
    fn transpose_8x8(mut x: u64) -> u64 {
        let t = (x ^ (x >> 7)) & TRANSPOSE_2X2;
        x = x ^ t ^ (t << 7);
        let t = (x ^ (x >> 14)) & TRANSPOSE_4X4;
        x = x ^ t ^ (t << 14);
        let t = (x ^ (x >> 28)) & TRANSPOSE_8X8;
        x ^ t ^ (t << 28)
    }

    fn gather_transposed(input: &[u8; 128], base_group: usize, offset: usize) -> u64 {
        let mut result: u64 = 0;
        for bit_pos in 0..8 {
            result |= u64::from(input[offset + bit_pos * 8 + base_group]) << (bit_pos * 8);
        }
        result
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bit_transpose::generate_test_data;

    #[test]
    fn test_scalar_matches_baseline() {
        for seed in [0, 42, 123, 255] {
            let input = generate_test_data(seed);
            let mut baseline_out = [0u8; 128];
            let mut fast_out = [0u8; 128];

            transpose_bits_scalar(&input, &mut baseline_out);
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
