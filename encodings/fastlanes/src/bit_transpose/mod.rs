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
pub mod scalar;
#[cfg(feature = "_test-harness")]
pub mod x86;

#[cfg(not(feature = "_test-harness"))]
mod aarch64;
#[cfg(not(feature = "_test-harness"))]
mod scalar;
#[cfg(not(feature = "_test-harness"))]
mod x86;

mod validity;

pub use validity::*;

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

/// Transpose 1024-bits into FastLanes layout.
///
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
        scalar::transpose_bits_scalar(input, output);
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { aarch64::transpose_bits_neon(input, output) };
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    scalar::transpose_bits_scalar(input, output);
}

/// Untranspose 1024-bits from FastLanes layout.
///
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
        scalar::untranspose_bits_scalar(input, output);
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { aarch64::untranspose_bits_neon(input, output) };
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    scalar::untranspose_bits_scalar(input, output);
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
}
