// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![cfg(target_arch = "aarch64")]

use core::arch::aarch64::uint64x2_t;
use core::arch::aarch64::vandq_u64;
use core::arch::aarch64::vdupq_n_u64;
use core::arch::aarch64::veorq_u64;
use core::arch::aarch64::vgetq_lane_u64;
use core::arch::aarch64::vld1q_u8;
use core::arch::aarch64::vld1q_u8_x4;
use core::arch::aarch64::vorrq_u8;
use core::arch::aarch64::vqtbl4q_u8;
use core::arch::aarch64::vreinterpretq_u8_u64;
use core::arch::aarch64::vreinterpretq_u64_u8;
use core::arch::aarch64::vshlq_n_u64;
use core::arch::aarch64::vshrq_n_u64;
use core::arch::aarch64::vst1q_u8;

use crate::bit_transpose::BASE_PATTERN_FIRST;
use crate::bit_transpose::BASE_PATTERN_SECOND;
use crate::bit_transpose::TRANSPOSE_2X2;
use crate::bit_transpose::TRANSPOSE_4X4;
use crate::bit_transpose::TRANSPOSE_8X8;

/// Gather indices for the first half from input[0..64].
/// Each group needs 4 bytes at stride 16 (the low half of the stride pattern).
/// Layout: [`g0_from_lo(4` bytes), pad(4 bytes), `g1_from_lo(4` bytes), pad(4 bytes), ...]
/// Two groups per 16-byte NEON register.
static GATHER_FIRST_LO: [[u8; 16]; 4] = [
    // Groups 0,1 from BASE_PATTERN_FIRST: bases 0, 8
    [
        0, 16, 32, 48, 0xFF, 0xFF, 0xFF, 0xFF, 8, 24, 40, 56, 0xFF, 0xFF, 0xFF, 0xFF,
    ],
    // Groups 2,3: bases 4, 12
    [
        4, 20, 36, 52, 0xFF, 0xFF, 0xFF, 0xFF, 12, 28, 44, 60, 0xFF, 0xFF, 0xFF, 0xFF,
    ],
    // Groups 4,5: bases 2, 10
    [
        2, 18, 34, 50, 0xFF, 0xFF, 0xFF, 0xFF, 10, 26, 42, 58, 0xFF, 0xFF, 0xFF, 0xFF,
    ],
    // Groups 6,7: bases 6, 14
    [
        6, 22, 38, 54, 0xFF, 0xFF, 0xFF, 0xFF, 14, 30, 46, 62, 0xFF, 0xFF, 0xFF, 0xFF,
    ],
];

/// Gather indices for the first half from input[64..128].
/// These fill in bytes 4-7 of each u64 (the high half of the stride pattern).
static GATHER_FIRST_HI: [[u8; 16]; 4] = [
    // Groups 0,1: bases 0, 8 (offset by -64 since table starts at input[64])
    [
        0xFF, 0xFF, 0xFF, 0xFF, 0, 16, 32, 48, 0xFF, 0xFF, 0xFF, 0xFF, 8, 24, 40, 56,
    ],
    // Groups 2,3: bases 4, 12
    [
        0xFF, 0xFF, 0xFF, 0xFF, 4, 20, 36, 52, 0xFF, 0xFF, 0xFF, 0xFF, 12, 28, 44, 60,
    ],
    // Groups 4,5: bases 2, 10
    [
        0xFF, 0xFF, 0xFF, 0xFF, 2, 18, 34, 50, 0xFF, 0xFF, 0xFF, 0xFF, 10, 26, 42, 58,
    ],
    // Groups 6,7: bases 6, 14
    [
        0xFF, 0xFF, 0xFF, 0xFF, 6, 22, 38, 54, 0xFF, 0xFF, 0xFF, 0xFF, 14, 30, 46, 62,
    ],
];

/// Gather indices for the second half from input[0..64].
/// Uses `BASE_PATTERN_SECOND`: bases [1, 9, 5, 13, 3, 11, 7, 15]
static GATHER_SECOND_LO: [[u8; 16]; 4] = [
    [
        1, 17, 33, 49, 0xFF, 0xFF, 0xFF, 0xFF, 9, 25, 41, 57, 0xFF, 0xFF, 0xFF, 0xFF,
    ],
    [
        5, 21, 37, 53, 0xFF, 0xFF, 0xFF, 0xFF, 13, 29, 45, 61, 0xFF, 0xFF, 0xFF, 0xFF,
    ],
    [
        3, 19, 35, 51, 0xFF, 0xFF, 0xFF, 0xFF, 11, 27, 43, 59, 0xFF, 0xFF, 0xFF, 0xFF,
    ],
    [
        7, 23, 39, 55, 0xFF, 0xFF, 0xFF, 0xFF, 15, 31, 47, 63, 0xFF, 0xFF, 0xFF, 0xFF,
    ],
];

/// Gather indices for the second half from input[64..128].
static GATHER_SECOND_HI: [[u8; 16]; 4] = [
    [
        0xFF, 0xFF, 0xFF, 0xFF, 1, 17, 33, 49, 0xFF, 0xFF, 0xFF, 0xFF, 9, 25, 41, 57,
    ],
    [
        0xFF, 0xFF, 0xFF, 0xFF, 5, 21, 37, 53, 0xFF, 0xFF, 0xFF, 0xFF, 13, 29, 45, 61,
    ],
    [
        0xFF, 0xFF, 0xFF, 0xFF, 3, 19, 35, 51, 0xFF, 0xFF, 0xFF, 0xFF, 11, 27, 43, 59,
    ],
    [
        0xFF, 0xFF, 0xFF, 0xFF, 7, 23, 39, 55, 0xFF, 0xFF, 0xFF, 0xFF, 15, 31, 47, 63,
    ],
];

/// 8x8 byte transpose (scatter) permutation split into 4 × 16-byte chunks for NEON TBL.
/// Input layout:  [g0b0..g0b7, g1b0..g1b7, ..., g7b0..g7b7] (64 bytes, group-major)
/// Output layout: [g0b0,g1b0,..,g7b0, g0b1,g1b1,..,g7b1, ...] (64 bytes, row-major)
/// Same permutation as x86 `SCATTER_8X8`, split for 16-byte NEON registers.
static SCATTER_8X8_NEON: [[u8; 16]; 4] = [
    [0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57],
    [2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59],
    [4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61],
    [6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63],
];

/// Perform 8x8 bit transpose on two u64s packed in a `uint64x2_t`.
#[expect(unsafe_op_in_unsafe_fn)]
unsafe fn bit_transpose_8x8_neon(mut v: uint64x2_t) -> uint64x2_t {
    let mask1 = vdupq_n_u64(TRANSPOSE_2X2);
    let t = vandq_u64(veorq_u64(v, vshrq_n_u64::<7>(v)), mask1);
    v = veorq_u64(veorq_u64(v, t), vshlq_n_u64::<7>(t));

    let mask2 = vdupq_n_u64(TRANSPOSE_4X4);
    let t = vandq_u64(veorq_u64(v, vshrq_n_u64::<14>(v)), mask2);
    v = veorq_u64(veorq_u64(v, t), vshlq_n_u64::<14>(t));

    let mask3 = vdupq_n_u64(TRANSPOSE_8X8);
    let t = vandq_u64(veorq_u64(v, vshrq_n_u64::<28>(v)), mask3);
    veorq_u64(veorq_u64(v, t), vshlq_n_u64::<28>(t))
}

/// Transpose 1024 bits using ARM NEON with TBL-based vectorized gather and scatter.
///
/// Uses `vqtbl4q_u8` to gather bytes from the 128-byte input in parallel,
/// avoiding scalar byte-by-byte loads. Then uses `vqtbl4q_u8` again to perform
/// the 8x8 byte transpose for scatter. This is the NEON analog of x86 VBMI's
/// `vpermb`/`vpermi2b` byte permutation instructions.
///
/// # Safety
/// Requires `AArch64` with NEON (always available on `AArch64`).
#[expect(unsafe_op_in_unsafe_fn)]
#[inline(never)]
pub unsafe fn transpose_bits_neon(input: &[u8; 128], output: &mut [u8; 128]) {
    // Load all 128 input bytes into two uint8x16x4_t tables (64 bytes each)
    let tbl_lo = vld1q_u8_x4(input.as_ptr());
    let tbl_hi = vld1q_u8_x4(input.as_ptr().add(64));

    // Load scatter permutation indices (4 × 16 bytes)
    let scatter0 = vld1q_u8(SCATTER_8X8_NEON[0].as_ptr());
    let scatter1 = vld1q_u8(SCATTER_8X8_NEON[1].as_ptr());
    let scatter2 = vld1q_u8(SCATTER_8X8_NEON[2].as_ptr());
    let scatter3 = vld1q_u8(SCATTER_8X8_NEON[3].as_ptr());

    // Process first 64 output bytes (8 groups from BASE_PATTERN_FIRST)
    // Gather and bit-transpose all 4 pairs, then scatter the full 64 bytes
    let mut buf = [0u8; 64];
    for (i, (gather_lo, gather_high)) in [
        (GATHER_FIRST_LO, GATHER_FIRST_HI),
        (GATHER_SECOND_LO, GATHER_SECOND_HI),
    ]
    .iter()
    .enumerate()
    {
        for pair in 0..4 {
            let idx_lo = vld1q_u8(gather_lo[pair].as_ptr());
            let idx_hi = vld1q_u8(gather_high[pair].as_ptr());

            let from_lo = vqtbl4q_u8(tbl_lo, idx_lo);
            let from_hi = vqtbl4q_u8(tbl_hi, idx_hi);
            let gathered = vorrq_u8(from_lo, from_hi);

            let v = bit_transpose_8x8_neon(vreinterpretq_u64_u8(gathered));
            vst1q_u8(buf.as_mut_ptr().add(pair * 16), vreinterpretq_u8_u64(v));
        }

        // Load the 64-byte result as a TBL table and apply 8x8 byte transpose
        let result_tbl = vld1q_u8_x4(buf.as_ptr());
        vst1q_u8(
            output.as_mut_ptr().add(i * 64),
            vqtbl4q_u8(result_tbl, scatter0),
        );
        vst1q_u8(
            output.as_mut_ptr().add(i * 64 + 16),
            vqtbl4q_u8(result_tbl, scatter1),
        );
        vst1q_u8(
            output.as_mut_ptr().add(i * 64 + 32),
            vqtbl4q_u8(result_tbl, scatter2),
        );
        vst1q_u8(
            output.as_mut_ptr().add(i * 64 + 48),
            vqtbl4q_u8(result_tbl, scatter3),
        );
    }
}

/// Untranspose 1024 bits using ARM NEON with TBL-based vectorized operations.
///
/// # Safety
/// Requires `AArch64` with NEON (always available on `AArch64`).
#[expect(unsafe_op_in_unsafe_fn)]
#[inline(never)]
pub unsafe fn untranspose_bits_neon(input: &[u8; 128], output: &mut [u8; 128]) {
    // Load scatter indices (SCATTER_8X8 is self-inverse, so same table un-scatters)
    let scatter0 = vld1q_u8(SCATTER_8X8_NEON[0].as_ptr());
    let scatter1 = vld1q_u8(SCATTER_8X8_NEON[1].as_ptr());
    let scatter2 = vld1q_u8(SCATTER_8X8_NEON[2].as_ptr());
    let scatter3 = vld1q_u8(SCATTER_8X8_NEON[3].as_ptr());

    // Each iteration un-scatters the 64-byte input block to group-major order
    let mut buf = [0u8; 64];
    for (i, base_pattern) in [BASE_PATTERN_FIRST, BASE_PATTERN_SECOND].iter().enumerate() {
        let in_tbl = vld1q_u8_x4(input.as_ptr().add(i * 64));
        vst1q_u8(buf.as_mut_ptr(), vqtbl4q_u8(in_tbl, scatter0));
        vst1q_u8(buf.as_mut_ptr().add(16), vqtbl4q_u8(in_tbl, scatter1));
        vst1q_u8(buf.as_mut_ptr().add(32), vqtbl4q_u8(in_tbl, scatter2));
        vst1q_u8(buf.as_mut_ptr().add(48), vqtbl4q_u8(in_tbl, scatter3));

        // Bit-transpose each pair and scatter to stride-16 output
        for pair in 0..4 {
            let base_group_0 = pair * 2;
            let base_group_1 = pair * 2 + 1;

            let gathered = vld1q_u8(buf.as_ptr().add(pair * 16));
            let v = bit_transpose_8x8_neon(vreinterpretq_u64_u8(gathered));

            let result_0 = vgetq_lane_u64::<0>(v);
            let result_1 = vgetq_lane_u64::<1>(v);

            let out_base_0 = base_pattern[base_group_0];
            let out_base_1 = base_pattern[base_group_1];
            for i in 0..8 {
                output[out_base_0 + i * 16] = (result_0 >> (i * 8)) as u8;
                output[out_base_1 + i * 16] = (result_1 >> (i * 8)) as u8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::bit_transpose::aarch64::transpose_bits_neon;
    use crate::bit_transpose::aarch64::untranspose_bits_neon;
    use crate::bit_transpose::generate_test_data;
    use crate::bit_transpose::transpose_bits_baseline;
    use crate::bit_transpose::untranspose_bits_baseline;

    #[test]
    fn test_neon_matches_baseline() {
        for seed in [0, 42, 123, 255] {
            let input = generate_test_data(seed);
            let mut baseline_out = [0u8; 128];
            let mut tbl_out = [0u8; 128];

            transpose_bits_baseline(&input, &mut baseline_out);
            unsafe { transpose_bits_neon(&input, &mut tbl_out) };

            assert_eq!(
                baseline_out, tbl_out,
                "NEON TBL transpose doesn't match baseline for seed {seed}"
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
                transpose_bits_neon(&input, &mut transposed);
                untranspose_bits_neon(&transposed, &mut roundtrip);
            }

            assert_eq!(
                input, roundtrip,
                "NEON TBL roundtrip failed for seed {seed}"
            );
        }
    }

    #[test]
    fn test_untranspose_neon_matches_baseline() {
        for seed in [0, 42, 123, 255] {
            let input = generate_test_data(seed);
            let mut baseline_out = [0u8; 128];
            let mut tbl_out = [0u8; 128];

            untranspose_bits_baseline(&input, &mut baseline_out);
            unsafe { untranspose_bits_neon(&input, &mut tbl_out) };

            assert_eq!(
                baseline_out, tbl_out,
                "NEON TBL untranspose doesn't match baseline for seed {seed}"
            );
        }
    }
}
