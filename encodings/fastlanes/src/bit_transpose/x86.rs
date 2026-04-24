// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![cfg(target_arch = "x86_64")]

use core::arch::x86_64::__m512i;
use core::arch::x86_64::_mm512_and_si512;
use core::arch::x86_64::_mm512_loadu_si512;
use core::arch::x86_64::_mm512_permutex2var_epi8;
use core::arch::x86_64::_mm512_permutexvar_epi8;
use core::arch::x86_64::_mm512_set1_epi64;
use core::arch::x86_64::_mm512_slli_epi64;
use core::arch::x86_64::_mm512_srli_epi64;
use core::arch::x86_64::_mm512_storeu_si512;
use core::arch::x86_64::_mm512_xor_si512;
use core::arch::x86_64::_pdep_u64;
use core::arch::x86_64::_pext_u64;
use std::is_x86_feature_detected;

use crate::bit_transpose::BASE_PATTERN_FIRST;
use crate::bit_transpose::BASE_PATTERN_SECOND;
use crate::bit_transpose::TRANSPOSE_2X2;
use crate::bit_transpose::TRANSPOSE_4X4;
use crate::bit_transpose::TRANSPOSE_8X8;

/// Check if BMI2 is available.
#[inline]
#[must_use]
pub fn has_bmi2() -> bool {
    is_x86_feature_detected!("bmi2")
}

/// Check if AVX-512 VBMI is available (for byte permutation).
#[inline]
#[must_use]
pub fn has_vbmi() -> bool {
    is_x86_feature_detected!("avx512vbmi")
}

/// Transpose 1024 bits using BMI2 PEXT instruction.
///
/// PEXT extracts bits at positions specified by a mask into contiguous low bits.
/// Fully unrolled for ~12% better performance vs looped version.
///
/// # Safety
/// Requires BMI2 support. Check with `has_bmi2()` before calling.
#[target_feature(enable = "bmi2")]
#[inline(never)]
#[expect(clippy::too_many_lines)]
pub unsafe fn transpose_bits_bmi2(input: &[u8; 128], output: &mut [u8; 128]) {
    // Helper to gather 8 bytes at stride 16 into a u64
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

    // Gather all 16 groups (fully unrolled)
    // First half: BASE_PATTERN_FIRST = [0, 8, 4, 12, 2, 10, 6, 14]
    let g0 = gather(input, 0);
    let g1 = gather(input, 8);
    let g2 = gather(input, 4);
    let g3 = gather(input, 12);
    let g4 = gather(input, 2);
    let g5 = gather(input, 10);
    let g6 = gather(input, 6);
    let g7 = gather(input, 14);
    // Second half: BASE_PATTERN_SECOND = [1, 9, 5, 13, 3, 11, 7, 15]
    let g8 = gather(input, 1);
    let g9 = gather(input, 9);
    let g10 = gather(input, 5);
    let g11 = gather(input, 13);
    let g12 = gather(input, 3);
    let g13 = gather(input, 11);
    let g14 = gather(input, 7);
    let g15 = gather(input, 15);

    // Masks for each bit position
    let m0: u64 = 0x0101_0101_0101_0101;
    let m1: u64 = 0x0202_0202_0202_0202;
    let m2: u64 = 0x0404_0404_0404_0404;
    let m3: u64 = 0x0808_0808_0808_0808;
    let m4: u64 = 0x1010_1010_1010_1010;
    let m5: u64 = 0x2020_2020_2020_2020;
    let m6: u64 = 0x4040_4040_4040_4040;
    let m7: u64 = 0x8080_8080_8080_8080;

    // First half - 64 PEXT operations (fully unrolled)
    output[0] = _pext_u64(g0, m0) as u8;
    output[1] = _pext_u64(g1, m0) as u8;
    output[2] = _pext_u64(g2, m0) as u8;
    output[3] = _pext_u64(g3, m0) as u8;
    output[4] = _pext_u64(g4, m0) as u8;
    output[5] = _pext_u64(g5, m0) as u8;
    output[6] = _pext_u64(g6, m0) as u8;
    output[7] = _pext_u64(g7, m0) as u8;
    output[8] = _pext_u64(g0, m1) as u8;
    output[9] = _pext_u64(g1, m1) as u8;
    output[10] = _pext_u64(g2, m1) as u8;
    output[11] = _pext_u64(g3, m1) as u8;
    output[12] = _pext_u64(g4, m1) as u8;
    output[13] = _pext_u64(g5, m1) as u8;
    output[14] = _pext_u64(g6, m1) as u8;
    output[15] = _pext_u64(g7, m1) as u8;
    output[16] = _pext_u64(g0, m2) as u8;
    output[17] = _pext_u64(g1, m2) as u8;
    output[18] = _pext_u64(g2, m2) as u8;
    output[19] = _pext_u64(g3, m2) as u8;
    output[20] = _pext_u64(g4, m2) as u8;
    output[21] = _pext_u64(g5, m2) as u8;
    output[22] = _pext_u64(g6, m2) as u8;
    output[23] = _pext_u64(g7, m2) as u8;
    output[24] = _pext_u64(g0, m3) as u8;
    output[25] = _pext_u64(g1, m3) as u8;
    output[26] = _pext_u64(g2, m3) as u8;
    output[27] = _pext_u64(g3, m3) as u8;
    output[28] = _pext_u64(g4, m3) as u8;
    output[29] = _pext_u64(g5, m3) as u8;
    output[30] = _pext_u64(g6, m3) as u8;
    output[31] = _pext_u64(g7, m3) as u8;
    output[32] = _pext_u64(g0, m4) as u8;
    output[33] = _pext_u64(g1, m4) as u8;
    output[34] = _pext_u64(g2, m4) as u8;
    output[35] = _pext_u64(g3, m4) as u8;
    output[36] = _pext_u64(g4, m4) as u8;
    output[37] = _pext_u64(g5, m4) as u8;
    output[38] = _pext_u64(g6, m4) as u8;
    output[39] = _pext_u64(g7, m4) as u8;
    output[40] = _pext_u64(g0, m5) as u8;
    output[41] = _pext_u64(g1, m5) as u8;
    output[42] = _pext_u64(g2, m5) as u8;
    output[43] = _pext_u64(g3, m5) as u8;
    output[44] = _pext_u64(g4, m5) as u8;
    output[45] = _pext_u64(g5, m5) as u8;
    output[46] = _pext_u64(g6, m5) as u8;
    output[47] = _pext_u64(g7, m5) as u8;
    output[48] = _pext_u64(g0, m6) as u8;
    output[49] = _pext_u64(g1, m6) as u8;
    output[50] = _pext_u64(g2, m6) as u8;
    output[51] = _pext_u64(g3, m6) as u8;
    output[52] = _pext_u64(g4, m6) as u8;
    output[53] = _pext_u64(g5, m6) as u8;
    output[54] = _pext_u64(g6, m6) as u8;
    output[55] = _pext_u64(g7, m6) as u8;
    output[56] = _pext_u64(g0, m7) as u8;
    output[57] = _pext_u64(g1, m7) as u8;
    output[58] = _pext_u64(g2, m7) as u8;
    output[59] = _pext_u64(g3, m7) as u8;
    output[60] = _pext_u64(g4, m7) as u8;
    output[61] = _pext_u64(g5, m7) as u8;
    output[62] = _pext_u64(g6, m7) as u8;
    output[63] = _pext_u64(g7, m7) as u8;

    // Second half - 64 PEXT operations (fully unrolled)
    output[64] = _pext_u64(g8, m0) as u8;
    output[65] = _pext_u64(g9, m0) as u8;
    output[66] = _pext_u64(g10, m0) as u8;
    output[67] = _pext_u64(g11, m0) as u8;
    output[68] = _pext_u64(g12, m0) as u8;
    output[69] = _pext_u64(g13, m0) as u8;
    output[70] = _pext_u64(g14, m0) as u8;
    output[71] = _pext_u64(g15, m0) as u8;
    output[72] = _pext_u64(g8, m1) as u8;
    output[73] = _pext_u64(g9, m1) as u8;
    output[74] = _pext_u64(g10, m1) as u8;
    output[75] = _pext_u64(g11, m1) as u8;
    output[76] = _pext_u64(g12, m1) as u8;
    output[77] = _pext_u64(g13, m1) as u8;
    output[78] = _pext_u64(g14, m1) as u8;
    output[79] = _pext_u64(g15, m1) as u8;
    output[80] = _pext_u64(g8, m2) as u8;
    output[81] = _pext_u64(g9, m2) as u8;
    output[82] = _pext_u64(g10, m2) as u8;
    output[83] = _pext_u64(g11, m2) as u8;
    output[84] = _pext_u64(g12, m2) as u8;
    output[85] = _pext_u64(g13, m2) as u8;
    output[86] = _pext_u64(g14, m2) as u8;
    output[87] = _pext_u64(g15, m2) as u8;
    output[88] = _pext_u64(g8, m3) as u8;
    output[89] = _pext_u64(g9, m3) as u8;
    output[90] = _pext_u64(g10, m3) as u8;
    output[91] = _pext_u64(g11, m3) as u8;
    output[92] = _pext_u64(g12, m3) as u8;
    output[93] = _pext_u64(g13, m3) as u8;
    output[94] = _pext_u64(g14, m3) as u8;
    output[95] = _pext_u64(g15, m3) as u8;
    output[96] = _pext_u64(g8, m4) as u8;
    output[97] = _pext_u64(g9, m4) as u8;
    output[98] = _pext_u64(g10, m4) as u8;
    output[99] = _pext_u64(g11, m4) as u8;
    output[100] = _pext_u64(g12, m4) as u8;
    output[101] = _pext_u64(g13, m4) as u8;
    output[102] = _pext_u64(g14, m4) as u8;
    output[103] = _pext_u64(g15, m4) as u8;
    output[104] = _pext_u64(g8, m5) as u8;
    output[105] = _pext_u64(g9, m5) as u8;
    output[106] = _pext_u64(g10, m5) as u8;
    output[107] = _pext_u64(g11, m5) as u8;
    output[108] = _pext_u64(g12, m5) as u8;
    output[109] = _pext_u64(g13, m5) as u8;
    output[110] = _pext_u64(g14, m5) as u8;
    output[111] = _pext_u64(g15, m5) as u8;
    output[112] = _pext_u64(g8, m6) as u8;
    output[113] = _pext_u64(g9, m6) as u8;
    output[114] = _pext_u64(g10, m6) as u8;
    output[115] = _pext_u64(g11, m6) as u8;
    output[116] = _pext_u64(g12, m6) as u8;
    output[117] = _pext_u64(g13, m6) as u8;
    output[118] = _pext_u64(g14, m6) as u8;
    output[119] = _pext_u64(g15, m6) as u8;
    output[120] = _pext_u64(g8, m7) as u8;
    output[121] = _pext_u64(g9, m7) as u8;
    output[122] = _pext_u64(g10, m7) as u8;
    output[123] = _pext_u64(g11, m7) as u8;
    output[124] = _pext_u64(g12, m7) as u8;
    output[125] = _pext_u64(g13, m7) as u8;
    output[126] = _pext_u64(g14, m7) as u8;
    output[127] = _pext_u64(g15, m7) as u8;
}

/// Untranspose 1024 bits using BMI2 PDEP instruction.
///
/// Structured per-output-group: for each group of 8 output bytes at stride 16,
/// PDEP 8 input bytes into different bit positions, OR in registers, then
/// scatter-store once. Each output byte is written exactly once (no read-modify-write).
///
/// # Safety
/// Requires BMI2 support. Check with `has_bmi2()` before calling.
#[target_feature(enable = "bmi2")]
#[inline(never)]
#[expect(clippy::too_many_lines)]
pub unsafe fn untranspose_bits_bmi2(input: &[u8; 128], output: &mut [u8; 128]) {
    // Helper: scatter a u64 to 8 output bytes at stride 16
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

    // Masks for each bit position
    let m0: u64 = 0x0101_0101_0101_0101;
    let m1: u64 = 0x0202_0202_0202_0202;
    let m2: u64 = 0x0404_0404_0404_0404;
    let m3: u64 = 0x0808_0808_0808_0808;
    let m4: u64 = 0x1010_1010_1010_1010;
    let m5: u64 = 0x2020_2020_2020_2020;
    let m6: u64 = 0x4040_4040_4040_4040;
    let m7: u64 = 0x8080_8080_8080_8080;

    // For each output group, the input bytes that contribute are at
    // input[bit_pos * 8 + group_idx] for bit_pos 0..8.
    // PDEP deposits the 8 bits of the input byte into the bit_pos position
    // of each byte in the u64.

    // First half: 8 groups using BASE_PATTERN_FIRST
    // Group 0 (base=0): input bytes [0, 8, 16, 24, 32, 40, 48, 56]
    let v = _pdep_u64(input[0] as u64, m0)
        | _pdep_u64(input[8] as u64, m1)
        | _pdep_u64(input[16] as u64, m2)
        | _pdep_u64(input[24] as u64, m3)
        | _pdep_u64(input[32] as u64, m4)
        | _pdep_u64(input[40] as u64, m5)
        | _pdep_u64(input[48] as u64, m6)
        | _pdep_u64(input[56] as u64, m7);
    scatter(output, 0, v);

    // Group 1 (base=8)
    let v = _pdep_u64(input[1] as u64, m0)
        | _pdep_u64(input[9] as u64, m1)
        | _pdep_u64(input[17] as u64, m2)
        | _pdep_u64(input[25] as u64, m3)
        | _pdep_u64(input[33] as u64, m4)
        | _pdep_u64(input[41] as u64, m5)
        | _pdep_u64(input[49] as u64, m6)
        | _pdep_u64(input[57] as u64, m7);
    scatter(output, 8, v);

    // Group 2 (base=4)
    let v = _pdep_u64(input[2] as u64, m0)
        | _pdep_u64(input[10] as u64, m1)
        | _pdep_u64(input[18] as u64, m2)
        | _pdep_u64(input[26] as u64, m3)
        | _pdep_u64(input[34] as u64, m4)
        | _pdep_u64(input[42] as u64, m5)
        | _pdep_u64(input[50] as u64, m6)
        | _pdep_u64(input[58] as u64, m7);
    scatter(output, 4, v);

    // Group 3 (base=12)
    let v = _pdep_u64(input[3] as u64, m0)
        | _pdep_u64(input[11] as u64, m1)
        | _pdep_u64(input[19] as u64, m2)
        | _pdep_u64(input[27] as u64, m3)
        | _pdep_u64(input[35] as u64, m4)
        | _pdep_u64(input[43] as u64, m5)
        | _pdep_u64(input[51] as u64, m6)
        | _pdep_u64(input[59] as u64, m7);
    scatter(output, 12, v);

    // Group 4 (base=2)
    let v = _pdep_u64(input[4] as u64, m0)
        | _pdep_u64(input[12] as u64, m1)
        | _pdep_u64(input[20] as u64, m2)
        | _pdep_u64(input[28] as u64, m3)
        | _pdep_u64(input[36] as u64, m4)
        | _pdep_u64(input[44] as u64, m5)
        | _pdep_u64(input[52] as u64, m6)
        | _pdep_u64(input[60] as u64, m7);
    scatter(output, 2, v);

    // Group 5 (base=10)
    let v = _pdep_u64(input[5] as u64, m0)
        | _pdep_u64(input[13] as u64, m1)
        | _pdep_u64(input[21] as u64, m2)
        | _pdep_u64(input[29] as u64, m3)
        | _pdep_u64(input[37] as u64, m4)
        | _pdep_u64(input[45] as u64, m5)
        | _pdep_u64(input[53] as u64, m6)
        | _pdep_u64(input[61] as u64, m7);
    scatter(output, 10, v);

    // Group 6 (base=6)
    let v = _pdep_u64(input[6] as u64, m0)
        | _pdep_u64(input[14] as u64, m1)
        | _pdep_u64(input[22] as u64, m2)
        | _pdep_u64(input[30] as u64, m3)
        | _pdep_u64(input[38] as u64, m4)
        | _pdep_u64(input[46] as u64, m5)
        | _pdep_u64(input[54] as u64, m6)
        | _pdep_u64(input[62] as u64, m7);
    scatter(output, 6, v);

    // Group 7 (base=14)
    let v = _pdep_u64(input[7] as u64, m0)
        | _pdep_u64(input[15] as u64, m1)
        | _pdep_u64(input[23] as u64, m2)
        | _pdep_u64(input[31] as u64, m3)
        | _pdep_u64(input[39] as u64, m4)
        | _pdep_u64(input[47] as u64, m5)
        | _pdep_u64(input[55] as u64, m6)
        | _pdep_u64(input[63] as u64, m7);
    scatter(output, 14, v);

    // Second half: 8 groups using BASE_PATTERN_SECOND
    // Group 0 (base=1)
    let v = _pdep_u64(input[64] as u64, m0)
        | _pdep_u64(input[72] as u64, m1)
        | _pdep_u64(input[80] as u64, m2)
        | _pdep_u64(input[88] as u64, m3)
        | _pdep_u64(input[96] as u64, m4)
        | _pdep_u64(input[104] as u64, m5)
        | _pdep_u64(input[112] as u64, m6)
        | _pdep_u64(input[120] as u64, m7);
    scatter(output, 1, v);

    // Group 1 (base=9)
    let v = _pdep_u64(input[65] as u64, m0)
        | _pdep_u64(input[73] as u64, m1)
        | _pdep_u64(input[81] as u64, m2)
        | _pdep_u64(input[89] as u64, m3)
        | _pdep_u64(input[97] as u64, m4)
        | _pdep_u64(input[105] as u64, m5)
        | _pdep_u64(input[113] as u64, m6)
        | _pdep_u64(input[121] as u64, m7);
    scatter(output, 9, v);

    // Group 2 (base=5)
    let v = _pdep_u64(input[66] as u64, m0)
        | _pdep_u64(input[74] as u64, m1)
        | _pdep_u64(input[82] as u64, m2)
        | _pdep_u64(input[90] as u64, m3)
        | _pdep_u64(input[98] as u64, m4)
        | _pdep_u64(input[106] as u64, m5)
        | _pdep_u64(input[114] as u64, m6)
        | _pdep_u64(input[122] as u64, m7);
    scatter(output, 5, v);

    // Group 3 (base=13)
    let v = _pdep_u64(input[67] as u64, m0)
        | _pdep_u64(input[75] as u64, m1)
        | _pdep_u64(input[83] as u64, m2)
        | _pdep_u64(input[91] as u64, m3)
        | _pdep_u64(input[99] as u64, m4)
        | _pdep_u64(input[107] as u64, m5)
        | _pdep_u64(input[115] as u64, m6)
        | _pdep_u64(input[123] as u64, m7);
    scatter(output, 13, v);

    // Group 4 (base=3)
    let v = _pdep_u64(input[68] as u64, m0)
        | _pdep_u64(input[76] as u64, m1)
        | _pdep_u64(input[84] as u64, m2)
        | _pdep_u64(input[92] as u64, m3)
        | _pdep_u64(input[100] as u64, m4)
        | _pdep_u64(input[108] as u64, m5)
        | _pdep_u64(input[116] as u64, m6)
        | _pdep_u64(input[124] as u64, m7);
    scatter(output, 3, v);

    // Group 5 (base=11)
    let v = _pdep_u64(input[69] as u64, m0)
        | _pdep_u64(input[77] as u64, m1)
        | _pdep_u64(input[85] as u64, m2)
        | _pdep_u64(input[93] as u64, m3)
        | _pdep_u64(input[101] as u64, m4)
        | _pdep_u64(input[109] as u64, m5)
        | _pdep_u64(input[117] as u64, m6)
        | _pdep_u64(input[125] as u64, m7);
    scatter(output, 11, v);

    // Group 6 (base=7)
    let v = _pdep_u64(input[70] as u64, m0)
        | _pdep_u64(input[78] as u64, m1)
        | _pdep_u64(input[86] as u64, m2)
        | _pdep_u64(input[94] as u64, m3)
        | _pdep_u64(input[102] as u64, m4)
        | _pdep_u64(input[110] as u64, m5)
        | _pdep_u64(input[118] as u64, m6)
        | _pdep_u64(input[126] as u64, m7);
    scatter(output, 7, v);

    // Group 7 (base=15)
    let v = _pdep_u64(input[71] as u64, m0)
        | _pdep_u64(input[79] as u64, m1)
        | _pdep_u64(input[87] as u64, m2)
        | _pdep_u64(input[95] as u64, m3)
        | _pdep_u64(input[103] as u64, m4)
        | _pdep_u64(input[111] as u64, m5)
        | _pdep_u64(input[119] as u64, m6)
        | _pdep_u64(input[127] as u64, m7);
    scatter(output, 15, v);
}

// Static permutation tables for VBMI gather operations
static GATHER_FIRST: [u8; 64] = [
    // Gather bytes at stride 16 for first 8 groups (bases from BASE_PATTERN_FIRST)
    // Group 0: base=0
    0, 16, 32, 48, 64, 80, 96, 112, // Group 1: base=8
    8, 24, 40, 56, 72, 88, 104, 120, // Group 2: base=4
    4, 20, 36, 52, 68, 84, 100, 116, // Group 3: base=12
    12, 28, 44, 60, 76, 92, 108, 124, // Group 4: base=2
    2, 18, 34, 50, 66, 82, 98, 114, // Group 5: base=10
    10, 26, 42, 58, 74, 90, 106, 122, // Group 6: base=6
    6, 22, 38, 54, 70, 86, 102, 118, // Group 7: base=14
    14, 30, 46, 62, 78, 94, 110, 126,
];

static GATHER_SECOND: [u8; 64] = [
    // Gather bytes at stride 16 for second 8 groups (bases from BASE_PATTERN_SECOND)
    // Group 0: base=1
    1, 17, 33, 49, 65, 81, 97, 113, // Group 1: base=9
    9, 25, 41, 57, 73, 89, 105, 121, // Group 2: base=5
    5, 21, 37, 53, 69, 85, 101, 117, // Group 3: base=13
    13, 29, 45, 61, 77, 93, 109, 125, // Group 4: base=3
    3, 19, 35, 51, 67, 83, 99, 115, // Group 5: base=11
    11, 27, 43, 59, 75, 91, 107, 123, // Group 6: base=7
    7, 23, 39, 55, 71, 87, 103, 119, // Group 7: base=15
    15, 31, 47, 63, 79, 95, 111, 127,
];

// 8x8 byte transpose permutation for scatter phase
// Input:  [g0b0..g0b7, g1b0..g1b7, ..., g7b0..g7b7] (8 groups of 8 bytes)
// Output: [g0b0,g1b0,..,g7b0, g0b1,g1b1,..,g7b1, ...] (8 rows of 8 bytes)
static SCATTER_8X8: [u8; 64] = [
    0, 8, 16, 24, 32, 40, 48, 56, // byte 0 from each group
    1, 9, 17, 25, 33, 41, 49, 57, // byte 1 from each group
    2, 10, 18, 26, 34, 42, 50, 58, // byte 2 from each group
    3, 11, 19, 27, 35, 43, 51, 59, // byte 3 from each group
    4, 12, 20, 28, 36, 44, 52, 60, // byte 4 from each group
    5, 13, 21, 29, 37, 45, 53, 61, // byte 5 from each group
    6, 14, 22, 30, 38, 46, 54, 62, // byte 6 from each group
    7, 15, 23, 31, 39, 47, 55, 63, // byte 7 from each group
];

/// Transpose 1024 bits using AVX-512 VBMI for vectorized gather and scatter.
///
/// Uses vpermi2b to gather bytes from stride-16 positions in parallel,
/// and vpermb for the final 8x8 byte transpose to output format.
///
/// # Safety
/// Requires AVX-512F, AVX-512BW, and AVX-512VBMI support.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vbmi")]
#[inline(never)]
#[expect(clippy::cast_possible_wrap)]
#[expect(clippy::cast_ptr_alignment)]
#[expect(unsafe_op_in_unsafe_fn)]
pub unsafe fn transpose_bits_vbmi(input: &[u8; 128], output: &mut [u8; 128]) {
    // Load all 128 input bytes into two ZMM registers
    let in_lo = _mm512_loadu_si512(input.as_ptr().cast::<__m512i>());
    let in_hi = _mm512_loadu_si512(input.as_ptr().add(64).cast::<__m512i>());

    // Load permutation indices (static tables)
    let idx_first = _mm512_loadu_si512(GATHER_FIRST.as_ptr().cast::<__m512i>());
    let idx_second = _mm512_loadu_si512(GATHER_SECOND.as_ptr().cast::<__m512i>());
    let idx_scatter = _mm512_loadu_si512(SCATTER_8X8.as_ptr().cast::<__m512i>());

    // Masks for 8x8 bit transpose
    let mask1 = _mm512_set1_epi64(TRANSPOSE_2X2 as i64);
    let mask2 = _mm512_set1_epi64(TRANSPOSE_4X4 as i64);
    let mask3 = _mm512_set1_epi64(TRANSPOSE_8X8 as i64);

    // Process first half
    let gathered = _mm512_permutex2var_epi8(in_lo, idx_first, in_hi);

    // 8x8 bit transpose on all 8 groups in parallel
    let mut v = gathered;
    let t = _mm512_and_si512(_mm512_xor_si512(v, _mm512_srli_epi64::<7>(v)), mask1);
    v = _mm512_xor_si512(_mm512_xor_si512(v, t), _mm512_slli_epi64::<7>(t));
    let t = _mm512_and_si512(_mm512_xor_si512(v, _mm512_srli_epi64::<14>(v)), mask2);
    v = _mm512_xor_si512(_mm512_xor_si512(v, t), _mm512_slli_epi64::<14>(t));
    let t = _mm512_and_si512(_mm512_xor_si512(v, _mm512_srli_epi64::<28>(v)), mask3);
    v = _mm512_xor_si512(_mm512_xor_si512(v, t), _mm512_slli_epi64::<28>(t));

    // 8x8 byte transpose for scatter using vpermb
    let scattered = _mm512_permutexvar_epi8(idx_scatter, v);
    _mm512_storeu_si512(output.as_mut_ptr().cast::<__m512i>(), scattered);

    // Process second half
    let gathered = _mm512_permutex2var_epi8(in_lo, idx_second, in_hi);

    let mut v = gathered;
    let t = _mm512_and_si512(_mm512_xor_si512(v, _mm512_srli_epi64::<7>(v)), mask1);
    v = _mm512_xor_si512(_mm512_xor_si512(v, t), _mm512_slli_epi64::<7>(t));
    let t = _mm512_and_si512(_mm512_xor_si512(v, _mm512_srli_epi64::<14>(v)), mask2);
    v = _mm512_xor_si512(_mm512_xor_si512(v, t), _mm512_slli_epi64::<14>(t));
    let t = _mm512_and_si512(_mm512_xor_si512(v, _mm512_srli_epi64::<28>(v)), mask3);
    v = _mm512_xor_si512(_mm512_xor_si512(v, t), _mm512_slli_epi64::<28>(t));

    let scattered = _mm512_permutexvar_epi8(idx_scatter, v);
    _mm512_storeu_si512(output.as_mut_ptr().add(64).cast::<__m512i>(), scattered);
}

/// Untranspose 1024 bits using AVX-512 VBMI for vectorized scatter.
///
/// # Safety
/// Requires AVX-512F, AVX-512BW, and AVX-512VBMI support.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vbmi")]
#[inline(never)]
#[expect(clippy::cast_possible_wrap)]
#[expect(clippy::cast_ptr_alignment)]
#[expect(unsafe_op_in_unsafe_fn)]
pub unsafe fn untranspose_bits_vbmi(input: &[u8; 128], output: &mut [u8; 128]) {
    // For untranspose, we gather consecutive bytes from transposed layout,
    // then scatter back to stride-16 positions

    // Gather indices for first half - collect 8 bytes per group from transposed layout
    // In transposed layout, bytes for group 0 are at: [0, 8, 16, 24, 32, 40, 48, 56]
    let gather_indices: [u8; 64] = [
        0, 8, 16, 24, 32, 40, 48, 56, // Group 0
        1, 9, 17, 25, 33, 41, 49, 57, // Group 1
        2, 10, 18, 26, 34, 42, 50, 58, // Group 2
        3, 11, 19, 27, 35, 43, 51, 59, // Group 3
        4, 12, 20, 28, 36, 44, 52, 60, // Group 4
        5, 13, 21, 29, 37, 45, 53, 61, // Group 5
        6, 14, 22, 30, 38, 46, 54, 62, // Group 6
        7, 15, 23, 31, 39, 47, 55, 63, // Group 7
    ];

    let in_first = _mm512_loadu_si512(input.as_ptr().cast::<__m512i>());
    let idx = _mm512_loadu_si512(gather_indices.as_ptr().cast::<__m512i>());
    let gathered = _mm512_permutexvar_epi8(idx, in_first);

    // 8x8 bit transpose
    let mask1 = _mm512_set1_epi64(TRANSPOSE_2X2 as i64);
    let mask2 = _mm512_set1_epi64(TRANSPOSE_4X4 as i64);
    let mask3 = _mm512_set1_epi64(TRANSPOSE_8X8 as i64);

    let mut v = gathered;
    let t = _mm512_and_si512(_mm512_xor_si512(v, _mm512_srli_epi64::<7>(v)), mask1);
    v = _mm512_xor_si512(_mm512_xor_si512(v, t), _mm512_slli_epi64::<7>(t));

    let t = _mm512_and_si512(_mm512_xor_si512(v, _mm512_srli_epi64::<14>(v)), mask2);
    v = _mm512_xor_si512(_mm512_xor_si512(v, t), _mm512_slli_epi64::<14>(t));

    let t = _mm512_and_si512(_mm512_xor_si512(v, _mm512_srli_epi64::<28>(v)), mask3);
    v = _mm512_xor_si512(_mm512_xor_si512(v, t), _mm512_slli_epi64::<28>(t));

    // Scatter to output at stride 16 - need to use scalar stores for now
    // (AVX-512 scatter is available but complex for this pattern)
    let mut result = [0u64; 8];
    _mm512_storeu_si512(result.as_mut_ptr().cast::<__m512i>(), v);

    for base_group in 0..8 {
        let out_base = BASE_PATTERN_FIRST[base_group];
        for i in 0..8 {
            output[out_base + i * 16] = (result[base_group] >> (i * 8)) as u8;
        }
    }

    // Second half
    let in_second = _mm512_loadu_si512(input.as_ptr().add(64).cast::<__m512i>());
    let gathered = _mm512_permutexvar_epi8(idx, in_second);

    let mut v = gathered;
    let t = _mm512_and_si512(_mm512_xor_si512(v, _mm512_srli_epi64::<7>(v)), mask1);
    v = _mm512_xor_si512(_mm512_xor_si512(v, t), _mm512_slli_epi64::<7>(t));

    let t = _mm512_and_si512(_mm512_xor_si512(v, _mm512_srli_epi64::<14>(v)), mask2);
    v = _mm512_xor_si512(_mm512_xor_si512(v, t), _mm512_slli_epi64::<14>(t));

    let t = _mm512_and_si512(_mm512_xor_si512(v, _mm512_srli_epi64::<28>(v)), mask3);
    v = _mm512_xor_si512(_mm512_xor_si512(v, t), _mm512_slli_epi64::<28>(t));

    _mm512_storeu_si512(result.as_mut_ptr().cast::<__m512i>(), v);

    for base_group in 0..8 {
        let out_base = BASE_PATTERN_SECOND[base_group];
        for i in 0..8 {
            output[out_base + i * 16] = (result[base_group] >> (i * 8)) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::bit_transpose::generate_test_data;
    use crate::bit_transpose::transpose_bits_baseline;
    use crate::bit_transpose::x86::has_bmi2;
    use crate::bit_transpose::x86::has_vbmi;
    use crate::bit_transpose::x86::transpose_bits_bmi2;
    use crate::bit_transpose::x86::transpose_bits_vbmi;
    use crate::bit_transpose::x86::untranspose_bits_bmi2;
    use crate::bit_transpose::x86::untranspose_bits_vbmi;

    #[test]
    fn test_bmi2_matches_baseline() {
        if !has_bmi2() {
            return;
        }

        for seed in [0, 42, 123, 255] {
            let input = generate_test_data(seed);
            let mut baseline_out = [0u8; 128];
            let mut bmi2_out = [0u8; 128];

            transpose_bits_baseline(&input, &mut baseline_out);
            unsafe { transpose_bits_bmi2(&input, &mut bmi2_out) };

            assert_eq!(
                baseline_out, bmi2_out,
                "BMI2 transpose doesn't match baseline for seed {seed}"
            );
        }
    }

    #[test]
    fn test_bmi2_roundtrip() {
        if !has_bmi2() {
            return;
        }

        for seed in [0, 42, 123, 255] {
            let input = generate_test_data(seed);
            let mut transposed = [0u8; 128];
            let mut roundtrip = [0u8; 128];

            unsafe {
                transpose_bits_bmi2(&input, &mut transposed);
                untranspose_bits_bmi2(&transposed, &mut roundtrip);
            }

            assert_eq!(input, roundtrip, "BMI2 roundtrip failed for seed {seed}");
        }
    }

    #[test]
    fn test_vbmi_matches_baseline() {
        if !has_vbmi() {
            return;
        }

        for seed in [0, 42, 123, 255] {
            let input = generate_test_data(seed);
            let mut baseline_out = [0u8; 128];
            let mut vbmi_out = [0u8; 128];

            transpose_bits_baseline(&input, &mut baseline_out);
            unsafe { transpose_bits_vbmi(&input, &mut vbmi_out) };

            assert_eq!(
                baseline_out, vbmi_out,
                "VBMI transpose doesn't match baseline for seed {seed}"
            );
        }
    }

    #[test]
    fn test_vbmi_roundtrip() {
        if !has_vbmi() {
            return;
        }

        for seed in [0, 42, 123, 255] {
            let input = generate_test_data(seed);
            let mut transposed = [0u8; 128];
            let mut roundtrip = [0u8; 128];

            unsafe {
                transpose_bits_vbmi(&input, &mut transposed);
                untranspose_bits_vbmi(&transposed, &mut roundtrip);
            }

            assert_eq!(input, roundtrip, "VBMI roundtrip failed for seed {seed}");
        }
    }
}
