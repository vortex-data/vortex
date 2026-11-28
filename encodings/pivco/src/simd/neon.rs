// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// SPDX-FileCopyrightText: 2026 Marcin Żukowski
// SPDX-FileComment: Derived from pivco-huffman at commit 16b48fcf at https://github.com/MarcinZukowski/pivco-huffman

//! `aarch64` NEON SIMD kernels for `vortex-pivco`.

#![expect(clippy::cast_possible_truncation)]

use core::arch::aarch64::uint8x8_t;
use core::arch::aarch64::uint8x16_t;
use core::arch::aarch64::uint8x16x2_t;
use core::arch::aarch64::uint8x16x4_t;
use core::arch::aarch64::vabdq_s8;
use core::arch::aarch64::vaddlvq_u16;
use core::arch::aarch64::vaddq_u8;
use core::arch::aarch64::vaddq_u16;
use core::arch::aarch64::vand_u8;
use core::arch::aarch64::vandq_u8;
use core::arch::aarch64::vandq_u16;
use core::arch::aarch64::vcgeq_u8;
use core::arch::aarch64::vcnt_u8;
use core::arch::aarch64::vcntq_u8;
use core::arch::aarch64::vcombine_u8;
use core::arch::aarch64::vdup_n_u8;
use core::arch::aarch64::vdupq_n_u8;
use core::arch::aarch64::vdupq_n_u16;
use core::arch::aarch64::vdupq_n_u32;
use core::arch::aarch64::vdupq_n_u64;
use core::arch::aarch64::vget_high_u8;
use core::arch::aarch64::vget_lane_u64;
use core::arch::aarch64::vget_low_s8;
use core::arch::aarch64::vget_low_u8;
use core::arch::aarch64::vgetq_lane_u32;
use core::arch::aarch64::vld1_u8;
use core::arch::aarch64::vld1q_s8;
use core::arch::aarch64::vld1q_s16;
use core::arch::aarch64::vld1q_s32;
use core::arch::aarch64::vld1q_u8;
use core::arch::aarch64::vmovn_u16;
use core::arch::aarch64::vmvn_u8;
use core::arch::aarch64::vpaddlq_u8;
use core::arch::aarch64::vpaddq_u8;
use core::arch::aarch64::vqtbl1q_u8;
use core::arch::aarch64::vqtbl2_u8;
use core::arch::aarch64::vqtbl2q_u8;
use core::arch::aarch64::vqtbl4_u8;
use core::arch::aarch64::vqtbl4q_u8;
use core::arch::aarch64::vreinterpret_u64_u8;
use core::arch::aarch64::vreinterpretq_u8_s8;
use core::arch::aarch64::vreinterpretq_u8_u16;
use core::arch::aarch64::vreinterpretq_u8_u32;
use core::arch::aarch64::vreinterpretq_u8_u64;
use core::arch::aarch64::vreinterpretq_u16_u8;
use core::arch::aarch64::vreinterpretq_u32_u8;
use core::arch::aarch64::vreinterpretq_u32_u16;
use core::arch::aarch64::vreinterpretq_u64_u32;
use core::arch::aarch64::vsetq_lane_u8;
use core::arch::aarch64::vsetq_lane_u16;
use core::arch::aarch64::vsetq_lane_u32;
use core::arch::aarch64::vsetq_lane_u64;
use core::arch::aarch64::vshl_u8;
use core::arch::aarch64::vshlq_u8;
use core::arch::aarch64::vshlq_u16;
use core::arch::aarch64::vshlq_u32;
use core::arch::aarch64::vshrq_n_u8;
use core::arch::aarch64::vshrq_n_u64;
use core::arch::aarch64::vst1_u8;
use core::arch::aarch64::vst1q_u8;
use core::arch::aarch64::vst2q_u8;
use core::arch::aarch64::vst4q_u8;
use core::arch::aarch64::vsubq_u8;
use core::arch::aarch64::vtbl1_u8;
use core::arch::aarch64::vtstq_u8;
use core::arch::aarch64::vzip1q_u8;
use core::arch::aarch64::vzip2q_u8;

use super::COMPRESS_LEFT_TAB;
use super::COMPRESS_RIGHT_TAB;
use super::G_MERGE_SHUF0;
use super::G_MERGE_SHUF1;
use super::POPCNT8;
use super::merge_cst_side_tail_raw;
use super::merge_vec_vec_tail_raw;
use super::unpack_flat_tail_scalar;

static BIT_WEIGHTS_16: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];

/// 64B/iter NEON popcount (`pivco-huffman/src/prims/neon.h:53-89`).
///
/// Accumulates into 8 `u16` lanes (safe up to `8 * 65,535 = 524,280` bits, guarded by the caller)
/// and horizontally widens into `u32` via `vaddlvq_u16`.
#[target_feature(enable = "neon")]
pub(super) unsafe fn popcount_bits_neon(bm: &[u8], k: usize) -> usize {
    let full_bytes = k >> 3;
    let partial_bits = k & 7;
    let bm_ptr = bm.as_ptr();
    let mut acc_vec = vdupq_n_u16(0);
    let mut pos = 0usize;

    while pos + 64 <= full_bytes {
        let v0 = unsafe { vld1q_u8(bm_ptr.add(pos)) };
        let v1 = unsafe { vld1q_u8(bm_ptr.add(pos + 16)) };
        let v2 = unsafe { vld1q_u8(bm_ptr.add(pos + 32)) };
        let v3 = unsafe { vld1q_u8(bm_ptr.add(pos + 48)) };
        let s01 = vaddq_u8(vcntq_u8(v0), vcntq_u8(v1));
        let s23 = vaddq_u8(vcntq_u8(v2), vcntq_u8(v3));
        acc_vec = vaddq_u16(acc_vec, vpaddlq_u8(vaddq_u8(s01, s23)));
        pos += 64;
    }
    while pos + 16 <= full_bytes {
        let v0 = unsafe { vld1q_u8(bm_ptr.add(pos)) };
        acc_vec = vaddq_u16(acc_vec, vpaddlq_u8(vcntq_u8(v0)));
        pos += 16;
    }
    let mut sum = vaddlvq_u16(acc_vec) as usize;
    while pos < full_bytes {
        sum += usize::from(POPCNT8[usize::from(bm[pos])]);
        pos += 1;
    }
    if partial_bits > 0 {
        let mask = (1u8 << partial_bits) - 1;
        sum += usize::from(POPCNT8[usize::from(bm[full_bytes] & mask)]);
    }
    sum
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn merge_neon_16b(dest: *mut u8, l_ptr: *const u8, r_ptr: *const u8, mask16: usize) {
    let shuf0 = unsafe { vld1q_s8(G_MERGE_SHUF0.0[mask16 & 0xFF].as_ptr()) };
    let shuf1 = unsafe { vld1q_s8(G_MERGE_SHUF1.0[(mask16 >> 8) & 0xFF].as_ptr()) };
    let shuf = vreinterpretq_u8_s8(vabdq_s8(shuf0, shuf1));
    let src = uint8x16x2_t(unsafe { vld1q_u8(r_ptr) }, unsafe { vld1q_u8(l_ptr) });
    unsafe { vst1q_u8(dest, vqtbl2q_u8(src, shuf)) };
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn merge_neon_8b_lo(dest: *mut u8, l_ptr: *const u8, r_ptr: *const u8, m8: usize) {
    let shuf0 = unsafe { vld1q_s8(G_MERGE_SHUF0.0[m8 & 0xFF].as_ptr()) };
    let shuf1 = unsafe { vld1q_s8(G_MERGE_SHUF1.0[0].as_ptr()) };
    let shuf = vreinterpretq_u8_s8(vabdq_s8(shuf0, shuf1));
    let zero8 = vdup_n_u8(0);
    let src = uint8x16x2_t(
        vcombine_u8(unsafe { vld1_u8(r_ptr) }, zero8),
        vcombine_u8(unsafe { vld1_u8(l_ptr) }, zero8),
    );
    unsafe { vst1_u8(dest, vget_low_u8(vqtbl2q_u8(src, shuf))) };
}

/// 1-iter software-pipelined 64B/iter NEON `SABD` two-table merge (`neon.h:156-225`).
#[target_feature(enable = "neon")]
pub(super) unsafe fn merge_vec_vec_neon(
    bm: &[u8],
    k: usize,
    l_base: *const u8,
    l_len: usize,
    r_base: *const u8,
    r_len: usize,
    out_ptr: *mut u8,
) {
    let mut lc = 0usize;
    let mut rc = 0usize;
    let mut idx = 0usize;
    let bm_ptr = bm.as_ptr();

    // Preload iteration 0's mask + prefix popcount; pipeline next iter under current 4 merges.
    if idx + 64 <= k {
        let mut mask = unsafe { bm_ptr.cast::<u64>().read_unaligned() }.to_le();
        let mut pfx = vget_lane_u64::<0>(vreinterpret_u64_u8(vcnt_u8(unsafe { vld1_u8(bm_ptr) })))
            .wrapping_mul(0x0101_0101_0101_0101);

        while idx + 128 <= k {
            let p3 = (pfx >> 56) as usize;
            if lc + (64 - p3) + 16 > l_len || rc + p3 + 16 > r_len {
                break;
            }
            let next_bm = unsafe { bm_ptr.add((idx + 64) >> 3) };
            let next_mask = unsafe { next_bm.cast::<u64>().read_unaligned() }.to_le();
            let next_pfx =
                vget_lane_u64::<0>(vreinterpret_u64_u8(vcnt_u8(unsafe { vld1_u8(next_bm) })))
                    .wrapping_mul(0x0101_0101_0101_0101);

            let p0 = ((pfx >> 8) & 0xFF) as usize;
            let p1 = ((pfx >> 24) & 0xFF) as usize;
            let p2 = ((pfx >> 40) & 0xFF) as usize;
            unsafe {
                merge_neon_16b(
                    out_ptr.add(idx),
                    l_base.add(lc),
                    r_base.add(rc),
                    mask as usize,
                );
                merge_neon_16b(
                    out_ptr.add(idx + 16),
                    l_base.add(lc + 16 - p0),
                    r_base.add(rc + p0),
                    (mask >> 16) as usize,
                );
                merge_neon_16b(
                    out_ptr.add(idx + 32),
                    l_base.add(lc + 32 - p1),
                    r_base.add(rc + p1),
                    (mask >> 32) as usize,
                );
                merge_neon_16b(
                    out_ptr.add(idx + 48),
                    l_base.add(lc + 48 - p2),
                    r_base.add(rc + p2),
                    (mask >> 48) as usize,
                );
            }
            rc += p3;
            lc += 64 - p3;
            idx += 64;
            mask = next_mask;
            pfx = next_pfx;
        }
        let p3 = (pfx >> 56) as usize;
        if idx + 64 <= k && lc + (64 - p3) + 16 <= l_len && rc + p3 + 16 <= r_len {
            let p0 = ((pfx >> 8) & 0xFF) as usize;
            let p1 = ((pfx >> 24) & 0xFF) as usize;
            let p2 = ((pfx >> 40) & 0xFF) as usize;
            unsafe {
                merge_neon_16b(
                    out_ptr.add(idx),
                    l_base.add(lc),
                    r_base.add(rc),
                    mask as usize,
                );
                merge_neon_16b(
                    out_ptr.add(idx + 16),
                    l_base.add(lc + 16 - p0),
                    r_base.add(rc + p0),
                    (mask >> 16) as usize,
                );
                merge_neon_16b(
                    out_ptr.add(idx + 32),
                    l_base.add(lc + 32 - p1),
                    r_base.add(rc + p1),
                    (mask >> 32) as usize,
                );
                merge_neon_16b(
                    out_ptr.add(idx + 48),
                    l_base.add(lc + 48 - p2),
                    r_base.add(rc + p2),
                    (mask >> 48) as usize,
                );
            }
            rc += p3;
            lc += 64 - p3;
            idx += 64;
        }
    }

    while idx + 16 <= k && lc + 16 <= l_len && rc + 16 <= r_len {
        let m16 = unsafe { bm_ptr.add(idx >> 3).cast::<u16>().read_unaligned() }.to_le() as usize;
        unsafe { merge_neon_16b(out_ptr.add(idx), l_base.add(lc), r_base.add(rc), m16) };
        let pop = usize::from(POPCNT8[m16 & 0xFF]) + usize::from(POPCNT8[m16 >> 8]);
        rc += pop;
        lc += 16 - pop;
        idx += 16;
    }
    if idx + 8 <= k && lc + 8 <= l_len && rc + 8 <= r_len {
        let m8 = usize::from(bm[idx >> 3]);
        unsafe { merge_neon_8b_lo(out_ptr.add(idx), l_base.add(lc), r_base.add(rc), m8) };
        let pop = usize::from(POPCNT8[m8]);
        rc += pop;
        lc += 8 - pop;
        idx += 8;
    }
    unsafe { merge_vec_vec_tail_raw(bm, idx, k, l_base, lc, r_base, rc, out_ptr) };
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn merge_cst_16b(dst: *mut u8, v_ptr: *const u8, cst_vec: uint8x16_t, mask16: usize) {
    let s0 = unsafe { vld1q_s8(G_MERGE_SHUF0.0[mask16 & 0xFF].as_ptr()) };
    let s1 = unsafe { vld1q_s8(G_MERGE_SHUF1.0[(mask16 >> 8) & 0xFF].as_ptr()) };
    let sh = vreinterpretq_u8_s8(vabdq_s8(s0, s1));
    let src = uint8x16x2_t(unsafe { vld1q_u8(v_ptr) }, cst_vec);
    unsafe { vst1q_u8(dst, vqtbl2q_u8(src, sh)) };
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn merge_cst_8b_lo(dst: *mut u8, v_ptr: *const u8, cst_vec: uint8x16_t, m8: usize) {
    let s0 = unsafe { vld1q_s8(G_MERGE_SHUF0.0[m8 & 0xFF].as_ptr()) };
    let s1 = unsafe { vld1q_s8(G_MERGE_SHUF1.0[0].as_ptr()) };
    let sh = vreinterpretq_u8_s8(vabdq_s8(s0, s1));
    let src = uint8x16x2_t(
        vcombine_u8(unsafe { vld1_u8(v_ptr) }, vdup_n_u8(0)),
        cst_vec,
    );
    unsafe { vst1_u8(dst, vget_low_u8(vqtbl2q_u8(src, sh))) };
}

/// Unified NEON constant-plus-vector merge (`INVERT = false` => `cst_vec`, `true` => `vec_cst`).
#[target_feature(enable = "neon")]
pub(super) unsafe fn merge_cst_side_neon<const INVERT: bool>(
    bm: &[u8],
    k: usize,
    cst_sym: u8,
    v_base: *const u8,
    v_len: usize,
    out_ptr: *mut u8,
) {
    let cst_vec = vdupq_n_u8(cst_sym);
    let mut vc = 0usize;
    let mut idx = 0usize;
    let bm_ptr = bm.as_ptr();

    while idx + 64 <= k {
        let raw_mask = unsafe { bm_ptr.add(idx >> 3).cast::<u64>().read_unaligned() }.to_le();
        let mask = if INVERT { !raw_mask } else { raw_mask };
        let bm_vec = unsafe { vld1_u8(bm_ptr.add(idx >> 3)) };
        let pop_vec = vcnt_u8(if INVERT { vmvn_u8(bm_vec) } else { bm_vec });
        let pfx =
            vget_lane_u64::<0>(vreinterpret_u64_u8(pop_vec)).wrapping_mul(0x0101_0101_0101_0101);
        let p3 = (pfx >> 56) as usize;
        if vc + p3 + 16 > v_len {
            break;
        }
        let p0 = ((pfx >> 8) & 0xFF) as usize;
        let p1 = ((pfx >> 24) & 0xFF) as usize;
        let p2 = ((pfx >> 40) & 0xFF) as usize;

        unsafe {
            merge_cst_16b(out_ptr.add(idx), v_base.add(vc), cst_vec, mask as usize);
            merge_cst_16b(
                out_ptr.add(idx + 16),
                v_base.add(vc + p0),
                cst_vec,
                (mask >> 16) as usize,
            );
            merge_cst_16b(
                out_ptr.add(idx + 32),
                v_base.add(vc + p1),
                cst_vec,
                (mask >> 32) as usize,
            );
            merge_cst_16b(
                out_ptr.add(idx + 48),
                v_base.add(vc + p2),
                cst_vec,
                (mask >> 48) as usize,
            );
        }
        vc += p3;
        idx += 64;
    }
    while idx + 16 <= k && vc + 16 <= v_len {
        let raw_m16 = unsafe { bm_ptr.add(idx >> 3).cast::<u16>().read_unaligned() }.to_le();
        let m16 = usize::from(if INVERT { !raw_m16 } else { raw_m16 });
        unsafe { merge_cst_16b(out_ptr.add(idx), v_base.add(vc), cst_vec, m16) };
        vc += usize::from(POPCNT8[m16 & 0xFF]) + usize::from(POPCNT8[m16 >> 8]);
        idx += 16;
    }
    if idx + 8 <= k && vc + 8 <= v_len {
        let raw_m8 = bm[idx >> 3];
        let m8 = usize::from(if INVERT { !raw_m8 } else { raw_m8 });
        unsafe { merge_cst_8b_lo(out_ptr.add(idx), v_base.add(vc), cst_vec, m8) };
        vc += usize::from(POPCNT8[m8]);
        idx += 8;
    }
    unsafe { merge_cst_side_tail_raw::<INVERT>(bm, idx, k, cst_sym, v_base, vc, out_ptr) };
}

/// Table-free D=1 `vqtbl1q` bit-spread `merge_cst_cst_neon` (`neon.h:297-334`).
#[target_feature(enable = "neon")]
pub(super) unsafe fn merge_cst_cst_neon(
    bm: &[u8],
    k: usize,
    left_sym: u8,
    right_sym: u8,
    out: &mut [u8],
) {
    static DUP_TAB: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
    static SHIFT_TAB: [i8; 16] = [0, -1, -2, -3, -4, -5, -6, -7, 0, -1, -2, -3, -4, -5, -6, -7];

    let lr_word = u16::from(left_sym) | (u16::from(right_sym) << 8);
    let c2s_vec = vreinterpretq_u8_u16(vdupq_n_u16(lr_word));
    let dup_vec = unsafe { vld1q_u8(DUP_TAB.as_ptr()) };
    let shift_vec = unsafe { vld1q_s8(SHIFT_TAB.as_ptr()) };
    let one_vec = vdupq_n_u8(1);
    let bm_ptr = bm.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut pos = 0usize;

    while pos + 16 <= k {
        let bm_word = unsafe { bm_ptr.add(pos >> 3).cast::<u16>().read_unaligned() }.to_le();
        let bm_lo = vreinterpretq_u8_u16(vsetq_lane_u16::<0>(bm_word, vdupq_n_u16(0)));
        let dup = vqtbl1q_u8(bm_lo, dup_vec);
        let idx = vandq_u8(vshlq_u8(dup, shift_vec), one_vec);
        unsafe { vst1q_u8(out_ptr.add(pos), vqtbl1q_u8(c2s_vec, idx)) };
        pos += 16;
    }
    if pos + 8 <= k {
        let bm_vec = vdup_n_u8(bm[pos >> 3]);
        let dup = vtbl1_u8(bm_vec, vget_low_u8(dup_vec));
        let idx = vand_u8(vshl_u8(dup, vget_low_s8(shift_vec)), vget_low_u8(one_vec));
        unsafe { vst1_u8(out_ptr.add(pos), vtbl1_u8(vget_low_u8(c2s_vec), idx)) };
        pos += 8;
    }
    while pos < k {
        let mb = (bm[pos >> 3] >> (pos & 7)) & 1;
        out[pos] = if mb != 0 { right_sym } else { left_sym };
        pos += 1;
    }
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn flat_d2_unpack_16(bm_ptr: *const u8) -> uint8x16_t {
    static DUP_TAB: [u8; 16] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3];
    static SHIFT_TAB: [i8; 16] = [0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6];
    let packed = unsafe { bm_ptr.cast::<u32>().read_unaligned() }.to_le();
    let bm_lo = vreinterpretq_u8_u32(vsetq_lane_u32::<0>(packed, vdupq_n_u32(0)));
    let dup = vqtbl1q_u8(bm_lo, unsafe { vld1q_u8(DUP_TAB.as_ptr()) });
    vandq_u8(
        vshlq_u8(dup, unsafe { vld1q_s8(SHIFT_TAB.as_ptr()) }),
        vdupq_n_u8(0x03),
    )
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn unpack_flat_d2_neon(
    len: usize,
    packed: &[u8],
    c2s: &[u8; 64],
    out: &mut [u8],
) {
    static TH_IDX: [u8; 16] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3];
    let mut idx = 0usize;
    let p_ptr = packed.as_ptr();
    let out_ptr = out.as_mut_ptr();

    if len >= 64 {
        let word = u32::from_le_bytes([c2s[0], c2s[1], c2s[2], c2s[3]]);
        let tab_lo = vreinterpretq_u8_u32(vdupq_n_u32(word));
        let tab_hi = vqtbl1q_u8(tab_lo, unsafe { vld1q_u8(TH_IDX.as_ptr()) });
        let m0f = vdupq_n_u8(0x0F);
        while idx + 64 <= len {
            let vec = unsafe { vld1q_u8(p_ptr.add(idx >> 2)) };
            let lo = vandq_u8(vec, m0f);
            let hi = vshrq_n_u8::<4>(vec);
            let quad = uint8x16x4_t(
                vqtbl1q_u8(tab_lo, lo),
                vqtbl1q_u8(tab_hi, lo),
                vqtbl1q_u8(tab_lo, hi),
                vqtbl1q_u8(tab_hi, hi),
            );
            unsafe { vst4q_u8(out_ptr.add(idx), quad) };
            idx += 64;
        }
    }
    let c2s_vec = unsafe { vld1q_u8(c2s.as_ptr()) };
    while idx + 16 <= len {
        let codes = unsafe { flat_d2_unpack_16(p_ptr.add(idx >> 2)) };
        unsafe { vst1q_u8(out_ptr.add(idx), vqtbl1q_u8(c2s_vec, codes)) };
        idx += 16;
    }
    unpack_flat_tail_scalar(2, idx, len, packed, c2s, out);
}

/// Safe 8-code D=3 unpack via byte-wise `vsetq_lane_u8` (`neon_flat.h:66-77`).
#[target_feature(enable = "neon")]
#[inline]
unsafe fn flat_d3_unpack_safe(bm_ptr: *const u8) -> uint8x8_t {
    static SHUF_TAB: [u8; 16] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 2, 1, 2, 1, 2];
    static SHIFT_TAB: [i16; 8] = [0, -3, -6, -9, -12, -7, -10, -13];
    let mut bm_lo = vdupq_n_u8(0);
    bm_lo = vsetq_lane_u8::<0>(unsafe { *bm_ptr }, bm_lo);
    bm_lo = vsetq_lane_u8::<1>(unsafe { *bm_ptr.add(1) }, bm_lo);
    bm_lo = vsetq_lane_u8::<2>(unsafe { *bm_ptr.add(2) }, bm_lo);
    let shuffled = vqtbl1q_u8(bm_lo, unsafe { vld1q_u8(SHUF_TAB.as_ptr()) });
    let shifted = vshlq_u16(vreinterpretq_u16_u8(shuffled), unsafe {
        vld1q_s16(SHIFT_TAB.as_ptr())
    });
    vmovn_u16(vandq_u16(shifted, vdupq_n_u16(0x07)))
}

/// NEON pair-gather D=3 flat unpack (`neon.h:447-495`).
#[target_feature(enable = "neon")]
pub(super) unsafe fn unpack_flat_d3_neon(
    len: usize,
    packed: &[u8],
    c2s: &[u8; 64],
    out: &mut [u8],
) {
    let c2s8 = unsafe { vld1_u8(c2s.as_ptr()) };
    let c2s16 = vcombine_u8(c2s8, c2s8);
    let m7 = vdupq_n_u8(7);
    let p_ptr = packed.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut idx = 0usize;
    let mut byte_pos = 0usize;

    if len >= 48 {
        static PAIR6_SHUF: [u8; 16] = [0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11];
        static HSHIFT6: [i16; 8] = [2, -2, 2, -2, 2, -2, 2, -2];
        static BSHR6: [i8; 16] = [-2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0];
        let c2s32 = uint8x16x2_t(c2s16, c2s16);
        let shuf = unsafe { vld1q_u8(PAIR6_SHUF.as_ptr()) };
        let hshift = unsafe { vld1q_s16(HSHIFT6.as_ptr()) };
        let bshr = unsafe { vld1q_s8(BSHR6.as_ptr()) };
        while idx + 32 <= len && byte_pos + 16 <= packed.len() {
            let raw = unsafe { vld1q_u8(p_ptr.add(byte_pos)) };
            let words = vshlq_u16(vreinterpretq_u16_u8(vqtbl1q_u8(raw, shuf)), hshift);
            let pair6 = vshlq_u8(vreinterpretq_u8_u16(words), bshr);
            let pair_out = uint8x16x2_t(
                vqtbl1q_u8(c2s16, vandq_u8(pair6, m7)),
                vqtbl2q_u8(c2s32, vshrq_n_u8::<3>(pair6)),
            );
            unsafe { vst2q_u8(out_ptr.add(idx), pair_out) };
            idx += 32;
            byte_pos += 12;
        }
    }
    if idx + 16 <= len && byte_pos + 16 <= packed.len() {
        static PAIR_SHUF: [u8; 16] = [0, 1, 0, 1, 1, 2, 2, 3, 3, 4, 3, 4, 4, 5, 5, 6];
        static HSHIFT: [i16; 8] = [5, -1, 1, 3, 5, -1, 1, 3];
        static BSHR: [i8; 16] = [-5, 0, -5, 0, -5, 0, -5, 0, -5, 0, -5, 0, -5, 0, -5, 0];
        let raw = unsafe { vld1q_u8(p_ptr.add(byte_pos)) };
        let words = vshlq_u16(
            vreinterpretq_u16_u8(vqtbl1q_u8(raw, unsafe { vld1q_u8(PAIR_SHUF.as_ptr()) })),
            unsafe { vld1q_s16(HSHIFT.as_ptr()) },
        );
        let shifted = vshlq_u8(vreinterpretq_u8_u16(words), unsafe {
            vld1q_s8(BSHR.as_ptr())
        });
        unsafe { vst1q_u8(out_ptr.add(idx), vqtbl1q_u8(c2s16, vandq_u8(shifted, m7))) };
        idx += 16;
    }
    while idx + 8 <= len {
        let codes = unsafe { flat_d3_unpack_safe(p_ptr.add((idx * 3) >> 3)) };
        unsafe { vst1_u8(out_ptr.add(idx), vtbl1_u8(c2s8, codes)) };
        idx += 8;
    }
    unpack_flat_tail_scalar(3, idx, len, packed, c2s, out);
}

/// NEON D=4 flat unpack via `vzip1q`/`vzip2q` + `2x vst1q` (`neon.h:499-527`).
#[target_feature(enable = "neon")]
pub(super) unsafe fn unpack_flat_d4_neon(
    len: usize,
    packed: &[u8],
    c2s: &[u8; 64],
    out: &mut [u8],
) {
    static DUP_TAB: [u8; 16] = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7];
    static SHIFT_TAB: [i8; 16] = [0, -4, 0, -4, 0, -4, 0, -4, 0, -4, 0, -4, 0, -4, 0, -4];
    let tab = unsafe { vld1q_u8(c2s.as_ptr()) };
    let m0f = vdupq_n_u8(0x0F);
    let p_ptr = packed.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut idx = 0usize;

    while idx + 32 <= len {
        let vec = unsafe { vld1q_u8(p_ptr.add(idx >> 1)) };
        let lo = vqtbl1q_u8(tab, vandq_u8(vec, m0f));
        let hi = vqtbl1q_u8(tab, vshrq_n_u8::<4>(vec));
        unsafe {
            vst1q_u8(out_ptr.add(idx), vzip1q_u8(lo, hi));
            vst1q_u8(out_ptr.add(idx + 16), vzip2q_u8(lo, hi));
        }
        idx += 32;
    }
    while idx + 16 <= len {
        let raw = unsafe { p_ptr.add(idx >> 1).cast::<u64>().read_unaligned() }.to_le();
        let bm_lo = vreinterpretq_u8_u64(vsetq_lane_u64::<0>(raw, vdupq_n_u64(0)));
        let dup = vqtbl1q_u8(bm_lo, unsafe { vld1q_u8(DUP_TAB.as_ptr()) });
        let codes = vandq_u8(vshlq_u8(dup, unsafe { vld1q_s8(SHIFT_TAB.as_ptr()) }), m0f);
        unsafe { vst1q_u8(out_ptr.add(idx), vqtbl1q_u8(tab, codes)) };
        idx += 16;
    }
    unpack_flat_tail_scalar(4, idx, len, packed, c2s, out);
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn flat_d5_unpack_safe(bm_ptr: *const u8) -> uint8x8_t {
    static SHUF_TAB: [u8; 16] = [0, 1, 0, 1, 0, 1, 1, 2, 2, 3, 2, 3, 3, 4, 3, 4];
    static SHIFT_TAB: [i16; 8] = [0, -5, -10, -7, -4, -9, -6, -11];
    let mut bm_lo = vdupq_n_u8(0);
    bm_lo = vsetq_lane_u8::<0>(unsafe { *bm_ptr }, bm_lo);
    bm_lo = vsetq_lane_u8::<1>(unsafe { *bm_ptr.add(1) }, bm_lo);
    bm_lo = vsetq_lane_u8::<2>(unsafe { *bm_ptr.add(2) }, bm_lo);
    bm_lo = vsetq_lane_u8::<3>(unsafe { *bm_ptr.add(3) }, bm_lo);
    bm_lo = vsetq_lane_u8::<4>(unsafe { *bm_ptr.add(4) }, bm_lo);
    let shuffled = vqtbl1q_u8(bm_lo, unsafe { vld1q_u8(SHUF_TAB.as_ptr()) });
    let shifted = vshlq_u16(vreinterpretq_u16_u8(shuffled), unsafe {
        vld1q_s16(SHIFT_TAB.as_ptr())
    });
    vmovn_u16(vandq_u16(shifted, vdupq_n_u16(0x1F)))
}

/// NEON pair-gather D=5 flat unpack (`neon.h:533-569`).
#[target_feature(enable = "neon")]
pub(super) unsafe fn unpack_flat_d5_neon(
    len: usize,
    packed: &[u8],
    c2s: &[u8; 64],
    out: &mut [u8],
) {
    let c2s_vec = uint8x16x2_t(unsafe { vld1q_u8(c2s.as_ptr()) }, unsafe {
        vld1q_u8(c2s.as_ptr().add(16))
    });
    let p_ptr = packed.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut idx = 0usize;
    let mut byte_pos = 0usize;

    if len >= 16 && packed.len() >= 16 {
        static PAIR_SHUF: [u8; 16] = [0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8, 8, 9];
        static HSHIFT: [i16; 8] = [3, 1, -1, -3, 3, 1, -1, -3];
        static BSHR: [i8; 16] = [-3, 0, -3, 0, -3, 0, -3, 0, -3, 0, -3, 0, -3, 0, -3, 0];
        let shuf = unsafe { vld1q_u8(PAIR_SHUF.as_ptr()) };
        let hshift = unsafe { vld1q_s16(HSHIFT.as_ptr()) };
        let bshr = unsafe { vld1q_s8(BSHR.as_ptr()) };
        let m31 = vdupq_n_u8(0x1F);
        while idx + 16 <= len && byte_pos + 16 <= packed.len() {
            let raw = unsafe { vld1q_u8(p_ptr.add(byte_pos)) };
            let words = vshlq_u16(vreinterpretq_u16_u8(vqtbl1q_u8(raw, shuf)), hshift);
            let codes = vandq_u8(vshlq_u8(vreinterpretq_u8_u16(words), bshr), m31);
            unsafe { vst1q_u8(out_ptr.add(idx), vqtbl2q_u8(c2s_vec, codes)) };
            idx += 16;
            byte_pos += 10;
        }
    }
    while idx + 8 <= len {
        let codes = unsafe { flat_d5_unpack_safe(p_ptr.add((idx * 5) >> 3)) };
        unsafe { vst1_u8(out_ptr.add(idx), vqtbl2_u8(c2s_vec, codes)) };
        idx += 8;
    }
    unpack_flat_tail_scalar(5, idx, len, packed, c2s, out);
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn flat_d6_unpack_safe(bm_ptr: *const u8) -> uint8x8_t {
    static SHUF_TAB: [u8; 16] = [0, 1, 0, 1, 1, 2, 1, 2, 3, 4, 3, 4, 4, 5, 4, 5];
    static SHIFT_TAB: [i16; 8] = [0, -6, -4, -10, 0, -6, -4, -10];
    let mut bm_lo = vdupq_n_u8(0);
    bm_lo = vsetq_lane_u8::<0>(unsafe { *bm_ptr }, bm_lo);
    bm_lo = vsetq_lane_u8::<1>(unsafe { *bm_ptr.add(1) }, bm_lo);
    bm_lo = vsetq_lane_u8::<2>(unsafe { *bm_ptr.add(2) }, bm_lo);
    bm_lo = vsetq_lane_u8::<3>(unsafe { *bm_ptr.add(3) }, bm_lo);
    bm_lo = vsetq_lane_u8::<4>(unsafe { *bm_ptr.add(4) }, bm_lo);
    bm_lo = vsetq_lane_u8::<5>(unsafe { *bm_ptr.add(5) }, bm_lo);
    let shuffled = vqtbl1q_u8(bm_lo, unsafe { vld1q_u8(SHUF_TAB.as_ptr()) });
    let shifted = vshlq_u16(vreinterpretq_u16_u8(shuffled), unsafe {
        vld1q_s16(SHIFT_TAB.as_ptr())
    });
    vmovn_u16(vandq_u16(shifted, vdupq_n_u16(0x3F)))
}

/// NEON pair-gather D=6 flat unpack (`neon.h:573-611`).
#[target_feature(enable = "neon")]
pub(super) unsafe fn unpack_flat_d6_neon(
    len: usize,
    packed: &[u8],
    c2s: &[u8; 64],
    out: &mut [u8],
) {
    let c2s_vec = uint8x16x4_t(
        unsafe { vld1q_u8(c2s.as_ptr()) },
        unsafe { vld1q_u8(c2s.as_ptr().add(16)) },
        unsafe { vld1q_u8(c2s.as_ptr().add(32)) },
        unsafe { vld1q_u8(c2s.as_ptr().add(48)) },
    );
    let p_ptr = packed.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut idx = 0usize;
    let mut byte_pos = 0usize;

    if len >= 16 && packed.len() >= 16 {
        static PAIR_SHUF: [u8; 16] = [0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11];
        static HSHIFT: [i16; 8] = [2, -2, 2, -2, 2, -2, 2, -2];
        static BSHR: [i8; 16] = [-2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0];
        let shuf = unsafe { vld1q_u8(PAIR_SHUF.as_ptr()) };
        let hshift = unsafe { vld1q_s16(HSHIFT.as_ptr()) };
        let bshr = unsafe { vld1q_s8(BSHR.as_ptr()) };
        let m63 = vdupq_n_u8(0x3F);
        while idx + 16 <= len && byte_pos + 16 <= packed.len() {
            let raw = unsafe { vld1q_u8(p_ptr.add(byte_pos)) };
            let words = vshlq_u16(vreinterpretq_u16_u8(vqtbl1q_u8(raw, shuf)), hshift);
            let codes = vandq_u8(vshlq_u8(vreinterpretq_u8_u16(words), bshr), m63);
            unsafe { vst1q_u8(out_ptr.add(idx), vqtbl4q_u8(c2s_vec, codes)) };
            idx += 16;
            byte_pos += 12;
        }
    }
    while idx + 8 <= len {
        let codes = unsafe { flat_d6_unpack_safe(p_ptr.add((idx * 6) >> 3)) };
        unsafe { vst1_u8(out_ptr.add(idx), vqtbl4_u8(c2s_vec, codes)) };
        idx += 8;
    }
    unpack_flat_tail_scalar(6, idx, len, packed, c2s, out);
}

/// NEON SIMD flat packers for `D in 2..=6` (`neon.h:1189-1291` + `neon_pack.h`).
#[target_feature(enable = "neon")]
pub(super) unsafe fn pack_flat_codes_neon(
    depth: u8,
    ranks: &[u8],
    base_rank: u8,
    out: &mut [u8],
) -> usize {
    let len = ranks.len();
    let total_bytes = out.len();
    let r_ptr = ranks.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut idx = 0usize;

    match depth {
        2 => {
            static SHIFTS_D2: [i8; 16] = [0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6];
            let sh = unsafe { vld1q_s8(SHIFTS_D2.as_ptr()) };
            let b85 = vdupq_n_u8(85u8.wrapping_mul(base_rank));
            while idx + 64 <= len {
                let b0 = vshlq_u8(unsafe { vld1q_u8(r_ptr.add(idx)) }, sh);
                let b1 = vshlq_u8(unsafe { vld1q_u8(r_ptr.add(idx + 16)) }, sh);
                let b2 = vshlq_u8(unsafe { vld1q_u8(r_ptr.add(idx + 32)) }, sh);
                let b3 = vshlq_u8(unsafe { vld1q_u8(r_ptr.add(idx + 48)) }, sh);
                let folded = vpaddq_u8(vpaddq_u8(b0, b1), vpaddq_u8(b2, b3));
                unsafe { vst1q_u8(out_ptr.add(idx >> 2), vsubq_u8(folded, b85)) };
                idx += 64;
            }
            while idx + 16 <= len {
                let b0 = vshlq_u8(unsafe { vld1q_u8(r_ptr.add(idx)) }, sh);
                let s1 = vpaddq_u8(b0, b0);
                let s2 = vsubq_u8(vpaddq_u8(s1, s1), b85);
                let packed4 = vgetq_lane_u32::<0>(vreinterpretq_u32_u8(s2));
                unsafe {
                    out_ptr
                        .add(idx >> 2)
                        .cast::<u32>()
                        .write_unaligned(packed4.to_le())
                };
                idx += 16;
            }
        }
        3 => {
            static SHIFTS_P: [i8; 16] = [0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3];
            static SHIFTS_1: [i8; 16] = [2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0];
            static SHIFTS_2: [i16; 8] = [2, -2, 2, -2, 2, -2, 2, -2];
            static SHIFTS_4: [i32; 4] = [4, -4, 4, -4];
            static COMPACT_D6: [u8; 16] = [
                1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 0xFF, 0xFF, 0xFF, 0xFF,
            ];
            let shp = unsafe { vld1q_s8(SHIFTS_P.as_ptr()) };
            let b9 = vdupq_n_u8(9u8.wrapping_mul(base_rank));
            let s1 = unsafe { vld1q_s8(SHIFTS_1.as_ptr()) };
            let s2 = unsafe { vld1q_s16(SHIFTS_2.as_ptr()) };
            let s3 = unsafe { vld1q_s32(SHIFTS_4.as_ptr()) };
            let compact = unsafe { vld1q_u8(COMPACT_D6.as_ptr()) };
            while idx + 32 <= len && ((idx * 3) >> 3) + 16 <= total_bytes {
                let b0 = vshlq_u8(unsafe { vld1q_u8(r_ptr.add(idx)) }, shp);
                let b1 = vshlq_u8(unsafe { vld1q_u8(r_ptr.add(idx + 16)) }, shp);
                let pair = vsubq_u8(vpaddq_u8(b0, b1), b9);
                let w16 = vreinterpretq_u16_u8(vshlq_u8(pair, s1));
                let w32 = vreinterpretq_u32_u16(vshlq_u16(w16, s2));
                let w64 = vreinterpretq_u64_u32(vshlq_u32(w32, s3));
                unsafe {
                    vst1q_u8(
                        out_ptr.add((idx * 3) >> 3),
                        vqtbl1q_u8(vreinterpretq_u8_u64(w64), compact),
                    );
                }
                idx += 32;
            }
        }
        4 => {
            static SHIFTS_D4: [i8; 16] = [0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4];
            let sh = unsafe { vld1q_s8(SHIFTS_D4.as_ptr()) };
            let b17 = vdupq_n_u8(17u8.wrapping_mul(base_rank));
            while idx + 32 <= len {
                let b0 = vshlq_u8(unsafe { vld1q_u8(r_ptr.add(idx)) }, sh);
                let b1 = vshlq_u8(unsafe { vld1q_u8(r_ptr.add(idx + 16)) }, sh);
                unsafe { vst1q_u8(out_ptr.add(idx >> 1), vsubq_u8(vpaddq_u8(b0, b1), b17)) };
                idx += 32;
            }
            while idx + 16 <= len {
                let b0 = vshlq_u8(unsafe { vld1q_u8(r_ptr.add(idx)) }, sh);
                let folded = vsubq_u8(vpaddq_u8(b0, b0), b17);
                unsafe { vst1_u8(out_ptr.add(idx >> 1), vget_low_u8(folded)) };
                idx += 16;
            }
        }
        5 => {
            static SH1: [i8; 16] = [3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0];
            static SH2: [i16; 8] = [3, -3, 3, -3, 3, -3, 3, -3];
            static SH3: [i32; 4] = [6, -6, 6, -6];
            static COMPACT: [u8; 16] = [
                1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            ];
            let s1 = unsafe { vld1q_s8(SH1.as_ptr()) };
            let s2 = unsafe { vld1q_s16(SH2.as_ptr()) };
            let s3 = unsafe { vld1q_s32(SH3.as_ptr()) };
            let compact = unsafe { vld1q_u8(COMPACT.as_ptr()) };
            let vbase = vdupq_n_u8(base_rank);
            let vmask = vdupq_n_u8(0x1F);
            while idx + 16 <= len && ((idx * 5) >> 3) + 16 <= total_bytes {
                let cb = vandq_u8(vsubq_u8(unsafe { vld1q_u8(r_ptr.add(idx)) }, vbase), vmask);
                let w16 = vreinterpretq_u16_u8(vshlq_u8(cb, s1));
                let w32 = vreinterpretq_u32_u16(vshlq_u16(w16, s2));
                let w64 = vshrq_n_u64::<4>(vreinterpretq_u64_u32(vshlq_u32(w32, s3)));
                unsafe {
                    vst1q_u8(
                        out_ptr.add((idx * 5) >> 3),
                        vqtbl1q_u8(vreinterpretq_u8_u64(w64), compact),
                    );
                }
                idx += 16;
            }
        }
        6 => {
            static SH1: [i8; 16] = [2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0];
            static SH2: [i16; 8] = [2, -2, 2, -2, 2, -2, 2, -2];
            static SH3: [i32; 4] = [4, -4, 4, -4];
            static COMPACT: [u8; 16] = [
                1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 0xFF, 0xFF, 0xFF, 0xFF,
            ];
            let s1 = unsafe { vld1q_s8(SH1.as_ptr()) };
            let s2 = unsafe { vld1q_s16(SH2.as_ptr()) };
            let s3 = unsafe { vld1q_s32(SH3.as_ptr()) };
            let compact = unsafe { vld1q_u8(COMPACT.as_ptr()) };
            let vbase = vdupq_n_u8(base_rank);
            let vmask = vdupq_n_u8(0x3F);
            while idx + 16 <= len && ((idx * 6) >> 3) + 16 <= total_bytes {
                let cb = vandq_u8(vsubq_u8(unsafe { vld1q_u8(r_ptr.add(idx)) }, vbase), vmask);
                let w16 = vreinterpretq_u16_u8(vshlq_u8(cb, s1));
                let w32 = vreinterpretq_u32_u16(vshlq_u16(w16, s2));
                let w64 = vreinterpretq_u64_u32(vshlq_u32(w32, s3));
                unsafe {
                    vst1q_u8(
                        out_ptr.add((idx * 6) >> 3),
                        vqtbl1q_u8(vreinterpretq_u8_u64(w64), compact),
                    );
                }
                idx += 16;
            }
        }
        _ => {}
    }
    idx
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn masks64v_ge_neon(
    v0: uint8x16_t,
    v1: uint8x16_t,
    v2: uint8x16_t,
    v3: uint8x16_t,
    vpivot: uint8x16_t,
    vweights: uint8x16_t,
) -> uint8x8_t {
    let w0 = vandq_u8(vcgeq_u8(v0, vpivot), vweights);
    let w1 = vandq_u8(vcgeq_u8(v1, vpivot), vweights);
    let w2 = vandq_u8(vcgeq_u8(v2, vpivot), vweights);
    let w3 = vandq_u8(vcgeq_u8(v3, vpivot), vweights);
    let t0 = vpaddq_u8(w0, w1);
    let t1 = vpaddq_u8(w2, w3);
    let u0 = vpaddq_u8(t0, t1);
    vget_low_u8(vpaddq_u8(u0, u0))
}

/// Packs 64 `0/1` bytes into one `u64` bitmask using the same 4x16B pairwise reduction as
/// [`masks64v_ge_neon`].
#[target_feature(enable = "neon")]
pub(super) unsafe fn pack_64_bytes01_neon(chunk64: &[u8]) -> u64 {
    let ptr = chunk64.as_ptr();
    let vone = vdupq_n_u8(1);
    let vweights = unsafe { vld1q_u8(BIT_WEIGHTS_16.as_ptr()) };
    let v0 = unsafe { vld1q_u8(ptr) };
    let v1 = unsafe { vld1q_u8(ptr.add(16)) };
    let v2 = unsafe { vld1q_u8(ptr.add(32)) };
    let v3 = unsafe { vld1q_u8(ptr.add(48)) };
    let w0 = vandq_u8(vtstq_u8(v0, vone), vweights);
    let w1 = vandq_u8(vtstq_u8(v1, vone), vweights);
    let w2 = vandq_u8(vtstq_u8(v2, vone), vweights);
    let w3 = vandq_u8(vtstq_u8(v3, vone), vweights);
    let t0 = vpaddq_u8(w0, w1);
    let t1 = vpaddq_u8(w2, w3);
    let u0 = vpaddq_u8(t0, t1);
    let packed = vget_low_u8(vpaddq_u8(u0, u0));
    vget_lane_u64::<0>(vreinterpret_u64_u8(packed))
}

/// 64-wide NEON encoder partition (`neon.h:1004-1166`).
#[target_feature(enable = "neon")]
pub(super) unsafe fn partition_ranks_neon(
    ranks: &[u8],
    pivot_rank: u8,
    bm_out: &mut [u8],
    left_out: &mut [u8],
    right_out: &mut [u8],
) -> usize {
    let len = ranks.len();
    let vpivot = vdupq_n_u8(pivot_rank);
    let vweights = unsafe { vld1q_u8(BIT_WEIGHTS_16.as_ptr()) };
    let r_ptr = ranks.as_ptr();
    let bm_ptr = bm_out.as_mut_ptr();
    let l_dst = left_out.as_mut_ptr();
    let r_dst = right_out.as_mut_ptr();

    let mut lc = 0usize;
    let mut rc = 0usize;
    let mut idx = 0usize;

    while idx + 64 <= len {
        let v0 = unsafe { vld1q_u8(r_ptr.add(idx)) };
        let v1 = unsafe { vld1q_u8(r_ptr.add(idx + 16)) };
        let v2 = unsafe { vld1q_u8(r_ptr.add(idx + 32)) };
        let v3 = unsafe { vld1q_u8(r_ptr.add(idx + 48)) };
        let mv = unsafe { masks64v_ge_neon(v0, v1, v2, v3, vpivot, vweights) };
        let mask_word = vget_lane_u64::<0>(vreinterpret_u64_u8(mv));
        unsafe {
            bm_ptr
                .add(idx >> 3)
                .cast::<u64>()
                .write_unaligned(mask_word.to_le())
        };

        let halves: [uint8x8_t; 8] = [
            vget_low_u8(v0),
            vget_high_u8(v0),
            vget_low_u8(v1),
            vget_high_u8(v1),
            vget_low_u8(v2),
            vget_high_u8(v2),
            vget_low_u8(v3),
            vget_high_u8(v3),
        ];
        for (slot, chunk) in halves.into_iter().enumerate() {
            let m8 = ((mask_word >> (slot * 8)) & 0xFF) as usize;
            unsafe {
                vst1_u8(
                    l_dst.add(lc),
                    vtbl1_u8(chunk, vld1_u8(COMPRESS_LEFT_TAB.0[m8].as_ptr())),
                );
                vst1_u8(
                    r_dst.add(rc),
                    vtbl1_u8(chunk, vld1_u8(COMPRESS_RIGHT_TAB.0[m8].as_ptr())),
                );
            }
            let pop = usize::from(POPCNT8[m8]);
            lc += 8 - pop;
            rc += pop;
        }
        idx += 64;
    }

    while idx + 16 <= len {
        let vec = unsafe { vld1q_u8(r_ptr.add(idx)) };
        let weighted = vandq_u8(vcgeq_u8(vec, vpivot), vweights);
        let p1 = vpaddq_u8(weighted, weighted);
        let p2 = vpaddq_u8(p1, p1);
        let p3 = vpaddq_u8(p2, p2);
        let m16 = vget_lane_u64::<0>(vreinterpret_u64_u8(vget_low_u8(p3))) as u16;
        let m0 = usize::from(m16 & 0xFF);
        let m1 = usize::from(m16 >> 8);
        bm_out[idx >> 3] = m0 as u8;
        bm_out[(idx >> 3) + 1] = m1 as u8;

        let vlo = vget_low_u8(vec);
        let vhi = vget_high_u8(vec);
        unsafe {
            vst1_u8(
                l_dst.add(lc),
                vtbl1_u8(vlo, vld1_u8(COMPRESS_LEFT_TAB.0[m0].as_ptr())),
            );
            vst1_u8(
                r_dst.add(rc),
                vtbl1_u8(vlo, vld1_u8(COMPRESS_RIGHT_TAB.0[m0].as_ptr())),
            );
        }
        let pop0 = usize::from(POPCNT8[m0]);
        lc += 8 - pop0;
        rc += pop0;

        unsafe {
            vst1_u8(
                l_dst.add(lc),
                vtbl1_u8(vhi, vld1_u8(COMPRESS_LEFT_TAB.0[m1].as_ptr())),
            );
            vst1_u8(
                r_dst.add(rc),
                vtbl1_u8(vhi, vld1_u8(COMPRESS_RIGHT_TAB.0[m1].as_ptr())),
            );
        }
        let pop1 = usize::from(POPCNT8[m1]);
        lc += 8 - pop1;
        rc += pop1;
        idx += 16;
    }

    if idx < len {
        bm_out[idx >> 3..len.div_ceil(8)].fill(0);
        while idx < len {
            let rank = ranks[idx];
            if rank >= pivot_rank {
                bm_out[idx >> 3] |= 1u8 << (idx & 7);
                right_out[rc] = rank;
                rc += 1;
            } else {
                left_out[lc] = rank;
                lc += 1;
            }
            idx += 1;
        }
    }
    rc
}
