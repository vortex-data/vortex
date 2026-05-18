//! aarch64 NEON kernels for `i32` add and equality.

use core::arch::aarch64::*;

use crate::kernels::scalar;

/// NEON `i32` add, 4 lanes per iteration.
///
/// # Safety
/// NEON must be available at runtime. Slices must be the same length.
#[target_feature(enable = "neon")]
pub unsafe fn add_i32_neon(a: &[i32], b: &[i32], out: &mut [i32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    let len = a.len();
    let bulk = len & !3;
    let mut idx = 0;
    while idx < bulk {
        // SAFETY: idx + 4 <= bulk <= len.
        unsafe {
            let vec_a = vld1q_s32(a.as_ptr().add(idx));
            let vec_b = vld1q_s32(b.as_ptr().add(idx));
            vst1q_s32(out.as_mut_ptr().add(idx), vaddq_s32(vec_a, vec_b));
        }
        idx += 4;
    }
    if idx < len {
        scalar::add_i32(&a[idx..], &b[idx..], &mut out[idx..]);
    }
}

/// NEON `i32` equality → packed bitmap.
///
/// NEON has no direct movemask, so we narrow `uint32x4_t` masks down to
/// `uint8x8_t` and use the shift-and-add trick to collect one bit per lane.
/// Eight lanes (two 4-wide compares) become one output byte.
///
/// # Safety
/// NEON must be available at runtime. Slices and bitmap must be sized
/// according to the contract.
#[target_feature(enable = "neon")]
pub unsafe fn eq_i32_neon(a: &[i32], b: &[i32], out: &mut [u8]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(out.len(), a.len().div_ceil(8));
    let len = a.len();
    out.fill(0);
    let bulk_bytes = len / 8;
    let mut idx = 0;
    let weights: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
    // SAFETY: stack-local 8-byte buffer.
    let lane_weights = unsafe { vld1_u8(weights.as_ptr()) };
    for byte_idx in 0..bulk_bytes {
        // SAFETY: idx + 8 <= len.
        unsafe {
            let a_lo = vld1q_s32(a.as_ptr().add(idx));
            let b_lo = vld1q_s32(b.as_ptr().add(idx));
            let a_hi = vld1q_s32(a.as_ptr().add(idx + 4));
            let b_hi = vld1q_s32(b.as_ptr().add(idx + 4));
            let mask_lo = vceqq_s32(a_lo, b_lo);
            let mask_hi = vceqq_s32(a_hi, b_hi);
            // Narrow 32-bit masks down to 8-bit masks: 4+4 lanes -> 8 lanes.
            let narrowed_lo = vmovn_u32(mask_lo);
            let narrowed_hi = vmovn_u32(mask_hi);
            let combined_16 = vcombine_u16(narrowed_lo, narrowed_hi);
            let narrowed_8 = vmovn_u16(combined_16);
            // Each lane is 0x00 or 0xFF — AND with weights leaves just the
            // bit at that lane's position. Horizontal add collects them.
            let weighted = vand_u8(narrowed_8, lane_weights);
            *out.get_unchecked_mut(byte_idx) = vaddv_u8(weighted);
        }
        idx += 8;
    }
    while idx < len {
        if a[idx] == b[idx] {
            out[idx / 8] |= 1 << (idx % 8);
        }
        idx += 1;
    }
}
