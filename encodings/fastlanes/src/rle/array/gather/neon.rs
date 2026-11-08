// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! NEON `tbl` window-permute kernels.

use core::arch::aarch64::uint8x8_t;
use core::arch::aarch64::uint8x16x2_t;
use core::arch::aarch64::vadd_u8;
use core::arch::aarch64::vaddq_u8;
use core::arch::aarch64::vcombine_u8;
use core::arch::aarch64::vdup_n_u8;
use core::arch::aarch64::vdupq_n_u16;
use core::arch::aarch64::vld1_u8;
use core::arch::aarch64::vld1q_u8;
use core::arch::aarch64::vld1q_u16;
use core::arch::aarch64::vmaxv_u8;
use core::arch::aarch64::vmaxvq_u16;
use core::arch::aarch64::vmovn_u16;
use core::arch::aarch64::vqtbl1q_u8;
use core::arch::aarch64::vqtbl2q_u8;
use core::arch::aarch64::vshl_n_u8;
use core::arch::aarch64::vst1_u8;
use core::arch::aarch64::vst1q_u8;
use core::arch::aarch64::vsub_u8;
use core::arch::aarch64::vsubq_u16;
use core::arch::aarch64::vtbl1_u8;
use core::arch::aarch64::vzip1q_u8;
use std::ptr;

use super::DELTA_LUT;
use super::GROUPS;
use super::U8_LANES;
use super::u8_lane_start;
use crate::FL_CHUNK_SIZE;

/// For 4-byte values: table rows that repeat each of 4 offsets 4 times, for outputs 0-3 and 4-7.
static DUP4: [[u8; 16]; 2] = [
    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
    [4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7],
];
static INC4: [u8; 16] = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
/// Gather one chunk with NEON. Returns `false`, writing nothing, for unsupported widths.
///
/// # Safety
///
/// Every index must be less than `values.len()`.
pub(super) unsafe fn gather_chunk<V: Copy, I: Copy + Into<usize>>(
    values: &[V],
    indices: &[I; FL_CHUNK_SIZE],
    out: &mut [V; FL_CHUNK_SIZE],
) -> bool {
    let vals = values.as_ptr().cast::<u8>();
    let len = values.len();
    let dst = out.as_mut_ptr().cast::<u8>();
    // SAFETY: the indices are bounded per this function's contract. `I` is `u8` or `u16`, the
    // only unsigned integer types of these sizes.
    unsafe {
        match (size_of::<V>(), size_of::<I>()) {
            (1, 1) => chunk_u8::<1>(vals, len, indices.as_ptr().cast(), dst),
            (2, 1) => chunk_u8::<2>(vals, len, indices.as_ptr().cast(), dst),
            (4, 1) => chunk_u8::<4>(vals, len, indices.as_ptr().cast(), dst),
            (1, 2) => chunk_u16::<1>(vals, len, indices.as_ptr().cast(), dst),
            (2, 2) => chunk_u16::<2>(vals, len, indices.as_ptr().cast(), dst),
            (4, 2) => chunk_u16::<4>(vals, len, indices.as_ptr().cast(), dst),
            _ => return false,
        }
    }
    true
}

/// Gather one chunk from `u8` Delta lanes with NEON. Returns `false`, writing nothing, for
/// unsupported widths.
///
/// # Safety
///
/// Each lane's last index must be less than `values.len()`.
pub(super) unsafe fn gather_delta_lanes<V: Copy>(
    values: &[V],
    bases: &[u8; U8_LANES],
    deltas: Option<&[u8; U8_LANES]>,
    out: &mut [V; FL_CHUNK_SIZE],
) -> bool {
    let vals = values.as_ptr().cast::<u8>();
    let len = values.len();
    let deltas = deltas.map_or(ptr::null(), |d| d.as_ptr());
    let dst = out.as_mut_ptr().cast::<u8>();
    // SAFETY: every lane is bounded per this function's contract.
    unsafe {
        match size_of::<V>() {
            1 => delta_lanes::<1>(vals, len, bases.as_ptr(), deltas, dst),
            2 => delta_lanes::<2>(vals, len, bases.as_ptr(), deltas, dst),
            4 => delta_lanes::<4>(vals, len, bases.as_ptr(), deltas, dst),
            _ => return false,
        }
    }
    true
}

/// Write the 8 values of `S` bytes at `window[rel[r]]` to `out`, where every `rel` is below 8.
///
/// # Safety
///
/// `window` must be readable for `8 * S` bytes and `out` writable for `8 * S` bytes.
#[inline]
unsafe fn permute8<const S: usize>(window: *const u8, rel: uint8x8_t, out: *mut u8) {
    // SAFETY: the caller guarantees `8 * S` readable and writable bytes; the loads and stores
    // below touch exactly that many.
    unsafe {
        match S {
            1 => vst1_u8(out, vtbl1_u8(vld1_u8(window), rel)),
            2 => {
                let lo = vadd_u8(rel, rel);
                let hi = vadd_u8(lo, vdup_n_u8(1));
                let idx = vzip1q_u8(vcombine_u8(lo, lo), vcombine_u8(hi, hi));
                vst1q_u8(out, vqtbl1q_u8(vld1q_u8(window), idx));
            }
            4 => {
                let table = uint8x16x2_t(vld1q_u8(window), vld1q_u8(window.add(16)));
                let base = vcombine_u8(vshl_n_u8::<2>(rel), vshl_n_u8::<2>(rel));
                for (half, dup) in DUP4.iter().enumerate() {
                    let idx = vaddq_u8(
                        vqtbl1q_u8(base, vld1q_u8(dup.as_ptr())),
                        vld1q_u8(INC4.as_ptr()),
                    );
                    vst1q_u8(out.add(half * 16), vqtbl2q_u8(table, idx));
                }
            }
            _ => unreachable!("unsupported value width {S}"),
        }
    }
}

/// Copy the `S`-byte values at `indices` to `out`, one at a time.
///
/// # Safety
///
/// Every index must be in bounds of `vals`, and `out` writable for `8 * S` bytes.
#[inline]
unsafe fn scalar_group<const S: usize>(
    vals: *const u8,
    indices: impl Iterator<Item = usize>,
    out: *mut u8,
) {
    for (row, index) in indices.enumerate() {
        // SAFETY: the caller bounds every index and the 8 output slots.
        unsafe { ptr::copy_nonoverlapping(vals.add(index * S), out.add(row * S), S) };
    }
}

/// # Safety
///
/// `vals` must hold `len` values of `S` bytes, `idx` must point to 1024 indices below `len`, and
/// `out` must be writable for 1024 values.
unsafe fn chunk_u8<const S: usize>(vals: *const u8, len: usize, idx: *const u8, out: *mut u8) {
    for group in 0..GROUPS {
        // SAFETY: `idx` holds 1024 indices, so each group's 8 are in bounds; `out` has room for
        // every group; the window is read only when it ends within `len` values.
        unsafe {
            let group_idx = idx.add(group * 8);
            let first = *group_idx;
            let rel = vsub_u8(vld1_u8(group_idx), vdup_n_u8(first));
            let group_out = out.add(group * 8 * S);
            if vmaxv_u8(rel) <= 7 && usize::from(first) + 8 <= len {
                permute8::<S>(vals.add(usize::from(first) * S), rel, group_out);
            } else {
                let indices = (0..8).map(|r| usize::from(*group_idx.add(r)));
                scalar_group::<S>(vals, indices, group_out);
            }
        }
    }
}

/// # Safety
///
/// As [`chunk_u8`], with `u16` indices.
unsafe fn chunk_u16<const S: usize>(vals: *const u8, len: usize, idx: *const u16, out: *mut u8) {
    for group in 0..GROUPS {
        // SAFETY: as in `chunk_u8`.
        unsafe {
            let group_idx = idx.add(group * 8);
            let first = *group_idx;
            let rel16 = vsubq_u16(vld1q_u16(group_idx), vdupq_n_u16(first));
            let group_out = out.add(group * 8 * S);
            if vmaxvq_u16(rel16) <= 7 && usize::from(first) + 8 <= len {
                permute8::<S>(
                    vals.add(usize::from(first) * S),
                    vmovn_u16(rel16),
                    group_out,
                );
            } else {
                let indices = (0..8).map(|r| usize::from(*group_idx.add(r)));
                scalar_group::<S>(vals, indices, group_out);
            }
        }
    }
}

/// # Safety
///
/// `vals` must hold `len` values of `S` bytes, `bases` must point to 128 bases, `deltas` to 128
/// delta bytes or be null, each lane's last index must be below `len`, and `out` must be writable
/// for 1024 values.
unsafe fn delta_lanes<const S: usize>(
    vals: *const u8,
    len: usize,
    bases: *const u8,
    deltas: *const u8,
    out: *mut u8,
) {
    for lane in 0..U8_LANES {
        // SAFETY: `bases` and `deltas` hold 128 bytes; every lane writes 8 values inside the
        // chunk; the window is read only when it ends within `len` values, and the caller bounds
        // every index the lane uses.
        unsafe {
            let bits = if deltas.is_null() {
                0
            } else {
                *deltas.add(lane)
            };
            let first = usize::from(*bases.add(lane)) + usize::from(bits & 1);
            let rel = &DELTA_LUT[usize::from(bits)];
            let lane_out = out.add(u8_lane_start(lane) * S);
            if first + 8 <= len {
                permute8::<S>(vals.add(first * S), vld1_u8(rel.as_ptr()), lane_out);
            } else {
                let indices = rel.iter().map(|&r| first + usize::from(r));
                scalar_group::<S>(vals, indices, lane_out);
            }
        }
    }
}
