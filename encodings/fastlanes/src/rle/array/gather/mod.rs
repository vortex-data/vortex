// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Gather kernels that expand one 1024-element RLE chunk.
//!
//! Within a chunk, consecutive RLE indices differ by 0 or 1, so the indices of any 8 consecutive
//! outputs span at most 8 values. A group of 8 outputs is therefore a permutation of the 8-value
//! window starting at the group's first index, which one vector shuffle produces.
//!
//! [`gather_chunk`] applies this to indices in their original order. [`gather_delta_lanes`]
//! decodes indices stored as FastLanes Delta over 1-bit deltas without materializing them: each
//! of the 128 `u8` lanes covers 8 consecutive outputs, its delta bits are one byte, and a lookup
//! table turns that byte into the lane's permutation.
//!
//! | Target    | 1-, 2-byte values | 4-byte values          | 8-byte values |
//! | --------- | ----------------- | ---------------------- | ------------- |
//! | x86 AVX2  | `pshufb`          | `vpermd`               | scalar        |
//! | aarch64   | `tbl`             | `tbl` over 2 registers | scalar        |
//! | otherwise | scalar            | scalar                 | scalar        |
//!
//! Eight-byte values stay scalar: the scalar gather already nearly saturates stores for them, and
//! an AVX2 permute of a 64-byte window (two `vpermd` and a blend per 4 values) measured twice as
//! slow.
//!
//! Groups whose indices spread by more than 7 (only possible in corrupt input), and groups whose
//! window would run past the end of the values, fall back to a scalar gather.

use fastlanes::FL_ORDER;

use crate::FL_CHUNK_SIZE;

#[cfg(all(target_arch = "aarch64", not(miri)))]
mod neon;
#[cfg(test)]
mod tests;
#[cfg(all(target_arch = "x86_64", not(miri)))]
mod x86;

/// Number of 8-output groups in a chunk.
const GROUPS: usize = FL_CHUNK_SIZE / 8;

/// Number of `u8` FastLanes lanes in a chunk.
pub(crate) const U8_LANES: usize = 128;

/// For each byte of 8 delta bits, the inclusive prefix sum of the bits minus the lowest bit.
///
/// A lane with base index `b` and delta byte `d` holds the values at `b + (d & 1) + LUT[d][r]`
/// for its rows `r`, and every entry is below 8, so the 8 values fit one window.
#[expect(
    clippy::cast_possible_truncation,
    reason = "entries are prefix sums of at most 8 bits"
)]
static DELTA_LUT: [[u8; 8]; 256] = {
    let mut lut = [[0u8; 8]; 256];
    let mut byte = 0;
    while byte < 256 {
        let mut sum = 0;
        let mut row = 0;
        while row < 8 {
            sum += (byte >> row) & 1;
            lut[byte][row] = (sum - (byte & 1)) as u8;
            row += 1;
        }
        byte += 1;
    }
    lut
};

/// The position of the first of the 8 consecutive outputs that `u8` lane `lane` covers.
///
/// FastLanes stores the transposed element at `row * 128 + lane` from original position
/// `(lane % 16) * 64 + FL_ORDER[lane / 16] * 8 + row`.
#[inline]
const fn u8_lane_start(lane: usize) -> usize {
    (lane % 16) * 64 + FL_ORDER[lane / 16] * 8
}

/// Write `out[i] = values[indices[i]]` for one chunk.
///
/// `I` must be `u8` or `u16`.
///
/// # Safety
///
/// Every index must be less than `values.len()`.
#[inline]
pub(crate) unsafe fn gather_chunk<V: Copy, I: Copy + Into<usize>>(
    values: &[V],
    indices: &[I; FL_CHUNK_SIZE],
    out: &mut [V; FL_CHUNK_SIZE],
) {
    debug_assert!(size_of::<I>() <= 2);
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 was detected, and the caller bounds every index by `values.len()`.
        if unsafe { x86::gather_chunk(values, indices, out) } {
            return;
        }
    }
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
        // SAFETY: NEON is part of the aarch64 baseline, and the caller bounds every index.
        if unsafe { neon::gather_chunk(values, indices, out) } {
            return;
        }
    }
    // SAFETY: the caller bounds every index by `values.len()`.
    unsafe { gather_chunk_scalar(values, indices, out) }
}

/// Write one chunk from indices stored as `u8` FastLanes Delta: `bases` holds one base per lane,
/// and `deltas` holds one byte of delta bits per lane, or `None` when every delta is zero.
///
/// # Safety
///
/// For every lane, `bases[lane]` plus the number of set bits in its delta byte must be less than
/// `values.len()`.
#[inline]
pub(crate) unsafe fn gather_delta_lanes<V: Copy>(
    values: &[V],
    bases: &[u8; U8_LANES],
    deltas: Option<&[u8; U8_LANES]>,
    out: &mut [V; FL_CHUNK_SIZE],
) {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 was detected, and the caller bounds every lane's last index.
        if unsafe { x86::gather_delta_lanes(values, bases, deltas, out) } {
            return;
        }
    }
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
        // SAFETY: NEON is part of the aarch64 baseline, and the caller bounds every lane.
        if unsafe { neon::gather_delta_lanes(values, bases, deltas, out) } {
            return;
        }
    }
    // SAFETY: the caller bounds every lane's last index by `values.len()`.
    unsafe { gather_delta_lanes_scalar(values, bases, deltas, out) }
}

/// # Safety
///
/// Every index must be less than `values.len()`.
unsafe fn gather_chunk_scalar<V: Copy, I: Copy + Into<usize>>(
    values: &[V],
    indices: &[I; FL_CHUNK_SIZE],
    out: &mut [V; FL_CHUNK_SIZE],
) {
    for (o, &index) in out.iter_mut().zip(indices) {
        let index: usize = index.into();
        debug_assert!(index < values.len());
        // SAFETY: the caller bounds every index by `values.len()`.
        *o = unsafe { *values.get_unchecked(index) };
    }
}

/// # Safety
///
/// See [`gather_delta_lanes`].
unsafe fn gather_delta_lanes_scalar<V: Copy>(
    values: &[V],
    bases: &[u8; U8_LANES],
    deltas: Option<&[u8; U8_LANES]>,
    out: &mut [V; FL_CHUNK_SIZE],
) {
    for lane in 0..U8_LANES {
        let bits = deltas.map_or(0, |d| d[lane]);
        let first = usize::from(bases[lane]) + usize::from(bits & 1);
        let start = u8_lane_start(lane);
        for (row, &rel) in DELTA_LUT[usize::from(bits)].iter().enumerate() {
            let index = first + usize::from(rel);
            debug_assert!(index < values.len());
            // SAFETY: `index` is at most the lane's last index, which the caller bounds.
            out[start + row] = unsafe { *values.get_unchecked(index) };
        }
    }
}
