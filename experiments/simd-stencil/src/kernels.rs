// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The shared stencil library.
//!
//! Every decode strategy is built from these kernels, so the only thing that
//! varies between strategies is *how* the kernels are composed, never the
//! kernels themselves. The integer kernels delegate to `fastlanes`, whose
//! per-bit-width unpack routines are themselves pre-compiled SIMD stencils.

use fastlanes::BitPacking;
use fastlanes::Delta;
use fastlanes::FoR;
use fastlanes::Transpose;

use crate::TILE;

/// `u32` lanes in a FastLanes tile (`1024 / 32`).
pub const LANES32: usize = 32;
/// `u64` lanes in a FastLanes tile (`1024 / 64`).
pub const LANES64: usize = 16;

const ZERO_BASE32: [u32; LANES32] = [0; LANES32];
const ZERO_BASE64: [u64; LANES64] = [0; LANES64];

/// Number of packed `u32` words a tile of `width`-bit values occupies.
#[inline(always)]
pub const fn packed_len_u32(width: usize) -> usize {
    TILE * width / 32
}

/// Number of packed `u64` words a tile of `width`-bit values occupies.
#[inline(always)]
pub const fn packed_len_u64(width: usize) -> usize {
    TILE * width / 64
}

/// Stencil: unpack a `width`-bit `u32` tile.
#[inline(always)]
pub fn unpack_u32(width: usize, packed: &[u32], out: &mut [u32; TILE]) {
    debug_assert_eq!(packed.len(), packed_len_u32(width));
    // SAFETY: lengths checked above; `out` is exactly one tile.
    unsafe { BitPacking::unchecked_unpack(width, packed, out.as_mut_slice()) }
}

/// Stencil: fused unpack + wrapping-add of a FoR `reference` for a `u64` tile.
#[inline(always)]
pub fn unfor_unpack_u64(width: usize, packed: &[u64], reference: u64, out: &mut [u64; TILE]) {
    debug_assert_eq!(packed.len(), packed_len_u64(width));
    // SAFETY: lengths checked above; `out` is exactly one tile.
    unsafe { FoR::unchecked_unfor_pack(width, packed, reference, out.as_mut_slice()) }
}

/// Stencil: invert a transposed `u32` delta tile (zero base seed).
#[inline(always)]
pub fn undelta_u32(transposed_deltas: &[u32; TILE], out: &mut [u32; TILE]) {
    u32::undelta::<LANES32>(transposed_deltas, &ZERO_BASE32, out);
}

/// Stencil: invert a transposed `u64` delta tile (zero base seed).
#[inline(always)]
pub fn undelta_u64(transposed_deltas: &[u64; TILE], out: &mut [u64; TILE]) {
    u64::undelta::<LANES64>(transposed_deltas, &ZERO_BASE64, out);
}

/// Stencil: undo the FastLanes transpose for a `u32` tile.
#[inline(always)]
pub fn untranspose_u32(transposed: &[u32; TILE], out: &mut [u32; TILE]) {
    Transpose::untranspose(transposed, out);
}

/// Stencil: undo the FastLanes transpose for a `u64` tile.
#[inline(always)]
pub fn untranspose_u64(transposed: &[u64; TILE], out: &mut [u64; TILE]) {
    Transpose::untranspose(transposed, out);
}

/// Stencil: ALP scale. Reinterpret `u64` bits as `i64` digits and multiply by
/// `scale` (`10^-exponent`) to recover the `f64` values. Exception patching is
/// intentionally omitted in this prototype.
#[inline(always)]
pub fn alp_scale_tile(digits: &[u64; TILE], scale: f64, out: &mut [f64; TILE]) {
    for i in 0..TILE {
        out[i] = (digits[i] as i64) as f64 * scale;
    }
}

/// Monolithic tail stencil: untranspose **and** ALP-scale in a single
/// fully-unrolled pass. `transposed[i]` (a transposed `u64` digit) is reinterpreted
/// as `i64`, scaled, and scattered straight to its natural position
/// `transpose(i)`. This fuses two stages — and the intermediate `digits` tile —
/// into one, which is what lets the AOT kernel beat the staged `fused` path.
#[inline(always)]
pub fn untranspose_scale_tile(transposed: &[u64; TILE], scale: f64, out: &mut [f64; TILE]) {
    seq_macro::seq!(I in 0..1024 {
        out[fastlanes::transpose(I)] = (transposed[I] as i64) as f64 * scale;
    });
}

const FL_ORDER: [usize; 8] = [0, 4, 2, 6, 1, 5, 3, 7];

/// Precompute, in FastLanes undelta iteration order, the transposed source index
/// to read and the natural destination index to write. Lets the fused tail kernel
/// avoid recomputing the permutation per element.
const fn undelta_perm() -> ([u16; TILE], [u16; TILE]) {
    let mut src = [0u16; TILE];
    let mut dst = [0u16; TILE];
    let mut p = 0;
    let mut lane = 0;
    while lane < LANES64 {
        let mut row = 0;
        while row < 64 {
            // FastLanes `index(row, lane)` — the transposed index undelta visits.
            let idx = FL_ORDER[row / 8] * 16 + (row % 8) * 128 + lane;
            src[p] = idx as u16;
            // `transpose(idx)` — its natural (untransposed) position.
            dst[p] = ((idx % 16) * 64 + FL_ORDER[(idx / 16) % 8] * 8 + idx / 128) as u16;
            p += 1;
            row += 1;
        }
        lane += 1;
    }
    (src, dst)
}

const UNDELTA_SRC: [u16; TILE] = undelta_perm().0;
const UNDELTA_DST: [u16; TILE] = undelta_perm().1;

/// Monolithic tail stencil for `u64`: fuse undelta + untranspose + ALP-scale into
/// a single pass. The transposed delta tile is prefix-summed per lane and each
/// running value is scaled and scattered straight to its natural position — so
/// neither the undelta'd tile nor the digits tile is ever materialised.
#[inline(always)]
pub fn undelta_untranspose_scale_tile(
    transposed_deltas: &[u64; TILE],
    scale: f64,
    out: &mut [f64; TILE],
) {
    let mut p = 0;
    for _lane in 0..LANES64 {
        let mut prev = 0u64;
        for _ in 0..64 {
            prev = prev.wrapping_add(transposed_deltas[UNDELTA_SRC[p] as usize]);
            out[UNDELTA_DST[p] as usize] = (prev as i64) as f64 * scale;
            p += 1;
        }
    }
}

/// Slice form of [`alp_scale_tile`] used by the materialized strategy.
#[inline(always)]
pub fn alp_scale_slice(digits: &[u64], scale: f64, out: &mut [f64]) {
    debug_assert_eq!(digits.len(), out.len());
    for (d, o) in digits.iter().zip(out.iter_mut()) {
        *o = (*d as i64) as f64 * scale;
    }
}

/// Stencil: expand run-length-encoded `values` over their exclusive `run_ends`.
///
/// `run_ends` is strictly increasing with `run_ends.last() == out.len()`.
#[inline(always)]
pub fn rle_expand(run_ends: &[u32], values: &[f64], out: &mut [f64]) {
    debug_assert_eq!(run_ends.len(), values.len());
    let mut start = 0usize;
    for (&end, &v) in run_ends.iter().zip(values.iter()) {
        let end = end as usize;
        out[start..end].fill(v);
        start = end;
    }
}
