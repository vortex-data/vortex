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
