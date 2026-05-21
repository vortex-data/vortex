// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fused decode: a tiled pipeline that runs every layer through L1-resident
//! scratch tiles, so no full-column intermediate is ever written to memory.
//!
//! This is "copy-and-patch" with stencils kept as ordinary (pre-compiled)
//! functions: per tile we select the right kernels and pass the runtime
//! constants (bit-width, FoR reference, ALP scale) as arguments.

use crate::TILE;
use crate::encode::EncodedA;
use crate::encode::EncodedB;
use crate::encode::EncodedC;
use crate::kernels::alp_scale_tile;
use crate::kernels::rle_expand;
use crate::kernels::undelta_u32;
use crate::kernels::undelta_u64;
use crate::kernels::unfor_unpack_u64;
use crate::kernels::unpack_u32;
use crate::kernels::untranspose_u32;
use crate::kernels::untranspose_u64;
use crate::strategies::tile_f64_mut;
use crate::strategies::tile_u32_mut;

pub fn decode_a(enc: &EncodedA) -> Vec<u32> {
    let n = enc.n;
    let tiles = n / TILE;
    let mut out = vec![0u32; n];

    let mut td = [0u32; TILE];
    let mut tv = [0u32; TILE];
    for t in 0..tiles {
        let w = enc.width[t] as usize;
        let off = enc.offsets[t];
        let plen = TILE * w / 32;
        unpack_u32(w, &enc.packed[off..off + plen], &mut td);
        undelta_u32(&td, &mut tv);
        untranspose_u32(&tv, tile_u32_mut(&mut out[t * TILE..(t + 1) * TILE]));
    }
    out
}

pub fn decode_b(enc: &EncodedB) -> Vec<f64> {
    let n = enc.n;
    let tiles = n / TILE;
    let mut out = vec![0f64; n];

    let mut td = [0u64; TILE];
    let mut tu = [0u64; TILE];
    let mut digits = [0u64; TILE];
    for t in 0..tiles {
        let w = enc.width[t] as usize;
        let off = enc.offsets[t];
        let plen = TILE * w / 64;
        unfor_unpack_u64(w, &enc.packed[off..off + plen], enc.reference[t], &mut td);
        undelta_u64(&td, &mut tu);
        untranspose_u64(&tu, &mut digits);
        alp_scale_tile(
            &digits,
            enc.scale,
            tile_f64_mut(&mut out[t * TILE..(t + 1) * TILE]),
        );
    }
    out
}

/// Integer core of stack B: decode to `i64` digits (no ALP scale), for an
/// apples-to-apples comparison against Vortex's `delta(ffor(bitpacking))`.
pub fn decode_b_core(enc: &EncodedB) -> Vec<i64> {
    let n = enc.n;
    let tiles = n / TILE;
    let mut out = vec![0i64; n];

    let mut td = [0u64; TILE];
    let mut tu = [0u64; TILE];
    let mut digits = [0u64; TILE];
    for t in 0..tiles {
        let w = enc.width[t] as usize;
        let off = enc.offsets[t];
        let plen = TILE * w / 64;
        unfor_unpack_u64(w, &enc.packed[off..off + plen], enc.reference[t], &mut td);
        undelta_u64(&td, &mut tu);
        untranspose_u64(&tu, &mut digits);
        for i in 0..TILE {
            out[t * TILE + i] = digits[i] as i64;
        }
    }
    out
}

pub fn decode_c(enc: &EncodedC) -> Vec<f64> {
    let ends = decode_a(&enc.ends);
    let vals = decode_b(&enc.vals);
    let mut out = vec![0f64; enc.n_logical];
    rle_expand(&ends, &vals, &mut out);
    out
}
