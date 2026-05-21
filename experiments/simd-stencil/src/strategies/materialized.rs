// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Materialized decode: each encoding layer is fully written to a heap buffer
//! before the next layer reads it. This models Vortex's array-by-array
//! `execute` path, where every layer canonicalises into a `PrimitiveArray`.

use crate::TILE;
use crate::encode::EncodedA;
use crate::encode::EncodedB;
use crate::encode::EncodedC;
use crate::kernels::alp_scale_slice;
use crate::kernels::rle_expand;
use crate::kernels::undelta_u32;
use crate::kernels::undelta_u64;
use crate::kernels::unfor_unpack_u64;
use crate::kernels::unpack_u32;
use crate::kernels::untranspose_u32;
use crate::kernels::untranspose_u64;
use crate::strategies::tile_u32;
use crate::strategies::tile_u32_mut;
use crate::strategies::tile_u64;
use crate::strategies::tile_u64_mut;

pub fn decode_a(enc: &EncodedA) -> Vec<u32> {
    let n = enc.n;
    let tiles = n / TILE;

    // Layer 1 (bitpacking): unpack the whole column of transposed deltas.
    let mut deltas = vec![0u32; n];
    for t in 0..tiles {
        let w = enc.width[t] as usize;
        let off = enc.offsets[t];
        let plen = TILE * w / 32;
        unpack_u32(
            w,
            &enc.packed[off..off + plen],
            tile_u32_mut(&mut deltas[t * TILE..(t + 1) * TILE]),
        );
    }

    // Layer 2 (delta): undelta + untranspose the materialized delta column.
    let mut out = vec![0u32; n];
    let mut tv = [0u32; TILE];
    for t in 0..tiles {
        undelta_u32(tile_u32(&deltas[t * TILE..(t + 1) * TILE]), &mut tv);
        untranspose_u32(&tv, tile_u32_mut(&mut out[t * TILE..(t + 1) * TILE]));
    }
    out
}

pub fn decode_b(enc: &EncodedB) -> Vec<f64> {
    let n = enc.n;
    let tiles = n / TILE;

    // Layer 1 (ffor+bitpacking, fused): unpack + add reference -> delta column.
    let mut deltas = vec![0u64; n];
    for t in 0..tiles {
        let w = enc.width[t] as usize;
        let off = enc.offsets[t];
        let plen = TILE * w / 64;
        unfor_unpack_u64(
            w,
            &enc.packed[off..off + plen],
            enc.reference[t],
            tile_u64_mut(&mut deltas[t * TILE..(t + 1) * TILE]),
        );
    }

    // Layer 2 (delta): undelta + untranspose -> digit column.
    let mut digits = vec![0u64; n];
    let mut tu = [0u64; TILE];
    for t in 0..tiles {
        undelta_u64(tile_u64(&deltas[t * TILE..(t + 1) * TILE]), &mut tu);
        untranspose_u64(&tu, tile_u64_mut(&mut digits[t * TILE..(t + 1) * TILE]));
    }

    // Layer 3 (ALP): scale -> f64 column.
    let mut out = vec![0f64; n];
    alp_scale_slice(&digits, enc.scale, &mut out);
    out
}

pub fn decode_c(enc: &EncodedC) -> Vec<f64> {
    let ends = decode_a(&enc.ends);
    let vals = decode_b(&enc.vals);
    let mut out = vec![0f64; enc.n_logical];
    rle_expand(&ends, &vals, &mut out);
    out
}
