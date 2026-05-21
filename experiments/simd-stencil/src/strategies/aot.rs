// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Ahead-of-time decode: the upper bound.
//!
//! Each tile is decoded by a fully-inlined, const-generic pipeline that is
//! monomorphised for the exact bit-width. Dispatch is a `match` over every
//! width, i.e. every `(stack, width)` combination is compiled ahead of time.
//! This is the quality you cannot get without either combinatorial AOT
//! compilation or a real run-time code generator.

use fastlanes::Delta;
use fastlanes::FoR;

use crate::TILE;
use crate::encode::EncodedA;
use crate::encode::EncodedB;
use crate::encode::EncodedC;
use crate::kernels::LANES32;
use crate::kernels::alp_scale_tile;
use crate::kernels::rle_expand;
use crate::kernels::undelta_u64;
use crate::kernels::untranspose_u32;
use crate::kernels::untranspose_u64;
use crate::strategies::tile_f64_mut;
use crate::strategies::tile_u32_mut;

const ZERO32: [u32; LANES32] = [0; LANES32];

pub fn decode_a(enc: &EncodedA) -> Vec<u32> {
    let n = enc.n;
    let tiles = n / TILE;
    let mut out = vec![0u32; n];
    let mut tv = [0u32; TILE];

    for t in 0..tiles {
        let w = enc.width[t] as usize;
        let off = enc.offsets[t];
        let out_tile = tile_u32_mut(&mut out[t * TILE..(t + 1) * TILE]);
        seq_macro::seq!(W in 1..=32 {
            match w {
                #(
                    W => {
                        let inp: &[u32; 32 * W] =
                            enc.packed[off..off + 32 * W].try_into().unwrap();
                        // Fused unpack + undelta, monomorphised for width W.
                        <u32 as Delta>::undelta_pack::<LANES32, W, { 32 * W }>(inp, &ZERO32, &mut tv);
                        untranspose_u32(&tv, out_tile);
                    }
                )*
                _ => unreachable!("u32 width out of range: {w}"),
            }
        });
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
        let reference = enc.reference[t];
        seq_macro::seq!(W in 1..=64 {
            match w {
                #(
                    W => {
                        let inp: &[u64; 16 * W] =
                            enc.packed[off..off + 16 * W].try_into().unwrap();
                        // Fused unpack + add reference, monomorphised for width W.
                        <u64 as FoR>::unfor_pack::<W, { 16 * W }>(inp, reference, &mut td);
                    }
                )*
                _ => unreachable!("u64 width out of range: {w}"),
            }
        });
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

pub fn decode_c(enc: &EncodedC) -> Vec<f64> {
    let ends = decode_a(&enc.ends);
    let vals = decode_b(&enc.vals);
    let mut out = vec![0f64; enc.n_logical];
    rle_expand(&ends, &vals, &mut out);
    out
}
