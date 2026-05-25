// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fused decode + filter.
//!
//! The per-stage breakdown shows ~half of a decode is *writing the f64 output
//! column*. A scan with a `WHERE` predicate doesn't need that column — it needs a
//! compact result: a count, an aggregate, or a selection mask. So we decode one
//! 1024-element tile into L1, apply the predicate in registers, fold it into the
//! compact result, and **never materialise the column**. The dominant
//! output-write cost disappears, along with the decode→DRAM→re-read round trip a
//! "decode then filter" pass pays.
//!
//! Compared here for `col > threshold` on stack B `alp(delta(ffor(bitpacking)))`:
//! - `fused_*` — decode tile in L1, fold predicate, emit count / mask.
//! - `materialized_*` — decode the whole column to a `Vec<f64>`, then scan it.

use crate::TILE;
use crate::encode::EncodedB;
use crate::kernels::alp_scale_tile;
use crate::kernels::undelta_u64;
use crate::kernels::unfor_unpack_u64;
use crate::kernels::untranspose_u64;
use crate::strategies::fused;

/// Decode one tile of stack B into a stack-resident `f64` tile (no heap output).
#[inline(always)]
fn decode_tile(enc: &EncodedB, t: usize, out: &mut [f64; TILE]) {
    let mut td = [0u64; TILE];
    let mut tu = [0u64; TILE];
    let mut digits = [0u64; TILE];
    let w = enc.width[t] as usize;
    let off = enc.offsets[t];
    let plen = TILE * w / 64;
    unfor_unpack_u64(w, &enc.packed[off..off + plen], enc.reference[t], &mut td);
    undelta_u64(&td, &mut tu);
    untranspose_u64(&tu, &mut digits);
    alp_scale_tile(&digits, enc.scale, out);
}

/// Fused `COUNT(*) WHERE col > threshold`. Output: one integer.
pub fn fused_count_gt(enc: &EncodedB, threshold: f64) -> usize {
    let tiles = enc.n / TILE;
    let mut tile = [0f64; TILE];
    let mut count = 0usize;
    for t in 0..tiles {
        decode_tile(enc, t, &mut tile);
        count += tile.iter().filter(|&&x| x > threshold).count();
    }
    count
}

/// Fused `SUM(col) WHERE col > threshold`. Output: one scalar.
pub fn fused_sum_gt(enc: &EncodedB, threshold: f64) -> f64 {
    let tiles = enc.n / TILE;
    let mut tile = [0f64; TILE];
    let mut sum = 0.0f64;
    for t in 0..tiles {
        decode_tile(enc, t, &mut tile);
        for &x in tile.iter() {
            if x > threshold {
                sum += x;
            }
        }
    }
    sum
}

/// Fused selection mask for `col > threshold`. Output: `n / 64` bitset words
/// (1 bit/row — 64× smaller than the decoded column).
pub fn fused_mask_gt(enc: &EncodedB, threshold: f64) -> Vec<u64> {
    let tiles = enc.n / TILE;
    let mut mask = vec![0u64; enc.n / 64];
    let mut tile = [0f64; TILE];
    for t in 0..tiles {
        decode_tile(enc, t, &mut tile);
        let base = t * (TILE / 64);
        for (w, chunk) in tile.chunks_exact(64).enumerate() {
            let mut word = 0u64;
            for (b, &x) in chunk.iter().enumerate() {
                word |= u64::from(x > threshold) << b;
            }
            mask[base + w] = word;
        }
    }
    mask
}

/// Baseline: decode the whole column to a `Vec<f64>`, then count.
pub fn materialized_count_gt(enc: &EncodedB, threshold: f64) -> usize {
    let v = fused::decode_b(enc);
    v.iter().filter(|&&x| x > threshold).count()
}

/// Baseline: decode the whole column to a `Vec<f64>`, then build the mask.
pub fn materialized_mask_gt(enc: &EncodedB, threshold: f64) -> Vec<u64> {
    let v = fused::decode_b(enc);
    let mut mask = vec![0u64; enc.n / 64];
    for (w, chunk) in v.chunks_exact(64).enumerate() {
        let mut word = 0u64;
        for (b, &x) in chunk.iter().enumerate() {
            word |= u64::from(x > threshold) << b;
        }
        mask[w] = word;
    }
    mask
}
