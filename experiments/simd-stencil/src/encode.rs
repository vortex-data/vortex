// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Synthetic data generation and encoding.
//!
//! Encoding is done with the same `fastlanes` primitives the decoders invert,
//! so every stack round-trips by construction. Data is generated smooth enough
//! that the per-tile bit-widths are realistic (i.e. the stacks actually
//! compress) rather than degenerate.

use fastlanes::BitPacking;
use fastlanes::Delta;
use fastlanes::Transpose;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::TILE;
use crate::kernels::LANES32;
use crate::kernels::LANES64;
use crate::kernels::packed_len_u32;
use crate::kernels::packed_len_u64;

/// `delta(bitpacking)` encoded `u32` column.
pub struct EncodedA {
    pub n: usize,
    pub width: Vec<u8>,
    pub offsets: Vec<usize>,
    pub packed: Vec<u32>,
    /// Canonical values, retained for round-trip checks and Vortex baselines.
    pub values: Vec<u32>,
}

/// `alp(delta(ffor(bitpacking)))` encoded `f64` column.
pub struct EncodedB {
    pub n: usize,
    pub width: Vec<u8>,
    pub offsets: Vec<usize>,
    pub packed: Vec<u64>,
    pub reference: Vec<u64>,
    pub scale: f64,
    pub exponent: i32,
    pub digits: Vec<i64>,
    pub values: Vec<f64>,
}

/// `rle(alp(delta(ffor(bitpacking))))` column: f64 run values + delta-bitpacked run ends.
pub struct EncodedC {
    pub n_logical: usize,
    pub ends: EncodedA,
    pub vals: EncodedB,
    pub values: Vec<f64>,
}

#[inline]
fn bit_width_u32(max: u32) -> usize {
    (32 - max.leading_zeros() as usize).max(1)
}

#[inline]
fn bit_width_u64(max: u64) -> usize {
    (64 - max.leading_zeros() as usize).max(1)
}

/// Generate a smooth, monotone `u32` column of `n` (tile-aligned) elements.
pub fn gen_u32(n: usize, seed: u64) -> Vec<u32> {
    assert_eq!(n % TILE, 0);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut acc = 0u32;
    (0..n)
        .map(|_| {
            acc = acc.wrapping_add(rng.random_range(0..8));
            acc
        })
        .collect()
}

/// Generate a smooth `f64` column with `exponent` decimal digits of precision.
pub fn gen_f64(n: usize, exponent: i32, seed: u64) -> Vec<f64> {
    assert_eq!(n % TILE, 0);
    let mut rng = StdRng::seed_from_u64(seed);
    let inv = 10f64.powi(exponent);
    let scale = 10f64.powi(-exponent);
    let mut acc = 1000.0f64;
    (0..n)
        .map(|_| {
            acc += rng.random_range(-2.0..2.0);
            // Snap to integer digits, then recover via the exact `*scale` op the
            // decoder uses, so the ALP round-trip is bit-identical.
            (acc * inv).round() * scale
        })
        .collect()
}

/// Encode a `u32` column as `delta(bitpacking)`.
pub fn encode_a(values: &[u32]) -> EncodedA {
    let n = values.len();
    assert_eq!(n % TILE, 0);
    let tiles = n / TILE;

    let mut width = Vec::with_capacity(tiles);
    let mut offsets = Vec::with_capacity(tiles);
    let mut packed: Vec<u32> = Vec::new();

    let mut tv = [0u32; TILE];
    let mut td = [0u32; TILE];
    for t in 0..tiles {
        let tile: &[u32; TILE] = values[t * TILE..(t + 1) * TILE].try_into().unwrap();
        <u32 as Transpose>::transpose(tile, &mut tv);
        <u32 as Delta>::delta::<LANES32>(&tv, &[0u32; LANES32], &mut td);

        let w = bit_width_u32(td.iter().copied().max().unwrap_or(0));
        let plen = packed_len_u32(w);
        offsets.push(packed.len());
        width.push(w as u8);
        let start = packed.len();
        packed.resize(start + plen, 0);
        // SAFETY: `td` is one tile, `packed[start..]` has exactly `plen` words.
        unsafe { BitPacking::unchecked_pack(w, &td, &mut packed[start..start + plen]) };
    }

    EncodedA {
        n,
        width,
        offsets,
        packed,
        values: values.to_vec(),
    }
}

/// Encode an `f64` column as `alp(delta(ffor(bitpacking)))`.
pub fn encode_b(values: &[f64], exponent: i32) -> EncodedB {
    let n = values.len();
    assert_eq!(n % TILE, 0);
    let tiles = n / TILE;

    let inv = 10f64.powi(exponent);
    let scale = 10f64.powi(-exponent);
    let digits: Vec<i64> = values.iter().map(|&v| (v * inv).round() as i64).collect();
    // Reinterpret as u64 for the unsigned FastLanes pipeline (wrapping arithmetic).
    let as_u64: Vec<u64> = digits.iter().map(|&d| d as u64).collect();

    let mut width = Vec::with_capacity(tiles);
    let mut offsets = Vec::with_capacity(tiles);
    let mut reference = Vec::with_capacity(tiles);
    let mut packed: Vec<u64> = Vec::new();

    let mut tv = [0u64; TILE];
    let mut td = [0u64; TILE];
    for t in 0..tiles {
        let tile: &[u64; TILE] = as_u64[t * TILE..(t + 1) * TILE].try_into().unwrap();
        <u64 as Transpose>::transpose(tile, &mut tv);
        <u64 as Delta>::delta::<LANES64>(&tv, &[0u64; LANES64], &mut td);

        // FoR: subtract the per-tile minimum so the residuals bit-pack tightly.
        let r = td.iter().copied().min().unwrap_or(0);
        let mut resid = [0u64; TILE];
        let mut max = 0u64;
        for i in 0..TILE {
            let v = td[i].wrapping_sub(r);
            resid[i] = v;
            max = max.max(v);
        }
        let w = bit_width_u64(max);
        let plen = packed_len_u64(w);
        offsets.push(packed.len());
        width.push(w as u8);
        reference.push(r);
        let start = packed.len();
        packed.resize(start + plen, 0);
        // SAFETY: `resid` is one tile, `packed[start..]` has exactly `plen` words.
        unsafe { BitPacking::unchecked_pack(w, &resid, &mut packed[start..start + plen]) };
    }

    EncodedB {
        n,
        width,
        offsets,
        packed,
        reference,
        scale,
        exponent,
        digits,
        values: values.to_vec(),
    }
}

/// Encode an RLE column: random run lengths whose values come from a stack-B
/// column and whose exclusive run ends come from a stack-A column.
pub fn encode_c(n_runs: usize, exponent: i32, seed: u64) -> EncodedC {
    let runs_padded = n_runs.next_multiple_of(TILE);
    let mut rng = StdRng::seed_from_u64(seed);

    // Run ends: cumulative run lengths (1..=16 each), then padded.
    let mut ends_u32 = Vec::with_capacity(runs_padded);
    let mut acc = 0u32;
    for _ in 0..n_runs {
        acc += rng.random_range(1..=16);
        ends_u32.push(acc);
    }
    let n_logical = acc as usize;
    // Pad with the final end so expand over padding is a no-op.
    ends_u32.resize(runs_padded, acc);

    let vals_f64 = gen_f64(runs_padded, exponent, seed ^ 0x9e37);

    let ends = encode_a(&ends_u32);
    let vals = encode_b(&vals_f64, exponent);

    // Materialise the logical column for round-trip checks.
    let mut values = vec![0f64; n_logical];
    let mut start = 0usize;
    for r in 0..n_runs {
        let end = ends_u32[r] as usize;
        values[start..end].fill(vals_f64[r]);
        start = end;
    }

    EncodedC {
        n_logical,
        ends,
        vals,
        values,
    }
}
