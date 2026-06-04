// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for `Mask::valid_counts_for_indices`.
//!
//! This mirrors the hot path in the pco/zstd slice decoders, which call
//! `valid_counts_for_indices(&[slice_start, slice_stop])` to translate a row
//! range into a value range. The cost is dominated by counting set bits in the
//! prefix `[0, slice_stop)`, so a SIMD popcount over the bit buffer should beat
//! a bit-by-bit walk handily for large `slice_stop`.

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use divan::Bencher;
use vortex_buffer::BitBuffer;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

const BENCH_SIZES: &[(usize, f64)] = &[
    (16_384, 0.1),
    (16_384, 0.9),
    (262_144, 0.1),
    (262_144, 0.9),
    (1_048_576, 0.1),
    (1_048_576, 0.9),
];

fn create_mask(len: usize, density: f64) -> Mask {
    Mask::from_buffer(BitBuffer::from_iter((0..len).map(|i| {
        let threshold = (density * 1000.0) as usize;
        (i * 7 + 13) % 1000 < threshold
    })))
}

/// The pco/zstd slice pattern: two indices bracketing a sub-range. The prefix
/// up to `slice_stop` must be counted, so `slice_stop` near the end is worst case.
#[divan::bench(args = BENCH_SIZES)]
fn slice_bounds(bencher: Bencher, (len, density): (usize, f64)) {
    let mask = create_mask(len, density);
    let indices = [len / 4, len - len / 8];
    bencher
        .with_inputs(|| (&mask, indices))
        .bench_refs(|(mask, indices)| mask.valid_counts_for_indices(indices));
}

/// Many monotonically increasing indices spread across the whole mask.
#[divan::bench(args = BENCH_SIZES)]
fn many_indices(bencher: Bencher, (len, density): (usize, f64)) {
    let mask = create_mask(len, density);
    let stride = (len / 256).max(1);
    let indices: Vec<usize> = (0..len).step_by(stride).collect();
    bencher
        .with_inputs(|| (&mask, indices.as_slice()))
        .bench_refs(|(mask, indices)| mask.valid_counts_for_indices(indices));
}
