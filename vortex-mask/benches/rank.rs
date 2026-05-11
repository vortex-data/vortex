// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for `Mask::rank`.

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use divan::Bencher;
use vortex_buffer::BitBuffer;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

const BENCH_SIZES: &[usize] = &[1_024, 16_384, 65_536, 262_144];

fn create_mask(len: usize, density: f64) -> Mask {
    Mask::from_buffer(BitBuffer::from_iter((0..len).map(|i| {
        let threshold = (density * 1000.0) as usize;
        (i * 7 + 13) % 1000 < threshold
    })))
}

/// Single rank lookup at the midpoint.
#[divan::bench(args = BENCH_SIZES)]
fn rank_single_mid(bencher: Bencher, len: usize) {
    let mask = create_mask(len, 0.5);
    let mid = mask.true_count() / 2;
    bencher
        .with_inputs(|| (&mask, mid))
        .bench_refs(|(mask, mid)| mask.rank(*mid));
}

/// Rank every set bit sequentially (worst-case total scan).
#[divan::bench(args = BENCH_SIZES)]
fn rank_all_sequential(bencher: Bencher, len: usize) {
    let mask = create_mask(len, 0.5);
    let tc = mask.true_count();
    bencher.with_inputs(|| &mask).bench_refs(|mask| {
        for nth in 0..tc {
            divan::black_box(mask.rank(nth));
        }
    });
}

/// Single rank on a sparse mask (1% density).
#[divan::bench(args = BENCH_SIZES)]
fn rank_single_sparse(bencher: Bencher, len: usize) {
    let mask = create_mask(len, 0.01);
    let mid = mask.true_count() / 2;
    bencher
        .with_inputs(|| (&mask, mid))
        .bench_refs(|(mask, mid)| mask.rank(*mid));
}

/// Single rank on a dense mask (90% density).
#[divan::bench(args = BENCH_SIZES)]
fn rank_single_dense(bencher: Bencher, len: usize) {
    let mask = create_mask(len, 0.9);
    let mid = mask.true_count() / 2;
    bencher
        .with_inputs(|| (&mask, mid))
        .bench_refs(|(mask, mid)| mask.rank(*mid));
}
