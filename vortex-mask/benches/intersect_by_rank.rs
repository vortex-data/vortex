// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for `intersect_by_rank`.
//!
//! Compares simple (indices-based) vs optimized (PDEP + fast paths).
//! For best performance, compile with BMI2: RUSTFLAGS="-C target-feature=+bmi2"

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use divan::Bencher;
use vortex_buffer::BitBuffer;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

const BENCH_ARGS: &[(usize, &str)] = &[
    (10_000, "random"),
    (10_000, "runs"),
    (100_000, "random"),
    (100_000, "runs"),
];

fn create_random_mask(len: usize, selectivity: f64) -> Mask {
    Mask::from_buffer(BitBuffer::from_iter((0..len).map(|i| {
        let threshold = (selectivity * 1000.0) as usize;
        (i * 7 + 13) % 1000 < threshold
    })))
}

fn create_runs_mask(len: usize, run_len: usize, gap_len: usize) -> Mask {
    Mask::from_buffer(BitBuffer::from_iter((0..len).map(|i| {
        let cycle = run_len + gap_len;
        (i % cycle) < run_len
    })))
}

fn create_fixture(size: usize, pattern: &str) -> (Mask, Mask) {
    match pattern {
        "random" => {
            let base = create_random_mask(size, 0.5);
            let rank_len = base.true_count();
            let rank = create_random_mask(rank_len, 0.5);
            (base, rank)
        }
        "runs" => {
            let base = create_runs_mask(size, 64, 64);
            let rank_len = base.true_count();
            let rank = create_runs_mask(rank_len, 64, 64);
            (base, rank)
        }
        _ => unreachable!(),
    }
}

/// Simple indices-based (baseline)
#[divan::bench(args = BENCH_ARGS)]
fn simple(bencher: Bencher, (size, pattern): (usize, &str)) {
    let (base, rank) = create_fixture(size, pattern);
    bencher.bench(|| base.intersect_by_rank_simple(&rank));
}

/// Optimized PDEP implementation
#[divan::bench(args = BENCH_ARGS)]
fn optimized(bencher: Bencher, (size, pattern): (usize, &str)) {
    let (base, rank) = create_fixture(size, pattern);
    bencher.bench(|| base.intersect_by_rank(&rank));
}
