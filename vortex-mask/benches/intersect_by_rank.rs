// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for the `intersect_by_rank` implementations.
//!
//! Tests each implementation with:
//! - Two data patterns: random and runs (correlated)
//! - Two sizes: 10k and 100k

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation, clippy::panic)]

use divan::Bencher;
use vortex_buffer::BitBuffer;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

/// Benchmark args: (size, pattern)
/// Pattern: "random" or "runs"
const BENCH_ARGS: &[(usize, &str)] = &[
    (10_000, "random"),
    (10_000, "runs"),
    (100_000, "random"),
    (100_000, "runs"),
];

/// Create a mask with random-ish bits (deterministic pseudo-random)
fn create_random_mask(len: usize, selectivity: f64) -> Mask {
    Mask::from_buffer(BitBuffer::from_iter((0..len).map(|i| {
        let threshold = (selectivity * 1000.0) as usize;
        (i * 7 + 13) % 1000 < threshold
    })))
}

/// Create a mask with runs of consecutive trues/falses
fn create_runs_mask(len: usize, run_len: usize, gap_len: usize) -> Mask {
    Mask::from_buffer(BitBuffer::from_iter((0..len).map(|i| {
        let cycle = run_len + gap_len;
        (i % cycle) < run_len
    })))
}

/// Create test fixtures based on pattern type
fn create_fixture(size: usize, pattern: &str) -> (Mask, Mask) {
    match pattern {
        "random" => {
            // Random base (50% selectivity), random rank (50% selectivity)
            let base = create_random_mask(size, 0.5);
            let rank_len = base.true_count();
            let rank = create_random_mask(rank_len, 0.5);
            (base, rank)
        }
        "runs" => {
            // Runs of 64 trues, gaps of 64 falses (50% selectivity with correlation)
            let base = create_runs_mask(size, 64, 64);
            let rank_len = base.true_count();
            let rank = create_runs_mask(rank_len, 64, 64);
            (base, rank)
        }
        _ => panic!("Unknown pattern: {pattern}"),
    }
}

// =============================================================================
// Main implementation (hybrid approach)
// =============================================================================

#[divan::bench(args = BENCH_ARGS)]
fn main_impl(bencher: Bencher, (size, pattern): (usize, &str)) {
    let (base, rank) = create_fixture(size, pattern);
    bencher.bench(|| base.intersect_by_rank(&rank));
}

// =============================================================================
// BitBuffer implementation (bit-by-bit iteration)
// =============================================================================

#[divan::bench(args = BENCH_ARGS)]
fn bitbuffer_impl(bencher: Bencher, (size, pattern): (usize, &str)) {
    let (base, rank) = create_fixture(size, pattern);
    bencher.bench(|| base.intersect_by_rank_bitbuffer(&rank));
}

// =============================================================================
// u64 implementation (chunk processing)
// =============================================================================

#[divan::bench(args = BENCH_ARGS)]
fn u64_impl(bencher: Bencher, (size, pattern): (usize, &str)) {
    let (base, rank) = create_fixture(size, pattern);
    bencher.bench(|| base.intersect_by_rank_u64(&rank));
}

// =============================================================================
// Hybrid implementation (indices lookup + u64 output)
// =============================================================================

#[divan::bench(args = BENCH_ARGS)]
fn hybrid_impl(bencher: Bencher, (size, pattern): (usize, &str)) {
    let (base, rank) = create_fixture(size, pattern);
    bencher.bench(|| base.intersect_by_rank_hybrid(&rank));
}

// =============================================================================
// Runs implementation (PDEP-style for correlated data)
// =============================================================================

#[divan::bench(args = BENCH_ARGS)]
fn runs_impl(bencher: Bencher, (size, pattern): (usize, &str)) {
    let (base, rank) = create_fixture(size, pattern);
    bencher.bench(|| base.intersect_by_rank_runs(&rank));
}
