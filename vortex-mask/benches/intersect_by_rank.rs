// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use divan::Bencher;
use vortex_buffer::BitBuffer;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

/// Generate a mask with approximately `density` fraction of true values.
fn make_mask_with_density(len: usize, density: f64) -> Mask {
    Mask::from_buffer(BitBuffer::from_iter(
        (0..len).map(|i| (i as f64 / len as f64) < density),
    ))
}

/// Generate a mask from specific indices.
fn make_sparse_mask(len: usize, indices: Vec<usize>) -> Mask {
    Mask::from_indices(len, indices)
}

// =============================================================================
// Benchmark: Varying mask sizes with fixed densities
// =============================================================================

const INPUT_SIZES: &[usize] = &[1_000, 10_000, 100_000, 1_000_000];

/// Base mask is sparse (1% density), rank mask selects 50% of those.
/// This is the "happy path" the function is optimized for.
#[divan::bench(args = INPUT_SIZES)]
fn sparse_base_half_rank(bencher: Bencher, len: usize) {
    let base = make_mask_with_density(len, 0.01); // ~1% true
    let rank_len = base.true_count();
    let rank = make_mask_with_density(rank_len, 0.5); // select 50%

    bencher
        .with_inputs(|| (&base, &rank))
        .bench_refs(|(base, rank)| base.intersect_by_rank(rank));
}

/// Base mask is dense (50% density), rank mask selects 50%.
/// Tests performance with larger indices arrays.
#[divan::bench(args = INPUT_SIZES)]
fn dense_base_half_rank(bencher: Bencher, len: usize) {
    let base = make_mask_with_density(len, 0.5); // 50% true
    let rank_len = base.true_count();
    let rank = make_mask_with_density(rank_len, 0.5); // select 50%

    bencher
        .with_inputs(|| (&base, &rank))
        .bench_refs(|(base, rank)| base.intersect_by_rank(rank));
}

/// Base mask is very sparse (0.1%), rank mask selects all.
/// Tests the AllTrue fast path for rank mask.
#[divan::bench(args = INPUT_SIZES)]
fn very_sparse_base_all_rank(bencher: Bencher, len: usize) {
    let base = make_mask_with_density(len, 0.001); // 0.1% true
    let rank_len = base.true_count();
    let rank = Mask::new_true(rank_len); // select all

    bencher
        .with_inputs(|| (&base, &rank))
        .bench_refs(|(base, rank)| base.intersect_by_rank(rank));
}

/// Base mask is very sparse (0.1%), rank mask selects none.
/// Tests the AllFalse fast path for rank mask.
#[divan::bench(args = INPUT_SIZES)]
fn very_sparse_base_none_rank(bencher: Bencher, len: usize) {
    let base = make_mask_with_density(len, 0.001);
    let rank_len = base.true_count();
    let rank = Mask::new_false(rank_len);

    bencher
        .with_inputs(|| (&base, &rank))
        .bench_refs(|(base, rank)| base.intersect_by_rank(rank));
}

// =============================================================================
// Benchmark: Effect of indices caching
// =============================================================================

/// Measures the cost when indices are NOT pre-cached (first call).
/// The base mask is created from a BitBuffer, so indices() must be computed.
#[divan::bench(args = INPUT_SIZES)]
fn uncached_indices(bencher: Bencher, len: usize) {
    // Create masks fresh each iteration to avoid caching
    bencher
        .with_inputs(|| {
            let base = make_mask_with_density(len, 0.1);
            let rank_len = base.true_count();
            let rank = make_mask_with_density(rank_len, 0.5);
            (base, rank)
        })
        .bench_refs(|(base, rank)| base.intersect_by_rank(rank));
}

/// Measures the cost when indices ARE pre-cached.
/// Call indices() before the benchmark to populate the cache.
#[divan::bench(args = INPUT_SIZES)]
fn cached_indices(bencher: Bencher, len: usize) {
    let base = make_mask_with_density(len, 0.1);
    let rank_len = base.true_count();
    let rank = make_mask_with_density(rank_len, 0.5);

    // Pre-cache the indices
    let _ = base.indices();
    let _ = rank.indices();

    bencher
        .with_inputs(|| (&base, &rank))
        .bench_refs(|(base, rank)| base.intersect_by_rank(rank));
}

// =============================================================================
// Benchmark: Different rank selectivity patterns
// =============================================================================

const SELECTIVITIES: &[usize] = &[1, 10, 50, 90, 99]; // percentage

/// Fixed base size, varying rank selectivity.
#[divan::bench(args = SELECTIVITIES)]
fn rank_selectivity(bencher: Bencher, selectivity_pct: usize) {
    let len = 100_000;
    let base = make_mask_with_density(len, 0.1); // 10% true = ~10k indices
    let rank_len = base.true_count();
    let rank = make_mask_with_density(rank_len, selectivity_pct as f64 / 100.0);

    // Pre-cache
    let _ = base.indices();
    let _ = rank.indices();

    bencher
        .with_inputs(|| (&base, &rank))
        .bench_refs(|(base, rank)| base.intersect_by_rank(rank));
}

// =============================================================================
// Benchmark: Access patterns (sequential vs scattered)
// =============================================================================

/// Rank mask selects sequential indices (good cache locality).
#[divan::bench(args = INPUT_SIZES)]
fn sequential_rank_pattern(bencher: Bencher, len: usize) {
    let base = make_mask_with_density(len, 0.1);
    let rank_len = base.true_count();
    // Select first half sequentially
    let rank_indices: Vec<usize> = (0..rank_len / 2).collect();
    let rank = make_sparse_mask(rank_len, rank_indices);

    let _ = base.indices();
    let _ = rank.indices();

    bencher
        .with_inputs(|| (&base, &rank))
        .bench_refs(|(base, rank)| base.intersect_by_rank(rank));
}

/// Rank mask selects strided indices (poor cache locality).
#[divan::bench(args = INPUT_SIZES)]
fn strided_rank_pattern(bencher: Bencher, len: usize) {
    let base = make_mask_with_density(len, 0.1);
    let rank_len = base.true_count();
    // Select every other index
    let rank_indices: Vec<usize> = (0..rank_len).step_by(2).collect();
    let rank = make_sparse_mask(rank_len, rank_indices);

    let _ = base.indices();
    let _ = rank.indices();

    bencher
        .with_inputs(|| (&base, &rank))
        .bench_refs(|(base, rank)| base.intersect_by_rank(rank));
}

// =============================================================================
// Benchmark: Compare with AllTrue/AllFalse fast paths
// =============================================================================

/// AllTrue base mask - should return rank mask directly.
#[divan::bench(args = INPUT_SIZES)]
fn all_true_base(bencher: Bencher, len: usize) {
    let base = Mask::new_true(len);
    let rank = make_mask_with_density(len, 0.5);

    bencher
        .with_inputs(|| (&base, &rank))
        .bench_refs(|(base, rank)| base.intersect_by_rank(rank));
}

/// AllFalse base mask - should return all false immediately.
#[divan::bench(args = &[1_000, 10_000])] // smaller sizes since rank must be 0
fn all_false_base(bencher: Bencher, len: usize) {
    let base = Mask::new_false(len);
    let rank = Mask::new_false(0); // must have length 0

    bencher
        .with_inputs(|| (&base, &rank))
        .bench_refs(|(base, rank)| base.intersect_by_rank(rank));
}
