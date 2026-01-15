// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use divan::Bencher;
use vortex_buffer::BitBuffer;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

// =============================================================================
// Benchmark: Mask::rank (select) operation - uncached vs cached
// =============================================================================

const RANK_SIZES: &[usize] = &[1_000, 10_000, 100_000, 1_000_000];

/// Benchmark rank() when indices are NOT cached (uses select algorithm).
/// Creates a fresh mask each iteration to ensure no caching.
#[divan::bench(args = RANK_SIZES)]
fn rank_uncached(bencher: Bencher, len: usize) {
    bencher
        .with_inputs(|| {
            // Create fresh mask from BitBuffer - indices won't be cached
            let mask = Mask::from_buffer(BitBuffer::from_iter((0..len).map(|i| i % 10 == 0)));
            let true_count = mask.true_count();
            // Pick a rank near the middle
            let target_rank = true_count / 2;
            (mask, target_rank)
        })
        .bench_refs(|(mask, target_rank)| mask.rank(*target_rank));
}

/// Benchmark rank() when indices ARE cached.
/// Pre-calls indices() to populate the cache.
#[divan::bench(args = RANK_SIZES)]
fn rank_cached(bencher: Bencher, len: usize) {
    let mask = Mask::from_buffer(BitBuffer::from_iter((0..len).map(|i| i % 10 == 0)));
    // Pre-cache the indices
    let _ = mask.indices();
    let true_count = mask.true_count();
    let target_rank = true_count / 2;

    bencher
        .with_inputs(|| (&mask, target_rank))
        .bench_refs(|(mask, target_rank)| mask.rank(*target_rank));
}

/// Benchmark multiple sequential rank() calls on uncached mask.
/// This shows the benefit of avoiding full indices materialization.
#[divan::bench(args = RANK_SIZES)]
fn rank_uncached_multiple(bencher: Bencher, len: usize) {
    bencher
        .with_inputs(|| {
            let mask = Mask::from_buffer(BitBuffer::from_iter((0..len).map(|i| i % 10 == 0)));
            let true_count = mask.true_count();
            // Query 10 different ranks
            let ranks: Vec<usize> = (0..10).map(|i| i * true_count / 10).collect();
            (mask, ranks)
        })
        .bench_refs(|(mask, ranks)| {
            for &r in ranks.iter() {
                divan::black_box(mask.rank(r));
            }
        });
}

/// Benchmark BitBuffer::select directly (the underlying operation).
#[divan::bench(args = RANK_SIZES)]
fn bitbuffer_select(bencher: Bencher, len: usize) {
    let buf = BitBuffer::from_iter((0..len).map(|i| i % 10 == 0));
    let true_count = buf.true_count();
    let target = true_count / 2;

    bencher
        .with_inputs(|| (&buf, target))
        .bench_refs(|(buf, target)| buf.select(*target));
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
