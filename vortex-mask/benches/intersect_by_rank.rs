// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for the `intersect_by_rank` method on `Mask`.
//!
//! This benchmarks the various fast and slow paths:
//! - Fast: AllTrue base, AllTrue mask, AllFalse (either)
//! - Slow: indices-based intersection

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use divan::Bencher;
use vortex_buffer::BitBuffer;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

// Base mask sizes to test
const BASE_SIZES: &[usize] = &[1_000, 10_000, 100_000, 1_000_000];

/// Create a mask with approximately `selectivity` fraction of true values
fn create_sparse_mask(len: usize, selectivity: f64) -> Mask {
    let step = (1.0 / selectivity).ceil() as usize;
    let indices: Vec<usize> = (0..len).step_by(step).collect();
    Mask::from_indices(len, indices)
}

/// Create a mask from a BitBuffer with given selectivity
fn create_dense_mask(len: usize, selectivity: f64) -> Mask {
    let buffer = BitBuffer::from_iter((0..len).map(|i| {
        // Use a deterministic pattern based on position
        let threshold = (selectivity * 1000.0) as usize;
        (i * 7 + 13) % 1000 < threshold
    }));
    Mask::from_buffer(buffer)
}

// =============================================================================
// Fast path benchmarks: AllTrue base mask
// =============================================================================

#[divan::bench(args = BASE_SIZES)]
fn fast_path_all_true_base_sparse_rank(bencher: Bencher, base_size: usize) {
    let base = Mask::new_true(base_size);
    // Rank mask length = base.true_count() = base_size
    let rank = create_sparse_mask(base_size, 0.1);

    bencher.bench(|| base.intersect_by_rank(&rank));
}

#[divan::bench(args = BASE_SIZES)]
fn fast_path_all_true_base_all_true_rank(bencher: Bencher, base_size: usize) {
    let base = Mask::new_true(base_size);
    let rank = Mask::new_true(base_size);

    bencher.bench(|| base.intersect_by_rank(&rank));
}

#[divan::bench(args = BASE_SIZES)]
fn fast_path_all_true_base_all_false_rank(bencher: Bencher, base_size: usize) {
    let base = Mask::new_true(base_size);
    let rank = Mask::new_false(base_size);

    bencher.bench(|| base.intersect_by_rank(&rank));
}

// =============================================================================
// Fast path benchmarks: AllTrue rank mask
// =============================================================================

#[divan::bench(args = BASE_SIZES)]
fn fast_path_sparse_base_all_true_rank(bencher: Bencher, base_size: usize) {
    let base = create_sparse_mask(base_size, 0.1);
    let rank_len = base.true_count();
    let rank = Mask::new_true(rank_len);

    bencher.bench(|| base.intersect_by_rank(&rank));
}

// =============================================================================
// Fast path benchmarks: AllFalse
// =============================================================================

#[divan::bench(args = BASE_SIZES)]
fn fast_path_all_false_base(bencher: Bencher, base_size: usize) {
    let base = Mask::new_false(base_size);
    let rank = Mask::new_true(0); // base.true_count() = 0

    bencher.bench(|| base.intersect_by_rank(&rank));
}

#[divan::bench(args = BASE_SIZES)]
fn fast_path_sparse_base_all_false_rank(bencher: Bencher, base_size: usize) {
    let base = create_sparse_mask(base_size, 0.1);
    let rank_len = base.true_count();
    let rank = Mask::new_false(rank_len);

    bencher.bench(|| base.intersect_by_rank(&rank));
}

// =============================================================================
// Slow path benchmarks: indices-based intersection
// =============================================================================

/// Benchmark with varying base selectivity (sparse base masks)
#[divan::bench(
    args = [
        (100_000, 0.01, 0.1),  // very sparse base, sparse rank
        (100_000, 0.01, 0.5),  // very sparse base, medium rank
        (100_000, 0.1, 0.1),   // sparse base, sparse rank
        (100_000, 0.1, 0.5),   // sparse base, medium rank
        (100_000, 0.5, 0.1),   // medium base, sparse rank
        (100_000, 0.5, 0.5),   // medium base, medium rank
    ]
)]
fn slow_path_indices_selectivity(bencher: Bencher, args: (usize, f64, f64)) {
    let (base_size, base_sel, rank_sel) = args;
    let base = create_sparse_mask(base_size, base_sel);
    let rank_len = base.true_count();
    let rank = create_sparse_mask(rank_len, rank_sel);

    bencher.bench(|| base.intersect_by_rank(&rank));
}

/// Benchmark scaling with base size
#[divan::bench(args = BASE_SIZES)]
fn slow_path_scaling_sparse_sparse(bencher: Bencher, base_size: usize) {
    // 10% base selectivity, 10% rank selectivity
    let base = create_sparse_mask(base_size, 0.1);
    let rank_len = base.true_count();
    let rank = create_sparse_mask(rank_len, 0.1);

    bencher.bench(|| base.intersect_by_rank(&rank));
}

#[divan::bench(args = BASE_SIZES)]
fn slow_path_scaling_sparse_dense(bencher: Bencher, base_size: usize) {
    // 10% base selectivity, 90% rank selectivity
    let base = create_sparse_mask(base_size, 0.1);
    let rank_len = base.true_count();
    let rank = create_dense_mask(rank_len, 0.9);

    bencher.bench(|| base.intersect_by_rank(&rank));
}

#[divan::bench(args = BASE_SIZES)]
fn slow_path_scaling_dense_sparse(bencher: Bencher, base_size: usize) {
    // 50% base selectivity, 10% rank selectivity
    let base = create_dense_mask(base_size, 0.5);
    let rank_len = base.true_count();
    let rank = create_sparse_mask(rank_len, 0.1);

    bencher.bench(|| base.intersect_by_rank(&rank));
}

#[divan::bench(args = BASE_SIZES)]
fn slow_path_scaling_dense_dense(bencher: Bencher, base_size: usize) {
    // 50% base selectivity, 50% rank selectivity
    let base = create_dense_mask(base_size, 0.5);
    let rank_len = base.true_count();
    let rank = create_dense_mask(rank_len, 0.5);

    bencher.bench(|| base.intersect_by_rank(&rank));
}

// =============================================================================
// Realistic use-case: low selectivity filter pushdown
// =============================================================================

/// This simulates the actual use case from flat/reader.rs where we have
/// a low-selectivity base mask (filter pushdown) intersected with a
/// predicate evaluation result.
#[divan::bench(args = BASE_SIZES)]
fn realistic_low_selectivity_filter(bencher: Bencher, base_size: usize) {
    // 1% selectivity base (like a highly selective WHERE clause)
    let base = create_sparse_mask(base_size, 0.01);
    let rank_len = base.true_count();
    // 50% of the filtered rows pass the secondary predicate
    let rank = create_dense_mask(rank_len, 0.5);

    bencher.bench(|| base.intersect_by_rank(&rank));
}

#[divan::bench(args = BASE_SIZES)]
fn realistic_medium_selectivity_filter(bencher: Bencher, base_size: usize) {
    // 10% selectivity base
    let base = create_sparse_mask(base_size, 0.1);
    let rank_len = base.true_count();
    // 30% of the filtered rows pass the secondary predicate
    let rank = create_dense_mask(rank_len, 0.3);

    bencher.bench(|| base.intersect_by_rank(&rank));
}

// =============================================================================
// High density rank mask (90%+) - potential optimization target
// =============================================================================

/// 90% rank selectivity - current implementation iterates 90% of indices
#[divan::bench(args = BASE_SIZES)]
fn high_density_rank_90_percent(bencher: Bencher, base_size: usize) {
    let base = create_sparse_mask(base_size, 0.1);
    let rank_len = base.true_count();
    let rank = create_dense_mask(rank_len, 0.9);

    bencher.bench(|| base.intersect_by_rank(&rank));
}

/// 95% rank selectivity - only 5% false values
#[divan::bench(args = BASE_SIZES)]
fn high_density_rank_95_percent(bencher: Bencher, base_size: usize) {
    let base = create_sparse_mask(base_size, 0.1);
    let rank_len = base.true_count();
    let rank = create_dense_mask(rank_len, 0.95);

    bencher.bench(|| base.intersect_by_rank(&rank));
}

/// 99% rank selectivity - only 1% false values
#[divan::bench(args = BASE_SIZES)]
fn high_density_rank_99_percent(bencher: Bencher, base_size: usize) {
    let base = create_sparse_mask(base_size, 0.1);
    let rank_len = base.true_count();
    let rank = create_dense_mask(rank_len, 0.99);

    bencher.bench(|| base.intersect_by_rank(&rank));
}

/// Compare: 10% rank (current sweet spot) vs high density
#[divan::bench(args = BASE_SIZES)]
fn comparison_rank_10_percent(bencher: Bencher, base_size: usize) {
    let base = create_sparse_mask(base_size, 0.1);
    let rank_len = base.true_count();
    let rank = create_dense_mask(rank_len, 0.1);

    bencher.bench(|| base.intersect_by_rank(&rank));
}

/// Dense base (50%) with very high density rank (95%)
#[divan::bench(args = BASE_SIZES)]
fn high_density_dense_base_95_rank(bencher: Bencher, base_size: usize) {
    let base = create_dense_mask(base_size, 0.5);
    let rank_len = base.true_count();
    let rank = create_dense_mask(rank_len, 0.95);

    bencher.bench(|| base.intersect_by_rank(&rank));
}

// =============================================================================
// BitBuffer vs Indices comparison benchmarks
// =============================================================================

/// Compare: indices vs bitbuffer for sparse base, sparse rank (indices should win)
#[divan::bench(args = BASE_SIZES)]
fn cmp_indices_sparse_base_10_rank(bencher: Bencher, base_size: usize) {
    let base = create_sparse_mask(base_size, 0.1);
    let rank_len = base.true_count();
    let rank = create_dense_mask(rank_len, 0.1);

    bencher.bench(|| base.intersect_by_rank(&rank));
}

#[divan::bench(args = BASE_SIZES)]
fn cmp_bitbuf_sparse_base_10_rank(bencher: Bencher, base_size: usize) {
    let base = create_sparse_mask(base_size, 0.1);
    let rank_len = base.true_count();
    let rank = create_dense_mask(rank_len, 0.1);

    bencher.bench(|| base.intersect_by_rank_bitbuffer(&rank));
}

/// Compare: indices vs bitbuffer for sparse base, high density rank
#[divan::bench(args = BASE_SIZES)]
fn cmp_indices_sparse_base_90_rank(bencher: Bencher, base_size: usize) {
    let base = create_sparse_mask(base_size, 0.1);
    let rank_len = base.true_count();
    let rank = create_dense_mask(rank_len, 0.9);

    bencher.bench(|| base.intersect_by_rank(&rank));
}

#[divan::bench(args = BASE_SIZES)]
fn cmp_bitbuf_sparse_base_90_rank(bencher: Bencher, base_size: usize) {
    let base = create_sparse_mask(base_size, 0.1);
    let rank_len = base.true_count();
    let rank = create_dense_mask(rank_len, 0.9);

    bencher.bench(|| base.intersect_by_rank_bitbuffer(&rank));
}

/// Compare: indices vs bitbuffer for dense base, high density rank
#[divan::bench(args = BASE_SIZES)]
fn cmp_indices_dense_base_90_rank(bencher: Bencher, base_size: usize) {
    let base = create_dense_mask(base_size, 0.5);
    let rank_len = base.true_count();
    let rank = create_dense_mask(rank_len, 0.9);

    bencher.bench(|| base.intersect_by_rank(&rank));
}

#[divan::bench(args = BASE_SIZES)]
fn cmp_bitbuf_dense_base_90_rank(bencher: Bencher, base_size: usize) {
    let base = create_dense_mask(base_size, 0.5);
    let rank_len = base.true_count();
    let rank = create_dense_mask(rank_len, 0.9);

    bencher.bench(|| base.intersect_by_rank_bitbuffer(&rank));
}

/// Compare: indices vs bitbuffer for 99% rank density
#[divan::bench(args = BASE_SIZES)]
fn cmp_indices_sparse_base_99_rank(bencher: Bencher, base_size: usize) {
    let base = create_sparse_mask(base_size, 0.1);
    let rank_len = base.true_count();
    let rank = create_dense_mask(rank_len, 0.99);

    bencher.bench(|| base.intersect_by_rank(&rank));
}

#[divan::bench(args = BASE_SIZES)]
fn cmp_bitbuf_sparse_base_99_rank(bencher: Bencher, base_size: usize) {
    let base = create_sparse_mask(base_size, 0.1);
    let rank_len = base.true_count();
    let rank = create_dense_mask(rank_len, 0.99);

    bencher.bench(|| base.intersect_by_rank_bitbuffer(&rank));
}

// =============================================================================
// u64-based implementation benchmarks
// =============================================================================

#[divan::bench(args = BASE_SIZES)]
fn cmp_u64_sparse_base_10_rank(bencher: Bencher, base_size: usize) {
    let base = create_sparse_mask(base_size, 0.1);
    let rank_len = base.true_count();
    let rank = create_dense_mask(rank_len, 0.1);

    bencher.bench(|| base.intersect_by_rank_u64(&rank));
}

#[divan::bench(args = BASE_SIZES)]
fn cmp_u64_sparse_base_90_rank(bencher: Bencher, base_size: usize) {
    let base = create_sparse_mask(base_size, 0.1);
    let rank_len = base.true_count();
    let rank = create_dense_mask(rank_len, 0.9);

    bencher.bench(|| base.intersect_by_rank_u64(&rank));
}

#[divan::bench(args = BASE_SIZES)]
fn cmp_u64_dense_base_90_rank(bencher: Bencher, base_size: usize) {
    let base = create_dense_mask(base_size, 0.5);
    let rank_len = base.true_count();
    let rank = create_dense_mask(rank_len, 0.9);

    bencher.bench(|| base.intersect_by_rank_u64(&rank));
}

#[divan::bench(args = BASE_SIZES)]
fn cmp_u64_sparse_base_99_rank(bencher: Bencher, base_size: usize) {
    let base = create_sparse_mask(base_size, 0.1);
    let rank_len = base.true_count();
    let rank = create_dense_mask(rank_len, 0.99);

    bencher.bench(|| base.intersect_by_rank_u64(&rank));
}
