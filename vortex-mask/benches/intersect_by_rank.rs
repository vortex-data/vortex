// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for `intersect_by_rank`.

use divan::Bencher;
use vortex_buffer::BitBuffer;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

// Standard test cases
const BENCH_ARGS: &[(usize, &str)] = &[
    (10_000, "random"),
    (10_000, "runs"),
    (100_000, "random"),
    (100_000, "runs"),
];

// Sparse test cases (varying base selectivity)
const SPARSE_ARGS: &[(usize, f64, &str)] = &[
    (100_000, 0.01, "sparse_1pct"),
    (100_000, 0.05, "sparse_5pct"),
    (100_000, 0.10, "sparse_10pct"),
    (100_000, 0.50, "dense_50pct"),
];

// Four-case density matrix: (self_density, mask_density)
const DENSITY_MATRIX_ARGS: &[(f64, f64, &str)] = &[
    (0.05, 0.05, "self_sparse_mask_sparse"),
    (0.05, 0.50, "self_sparse_mask_dense"),
    (0.50, 0.05, "self_dense_mask_sparse"),
    (0.50, 0.50, "self_dense_mask_dense"),
];

// Second mask stored as cached indices rather than only as a BitBuffer.
const RANK_INDICES_ARGS: &[(f64, f64, &str)] = &[
    (0.01, 0.50, "self_very_sparse_rank_dense"),
    (0.05, 0.05, "self_sparse_rank_sparse"),
    (0.05, 0.50, "self_sparse_rank_dense"),
    (0.50, 0.01, "self_dense_rank_very_sparse"),
    (0.50, 0.05, "self_dense_rank_sparse"),
    (0.50, 0.50, "self_dense_rank_dense"),
];

// Cases that target the mask-driven path: dense self with very-sparse mask. Both
// uncached (mask backed only by a BitBuffer) and cached (mask carries indices).
// Uncached cases stress the path that previously walked every self chunk via the
// `set_indices` iterator; cached cases stress the path that scanned chunks against
// `&[usize]`.
const VERY_SPARSE_MASK_ARGS: &[(f64, f64, &str)] = &[
    (0.50, 0.005, "self_dense_mask_0p5pct"),
    (0.50, 0.01, "self_dense_mask_1pct"),
    (0.50, 0.02, "self_dense_mask_2pct"),
    (0.10, 0.01, "self_10pct_mask_1pct"),
];

fn create_random_mask(len: usize, selectivity: f64) -> Mask {
    Mask::from_buffer(BitBuffer::from_iter((0..len).map(|i| {
        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let threshold = (selectivity * 1000.0) as usize;
        (i * 7 + 13) % 1000 < threshold
    })))
}

fn create_random_indices_mask(len: usize, selectivity: f64) -> Mask {
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let threshold = (selectivity * 1000.0) as usize;
    Mask::from_indices(
        len,
        (0..len)
            .filter(|&i| (i * 7 + 13) % 1000 < threshold)
            .collect(),
    )
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

fn create_sparse_fixture(size: usize, selectivity: f64) -> (Mask, Mask) {
    let base = create_random_mask(size, selectivity);
    let rank_len = base.true_count();
    let rank = create_random_mask(rank_len, 0.5);
    (base, rank)
}

fn create_density_matrix_fixture(
    size: usize,
    self_density: f64,
    mask_density: f64,
) -> (Mask, Mask) {
    let base = create_random_mask(size, self_density);
    let rank_len = base.true_count();
    let rank = create_random_mask(rank_len, mask_density);
    (base, rank)
}

fn create_rank_indices_fixture(size: usize, self_density: f64, mask_density: f64) -> (Mask, Mask) {
    let base = create_random_mask(size, self_density);
    let rank_len = base.true_count();
    let rank = create_random_indices_mask(rank_len, mask_density);
    (base, rank)
}

/// Standard patterns (random / runs)
#[divan::bench(args = BENCH_ARGS)]
fn intersect_by_rank(bencher: Bencher, (size, pattern): (usize, &str)) {
    let (base, rank) = create_fixture(size, pattern);
    bencher
        .with_inputs(|| (&base, &rank))
        .bench_refs(|(base, rank)| base.intersect_by_rank(rank));
}

/// Sparse base masks (varying selectivity)
#[divan::bench(args = SPARSE_ARGS)]
fn sparse(bencher: Bencher, (size, selectivity, _name): (usize, f64, &str)) {
    let (base, rank) = create_sparse_fixture(size, selectivity);
    bencher
        .with_inputs(|| (&base, &rank))
        .bench_refs(|(base, rank)| base.intersect_by_rank(rank));
}

/// Density matrix (self_density x mask_density)
#[divan::bench(args = DENSITY_MATRIX_ARGS)]
fn density_matrix(bencher: Bencher, (self_density, mask_density, _name): (f64, f64, &str)) {
    let (base, rank) = create_density_matrix_fixture(100_000, self_density, mask_density);
    bencher
        .with_inputs(|| (&base, &rank))
        .bench_refs(|(base, rank)| base.intersect_by_rank(rank));
}

/// Density matrix where the rank mask is backed by cached indices.
#[divan::bench(args = RANK_INDICES_ARGS)]
fn rank_indices(bencher: Bencher, (self_density, mask_density, _name): (f64, f64, &str)) {
    let (base, rank) = create_rank_indices_fixture(100_000, self_density, mask_density);
    bencher
        .with_inputs(|| (&base, &rank))
        .bench_refs(|(base, rank)| base.intersect_by_rank(rank));
}

/// Very-sparse mask backed only by a BitBuffer (no cached indices). Targets the
/// mask-driven dispatch path.
#[divan::bench(args = VERY_SPARSE_MASK_ARGS)]
fn very_sparse_mask_uncached(
    bencher: Bencher,
    (self_density, mask_density, _name): (f64, f64, &str),
) {
    let (base, rank) = create_density_matrix_fixture(100_000, self_density, mask_density);
    bencher
        .with_inputs(|| (&base, &rank))
        .bench_refs(|(base, rank)| base.intersect_by_rank(rank));
}

/// Very-sparse mask carrying cached indices. Targets the mask-driven dispatch path
/// when the caller has already paid for index materialization.
#[divan::bench(args = VERY_SPARSE_MASK_ARGS)]
fn very_sparse_mask_cached(
    bencher: Bencher,
    (self_density, mask_density, _name): (f64, f64, &str),
) {
    let (base, rank) = create_rank_indices_fixture(100_000, self_density, mask_density);
    bencher
        .with_inputs(|| (&base, &rank))
        .bench_refs(|(base, rank)| base.intersect_by_rank(rank));
}
