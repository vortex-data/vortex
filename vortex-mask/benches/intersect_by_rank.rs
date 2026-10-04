// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for `intersect_by_rank`.
//!
//! `random_rotating` carries `#[cpu_features]`, so it is measured in walltime on every
//! CPU-feature leg rather than in simulation. The entry point picks its kernel at runtime:
//! the x86 legs measure the BMI2 kernel and the NEON leg the portable one. On mixed masks
//! the portable kernel spends most of its time in branch mispredictions, which only a
//! walltime measurement on real hardware shows.

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_buffer::BitBuffer;
use vortex_mask::Mask;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

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

// Independent random bits: (self_density, mask_density).
const RANDOM_ROTATING_ARGS: &[(f64, f64, &str)] = &[
    (0.50, 0.50, "self_50pct_mask_50pct"),
    (0.90, 0.90, "self_90pct_mask_90pct"),
    (0.02, 0.50, "self_2pct_mask_50pct"),
    (0.50, 0.02, "self_50pct_mask_2pct"),
];

/// Rows of `self` in each pair of the pool.
const ROTATING_ROWS: usize = 1 << 18;

/// Pairs the benchmark rotates through, one per iteration. The pool holds 32K words of
/// `self`, twice the repeated sequence a Zen 5 predictor was seen to learn, in at most
/// about 0.5 MiB, which stays in L2 on the walltime legs.
const ROTATING_POOL: usize = 8;

fn create_bernoulli_mask(rng: &mut StdRng, len: usize, density: f64) -> Mask {
    Mask::from_buffer(BitBuffer::from_iter(
        (0..len).map(|_| rng.random_bool(density)),
    ))
}

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
    Mask::from_indices(len, (0..len).filter(|&i| (i * 7 + 13) % 1000 < threshold))
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

/// Random masks, a different pair on every iteration.
#[vortex_bench_support::cpu_features]
#[divan::bench(args = RANDOM_ROTATING_ARGS)]
fn random_rotating(bencher: Bencher, (self_density, mask_density, _name): (f64, f64, &str)) {
    let mut rng = StdRng::seed_from_u64(0);
    let pool: Vec<(Mask, Mask)> = (0..ROTATING_POOL)
        .map(|_| {
            let base = create_bernoulli_mask(&mut rng, ROTATING_ROWS, self_density);
            let rank = create_bernoulli_mask(&mut rng, base.true_count(), mask_density);
            (base, rank)
        })
        .collect();
    // Divan requires the input generator to be `Fn + Sync`, hence the atomic counter.
    let next = AtomicUsize::new(0);
    bencher
        .with_inputs(|| &pool[next.fetch_add(1, Ordering::Relaxed) % ROTATING_POOL])
        .bench_refs(|(base, rank)| base.intersect_by_rank(rank));
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
