// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for filtering BoolArray by a boolean mask.
//!
//! Tests multiple mask patterns (mostly-true, mostly-false, random, correlated runs)
//! with both uniform-random and power-law distributions, across array sizes from 1K to 250K.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::cast_sign_loss)]
#![expect(clippy::cast_precision_loss)]

use std::sync::LazyLock;

use divan::Bencher;
use rand::prelude::*;
use rand_distr::Zipf;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::session::ArraySession;
use vortex_buffer::BitBuffer;
use vortex_mask::Mask;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

const SIZES: &[usize] = &[1_000, 10_000, 100_000, 250_000];
const DENSITY_SWEEP_SIZE: usize = 100_000;
const ARRAY_SEED: u64 = 42;
const MASK_SEED: u64 = 43;

// --- Mask generators ---

/// Mostly-true mask: ~90% true bits.
fn make_mostly_true(len: usize, rng: &mut StdRng) -> Mask {
    Mask::from_buffer(BitBuffer::from_iter(
        (0..len).map(|_| rng.random_ratio(9, 10)),
    ))
}

/// Mostly-false mask: ~10% true bits.
fn make_mostly_false(len: usize, rng: &mut StdRng) -> Mask {
    Mask::from_buffer(BitBuffer::from_iter(
        (0..len).map(|_| rng.random_ratio(1, 10)),
    ))
}

/// Random 50/50 mask.
fn make_random(len: usize, rng: &mut StdRng) -> Mask {
    Mask::from_buffer(BitBuffer::from_iter((0..len).map(|_| rng.random_bool(0.5))))
}

/// Correlated runs: alternating runs of true/false with geometrically distributed lengths.
fn make_correlated_runs(len: usize, rng: &mut StdRng) -> Mask {
    let mut bits = Vec::with_capacity(len);
    let mut current = true;
    while bits.len() < len {
        // Average run length ~64
        let run_len = (rng.random::<f64>().ln() / (1.0_f64 - 1.0 / 64.0).ln()) as usize + 1;
        let run_len = run_len.min(len - bits.len());
        bits.extend(std::iter::repeat_n(current, run_len));
        current = !current;
    }
    Mask::from_buffer(BitBuffer::from_iter(bits))
}

/// Power-law (Zipfian) mask: indices chosen from a Zipf distribution, so lower indices are
/// much more likely to be selected.
fn make_power_law(len: usize, rng: &mut StdRng) -> Mask {
    let zipf = Zipf::new(len as f64, 1.0).unwrap();
    let num_selected = len / 2;
    let mut indices: Vec<usize> = rng
        .sample_iter(&zipf)
        .take(num_selected * 2)
        .map(|v: f64| (v as usize).saturating_sub(1).min(len - 1))
        .collect();
    indices.sort_unstable();
    indices.dedup();
    indices.truncate(num_selected);
    Mask::from_indices(len, indices)
}

/// Mask with exact density: fraction of bits set to true.
fn make_density_mask(len: usize, density: f64, rng: &mut StdRng) -> Mask {
    Mask::from_buffer(BitBuffer::from_iter(
        (0..len).map(|_| rng.random_bool(density)),
    ))
}

/// Dense runs mask: long runs of true with rare false gaps.
fn make_dense_runs(len: usize, false_rate: f64, rng: &mut StdRng) -> Mask {
    let mut bits = Vec::with_capacity(len);
    let mut current = true;
    while bits.len() < len {
        let avg_run = if current {
            (1.0 / false_rate).max(1.0)
        } else {
            2.0
        };
        let run_len = (rng.random::<f64>().ln() / (1.0 - 1.0 / avg_run).ln()).max(1.0) as usize;
        let run_len = run_len.min(len - bits.len());
        bits.extend(std::iter::repeat_n(current, run_len));
        current = !current;
    }
    Mask::from_buffer(BitBuffer::from_iter(bits))
}

/// Single contiguous block of true values (best case for slice path).
fn make_single_slice(len: usize, density: f64) -> Mask {
    let true_count = (len as f64 * density) as usize;
    let start = (len - true_count) / 2;
    Mask::from_indices(len, start..start + true_count)
}

// --- Source array generators ---

fn make_random_bool_array(len: usize, rng: &mut StdRng) -> BoolArray {
    BoolArray::from_iter((0..len).map(|_| rng.random_bool(0.5)))
}

fn make_power_law_bool_array(len: usize, rng: &mut StdRng) -> BoolArray {
    let zipf = Zipf::new(len as f64, 1.0).unwrap();
    BoolArray::from_iter((0..len).map(|i| {
        let threshold: f64 = rng.sample(zipf);
        (i as f64) < threshold
    }))
}

/// Create a fresh StdRng with the mask seed.
fn mask_rng() -> StdRng {
    StdRng::seed_from_u64(MASK_SEED)
}

// --- Benchmarks: Random source array ---

#[divan::bench(args = SIZES)]
fn filter_random_by_mostly_true(bencher: Bencher, n: usize) {
    let array = make_random_bool_array(n, &mut StdRng::seed_from_u64(ARRAY_SEED));
    bencher
        .with_inputs(|| {
            (
                array.clone(),
                make_mostly_true(n, &mut mask_rng()),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, m, ctx)| {
            array
                .clone()
                .filter(m.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
        });
}

#[divan::bench(args = SIZES)]
fn filter_random_by_mostly_false(bencher: Bencher, n: usize) {
    let array = make_random_bool_array(n, &mut StdRng::seed_from_u64(ARRAY_SEED));
    bencher
        .with_inputs(|| {
            (
                array.clone(),
                make_mostly_false(n, &mut mask_rng()),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, m, ctx)| {
            array
                .clone()
                .filter(m.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
        });
}

#[divan::bench(args = SIZES)]
fn filter_random_by_random(bencher: Bencher, n: usize) {
    let array = make_random_bool_array(n, &mut StdRng::seed_from_u64(ARRAY_SEED));
    bencher
        .with_inputs(|| {
            (
                array.clone(),
                make_random(n, &mut mask_rng()),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, m, ctx)| {
            array
                .clone()
                .filter(m.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
        });
}

#[divan::bench(args = SIZES)]
fn filter_random_by_correlated_runs(bencher: Bencher, n: usize) {
    let array = make_random_bool_array(n, &mut StdRng::seed_from_u64(ARRAY_SEED));
    bencher
        .with_inputs(|| {
            (
                array.clone(),
                make_correlated_runs(n, &mut mask_rng()),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, m, ctx)| {
            array
                .clone()
                .filter(m.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
        });
}

#[divan::bench(args = SIZES)]
fn filter_random_by_power_law(bencher: Bencher, n: usize) {
    let array = make_random_bool_array(n, &mut StdRng::seed_from_u64(ARRAY_SEED));
    bencher
        .with_inputs(|| {
            (
                array.clone(),
                make_power_law(n, &mut mask_rng()),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, m, ctx)| {
            array
                .clone()
                .filter(m.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
        });
}

// --- Benchmarks: Power-law source array ---

#[divan::bench(args = SIZES)]
fn filter_powerlaw_by_mostly_true(bencher: Bencher, n: usize) {
    let array = make_power_law_bool_array(n, &mut StdRng::seed_from_u64(ARRAY_SEED));
    bencher
        .with_inputs(|| {
            (
                array.clone(),
                make_mostly_true(n, &mut mask_rng()),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, m, ctx)| {
            array
                .clone()
                .filter(m.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
        });
}

#[divan::bench(args = SIZES)]
fn filter_powerlaw_by_mostly_false(bencher: Bencher, n: usize) {
    let array = make_power_law_bool_array(n, &mut StdRng::seed_from_u64(ARRAY_SEED));
    bencher
        .with_inputs(|| {
            (
                array.clone(),
                make_mostly_false(n, &mut mask_rng()),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, m, ctx)| {
            array
                .clone()
                .filter(m.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
        });
}

#[divan::bench(args = SIZES)]
fn filter_powerlaw_by_random(bencher: Bencher, n: usize) {
    let array = make_power_law_bool_array(n, &mut StdRng::seed_from_u64(ARRAY_SEED));
    bencher
        .with_inputs(|| {
            (
                array.clone(),
                make_random(n, &mut mask_rng()),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, m, ctx)| {
            array
                .clone()
                .filter(m.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
        });
}

#[divan::bench(args = SIZES)]
fn filter_powerlaw_by_correlated_runs(bencher: Bencher, n: usize) {
    let array = make_power_law_bool_array(n, &mut StdRng::seed_from_u64(ARRAY_SEED));
    bencher
        .with_inputs(|| {
            (
                array.clone(),
                make_correlated_runs(n, &mut mask_rng()),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, m, ctx)| {
            array
                .clone()
                .filter(m.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
        });
}

#[divan::bench(args = SIZES)]
fn filter_powerlaw_by_power_law(bencher: Bencher, n: usize) {
    let array = make_power_law_bool_array(n, &mut StdRng::seed_from_u64(ARRAY_SEED));
    bencher
        .with_inputs(|| {
            (
                array.clone(),
                make_power_law(n, &mut mask_rng()),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, m, ctx)| {
            array
                .clone()
                .filter(m.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
        });
}

// --- Density sweep ---

const DENSITIES: &[f64] = &[
    0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 0.999, 0.9999,
];

#[divan::bench(args = DENSITIES)]
fn density_sweep_random(bencher: Bencher, density: f64) {
    let array = make_random_bool_array(DENSITY_SWEEP_SIZE, &mut StdRng::seed_from_u64(ARRAY_SEED));
    bencher
        .with_inputs(|| {
            (
                array.clone(),
                make_density_mask(DENSITY_SWEEP_SIZE, density, &mut mask_rng()),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, m, ctx)| {
            array
                .clone()
                .filter(m.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
        });
}

#[divan::bench(args = DENSITIES)]
fn density_sweep_dense_runs(bencher: Bencher, density: f64) {
    let array = make_random_bool_array(DENSITY_SWEEP_SIZE, &mut StdRng::seed_from_u64(ARRAY_SEED));
    let false_rate = (1.0 - density).max(0.0001);
    bencher
        .with_inputs(|| {
            (
                array.clone(),
                make_dense_runs(DENSITY_SWEEP_SIZE, false_rate, &mut mask_rng()),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, m, ctx)| {
            array
                .clone()
                .filter(m.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
        });
}

#[divan::bench(args = DENSITIES)]
fn density_sweep_single_slice(bencher: Bencher, density: f64) {
    let array = make_random_bool_array(DENSITY_SWEEP_SIZE, &mut StdRng::seed_from_u64(ARRAY_SEED));
    bencher
        .with_inputs(|| {
            (
                array.clone(),
                make_single_slice(DENSITY_SWEEP_SIZE, density),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, m, ctx)| {
            array
                .clone()
                .filter(m.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
        });
}

// --- Extreme cases ---

const LARGE_SIZES: &[usize] = &[10_000, 100_000, 250_000];

#[divan::bench(args = LARGE_SIZES)]
fn filter_all_true(bencher: Bencher, n: usize) {
    let array = make_random_bool_array(n, &mut StdRng::seed_from_u64(ARRAY_SEED));
    bencher
        .with_inputs(|| {
            (
                array.clone(),
                Mask::new_true(n),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, m, ctx)| {
            array
                .clone()
                .filter(m.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
        });
}

#[divan::bench(args = LARGE_SIZES)]
fn filter_one_false(bencher: Bencher, n: usize) {
    let array = make_random_bool_array(n, &mut StdRng::seed_from_u64(ARRAY_SEED));
    bencher
        .with_inputs(|| {
            let mut bits: Vec<bool> = vec![true; n];
            bits[n / 2] = false;
            (
                array.clone(),
                Mask::from_buffer(BitBuffer::from_iter(bits)),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, m, ctx)| {
            array
                .clone()
                .filter(m.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
        });
}

#[divan::bench(args = LARGE_SIZES)]
fn filter_ultra_sparse(bencher: Bencher, n: usize) {
    let array = make_random_bool_array(n, &mut StdRng::seed_from_u64(ARRAY_SEED));
    bencher
        .with_inputs(|| {
            (
                array.clone(),
                make_density_mask(n, 0.0001, &mut mask_rng()),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, m, ctx)| {
            array
                .clone()
                .filter(m.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
        });
}
