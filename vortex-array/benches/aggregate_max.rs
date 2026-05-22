// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use divan::Bencher;
use rand::prelude::*;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;

fn main() {
    divan::main();
}

const N: usize = 1_000_000;

fn gen_i32(null_density: f64, seed: u64) -> Vec<Option<i32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..N)
        .map(|_| {
            if rng.random_bool(null_density) {
                None
            } else {
                Some(rng.random::<i32>())
            }
        })
        .collect()
}

fn gen_i64(null_density: f64, seed: u64) -> Vec<Option<i64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..N)
        .map(|_| {
            if rng.random_bool(null_density) {
                None
            } else {
                Some(rng.random::<i64>())
            }
        })
        .collect()
}

fn gen_f64(null_density: f64, seed: u64) -> Vec<Option<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..N)
        .map(|_| {
            if rng.random_bool(null_density) {
                None
            } else {
                Some(rng.random::<f64>())
            }
        })
        .collect()
}

#[divan::bench]
fn max_i32_all_valid(bencher: Bencher) {
    let data = gen_i32(0.0, 1);
    bencher
        .with_inputs(|| PrimitiveArray::from_option_iter(data.iter().copied()).into_array())
        .bench_refs(|a| {
            a.statistics()
                .compute_max::<i32>(&mut LEGACY_SESSION.create_execution_ctx())
        });
}

#[divan::bench]
fn max_i32_half_null(bencher: Bencher) {
    let data = gen_i32(0.5, 2);
    bencher
        .with_inputs(|| PrimitiveArray::from_option_iter(data.iter().copied()).into_array())
        .bench_refs(|a| {
            a.statistics()
                .compute_max::<i32>(&mut LEGACY_SESSION.create_execution_ctx())
        });
}

#[divan::bench]
fn max_i64_all_valid(bencher: Bencher) {
    let data = gen_i64(0.0, 3);
    bencher
        .with_inputs(|| PrimitiveArray::from_option_iter(data.iter().copied()).into_array())
        .bench_refs(|a| {
            a.statistics()
                .compute_max::<i64>(&mut LEGACY_SESSION.create_execution_ctx())
        });
}

#[divan::bench]
fn max_f64_all_valid(bencher: Bencher) {
    let data = gen_f64(0.0, 4);
    bencher
        .with_inputs(|| PrimitiveArray::from_option_iter(data.iter().copied()).into_array())
        .bench_refs(|a| {
            a.statistics()
                .compute_max::<f64>(&mut LEGACY_SESSION.create_execution_ctx())
        });
}

#[divan::bench]
fn max_f64_half_null(bencher: Bencher) {
    let data = gen_f64(0.5, 5);
    bencher
        .with_inputs(|| PrimitiveArray::from_option_iter(data.iter().copied()).into_array())
        .bench_refs(|a| {
            a.statistics()
                .compute_max::<f64>(&mut LEGACY_SESSION.create_execution_ctx())
        });
}
