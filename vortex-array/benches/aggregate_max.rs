// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use divan::Bencher;
use rand::prelude::*;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::session::ArraySession;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

const N: usize = 100_000;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

#[divan::bench]
fn max_i32(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);
    let data: Vec<i32> = (0..N).map(|_| rng.random::<i32>()).collect();
    bencher
        .with_inputs(|| {
            (
                PrimitiveArray::from_iter(data.iter().copied()).into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(a, ctx)| a.statistics().compute_max::<i32>(ctx));
}

#[divan::bench]
fn max_i64(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(2);
    let data: Vec<i64> = (0..N).map(|_| rng.random::<i64>()).collect();
    bencher
        .with_inputs(|| {
            (
                PrimitiveArray::from_iter(data.iter().copied()).into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(a, ctx)| a.statistics().compute_max::<i64>(ctx));
}

#[divan::bench]
fn max_f64(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(3);
    let data: Vec<f64> = (0..N).map(|_| rng.random::<f64>()).collect();
    bencher
        .with_inputs(|| {
            (
                PrimitiveArray::from_iter(data.iter().copied()).into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(a, ctx)| a.statistics().compute_max::<f64>(ctx));
}

// Clustered nulls: long valid runs broken up by null blocks (run-based path's best case).
#[divan::bench]
fn max_i32_nulls_clustered(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(4);
    let data: Vec<Option<i32>> = (0..N)
        .map(|i| {
            if (i / 64) % 10 == 0 {
                None
            } else {
                Some(rng.random::<i32>())
            }
        })
        .collect();
    bencher
        .with_inputs(|| {
            (
                PrimitiveArray::from_option_iter(data.iter().copied()).into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(a, ctx)| a.statistics().compute_max::<i32>(ctx));
}

// Scattered nulls: ~50% random nulls producing many short runs (run-based path's worst case).
#[divan::bench]
fn max_i32_nulls_scattered(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(5);
    let data: Vec<Option<i32>> = (0..N)
        .map(|_| rng.random_bool(0.5).then(|| rng.random::<i32>()))
        .collect();
    bencher
        .with_inputs(|| {
            (
                PrimitiveArray::from_option_iter(data.iter().copied()).into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(a, ctx)| a.statistics().compute_max::<i32>(ctx));
}
