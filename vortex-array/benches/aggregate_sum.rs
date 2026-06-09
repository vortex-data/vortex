// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use divan::Bencher;
use rand::prelude::*;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::expr::stats::Stat;
use vortex_array::session::ArraySession;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

const N: usize = 100_000;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

#[divan::bench]
fn sum_i32(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);
    let data: Vec<i32> = (0..N).map(|_| rng.random_range(-1000..1000)).collect();
    bencher
        .with_inputs(|| {
            (
                PrimitiveArray::from_iter(data.iter().copied()).into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(a, ctx)| a.statistics().compute_as::<i64>(Stat::Sum, ctx));
}

#[divan::bench]
fn sum_u32(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(2);
    let data: Vec<u32> = (0..N).map(|_| rng.random_range(0..2000)).collect();
    bencher
        .with_inputs(|| {
            (
                PrimitiveArray::from_iter(data.iter().copied()).into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(a, ctx)| a.statistics().compute_as::<u64>(Stat::Sum, ctx));
}

#[divan::bench]
fn sum_i64(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(3);
    let data: Vec<i64> = (0..N).map(|_| rng.random_range(-1000..1000)).collect();
    bencher
        .with_inputs(|| {
            (
                PrimitiveArray::from_iter(data.iter().copied()).into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(a, ctx)| a.statistics().compute_as::<i64>(Stat::Sum, ctx));
}

// Clustered nulls: long runs of valid values broken up by occasional null blocks. This is the
// case the run-based valid path is expected to accelerate.
#[divan::bench]
fn sum_i32_nulls_clustered(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(4);
    let data: Vec<Option<i32>> = (0..N)
        .map(|i| {
            if (i / 64) % 10 == 0 {
                None
            } else {
                Some(rng.random_range(-1000..1000))
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
        .bench_refs(|(a, ctx)| a.statistics().compute_as::<i64>(Stat::Sum, ctx));
}

// Scattered nulls: ~50% nulls placed at random, producing many short runs. This is the worst case
// for a run-based valid path, used to guard against regressions versus a per-element loop.
#[divan::bench]
fn sum_i32_nulls_scattered(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(5);
    let data: Vec<Option<i32>> = (0..N)
        .map(|_| rng.random_bool(0.5).then(|| rng.random_range(-1000..1000)))
        .collect();
    bencher
        .with_inputs(|| {
            (
                PrimitiveArray::from_option_iter(data.iter().copied()).into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(a, ctx)| a.statistics().compute_as::<i64>(Stat::Sum, ctx));
}
