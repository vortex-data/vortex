// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::prelude::*;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFn;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

// Sized to keep the CodSpeed simulation under 1ms per benchmark.
const N: usize = 15_000;

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

#[divan::bench]
fn max_i32(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);
    let data: Vec<i32> = (0..N).map(|_| rng.random::<i32>()).collect();
    let aggregate = AggregateFn::new(Max, NumericalAggregateOpts::default()).erased();

    bencher
        .with_inputs(|| {
            (
                PrimitiveArray::from_iter(data.iter().copied()).into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(a, ctx)| a.aggregations().compute_as::<i32>(&aggregate, ctx));
}

#[divan::bench]
fn max_i64(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(2);
    let data: Vec<i64> = (0..N).map(|_| rng.random::<i64>()).collect();
    let aggregate = AggregateFn::new(Max, NumericalAggregateOpts::default()).erased();

    bencher
        .with_inputs(|| {
            (
                PrimitiveArray::from_iter(data.iter().copied()).into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(a, ctx)| a.aggregations().compute_as::<i64>(&aggregate, ctx));
}

#[divan::bench]
fn max_f64(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(3);
    let data: Vec<f64> = (0..N).map(|_| rng.random::<f64>()).collect();
    let aggregate = AggregateFn::new(Max, NumericalAggregateOpts::default()).erased();

    bencher
        .with_inputs(|| {
            (
                PrimitiveArray::from_iter(data.iter().copied()).into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(a, ctx)| a.aggregations().compute_as::<f64>(&aggregate, ctx));
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
    let aggregate = AggregateFn::new(Max, NumericalAggregateOpts::default()).erased();

    bencher
        .with_inputs(|| {
            (
                PrimitiveArray::from_option_iter(data.iter().copied()).into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(a, ctx)| a.aggregations().compute_as::<i32>(&aggregate, ctx));
}

// Scattered nulls: ~50% random nulls producing many short runs (run-based path's worst case).
#[divan::bench]
fn max_i32_nulls_scattered(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(5);
    let data: Vec<Option<i32>> = (0..N)
        .map(|_| rng.random_bool(0.5).then(|| rng.random::<i32>()))
        .collect();
    let aggregate = AggregateFn::new(Max, NumericalAggregateOpts::default()).erased();

    bencher
        .with_inputs(|| {
            (
                PrimitiveArray::from_option_iter(data.iter().copied()).into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(a, ctx)| a.aggregations().compute_as::<i32>(&aggregate, ctx));
}
