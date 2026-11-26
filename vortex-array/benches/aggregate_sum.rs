// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

pub mod fixture;

use std::sync::LazyLock;

use divan::Bencher;
use fixture::VortexColumns;
use mimalloc::MiMalloc;
use populate_builtins::bernoulli;
use populate_builtins::draw_if;
use populate_builtins::indexed;
use populate_builtins::sample_range;
use populate_core::Columns;
use populate_core::Distribution;
use populate_core::Nullable;
use populate_rand::SeedableRng;
use populate_rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::sum_v2::sum_v2;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::stats::Stat;
use vortex_array::test_harness::mark_optimized;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

// Sized to keep the CodSpeed simulation under 1ms per benchmark.
const N: usize = 15_000;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    VortexSession::empty()
        .with::<KernelSession>()
        .with::<AggregateFnSession>()
});

fn batch<D: Distribution>(seed: u64, mut dist: D) -> Columns<D::Item> {
    dist.collect(&mut StdRng::seed_from_u64(seed), N)
}

fn clustered(i: usize) -> bool {
    !(i / 64).is_multiple_of(10)
}

// Statistics can be cached on the array. Give each measurement a fresh instance.
fn inputs(cols: impl Clone + VortexColumns + Sync) -> impl Fn() -> (ArrayRef, ExecutionCtx) {
    move || {
        let ctx = SESSION.create_execution_ctx();
        let array = cols.clone().into_vortex();
        mark_optimized(&array, &SESSION).unwrap();
        (array, ctx)
    }
}

#[divan::bench]
fn sum_i32(bencher: Bencher) {
    bencher
        .with_inputs(inputs(batch(1, sample_range(-1000i32..1000))))
        .bench_refs(|(a, ctx)| a.statistics().compute_as::<i64>(Stat::Sum, ctx));
}

#[divan::bench]
fn sum_v2_i32(bencher: Bencher) {
    bencher
        .with_inputs(inputs(batch(1, sample_range(-1000i32..1000))))
        .bench_refs(|(a, ctx)| sum_v2(a, ctx));
}

#[divan::bench]
fn sum_u32(bencher: Bencher) {
    bencher
        .with_inputs(inputs(batch(2, sample_range(0u32..2000))))
        .bench_refs(|(a, ctx)| a.statistics().compute_as::<u64>(Stat::Sum, ctx));
}

#[divan::bench]
fn sum_v2_u32(bencher: Bencher) {
    bencher
        .with_inputs(inputs(batch(2, sample_range(0u32..2000))))
        .bench_refs(|(a, ctx)| sum_v2(a, ctx));
}

#[divan::bench]
fn sum_i64(bencher: Bencher) {
    bencher
        .with_inputs(inputs(batch(3, sample_range(-1000i64..1000))))
        .bench_refs(|(a, ctx)| a.statistics().compute_as::<i64>(Stat::Sum, ctx));
}

#[divan::bench]
fn sum_v2_i64(bencher: Bencher) {
    bencher
        .with_inputs(inputs(batch(3, sample_range(-1000i64..1000))))
        .bench_refs(|(a, ctx)| sum_v2(a, ctx));
}

#[divan::bench]
fn sum_f64(bencher: Bencher) {
    bencher
        .with_inputs(inputs(batch(6, sample_range(-1000.0f64..1000.0))))
        .bench_refs(|(a, ctx)| a.statistics().compute_as::<f64>(Stat::Sum, ctx));
}

#[divan::bench]
fn sum_v2_f64(bencher: Bencher) {
    bencher
        .with_inputs(inputs(batch(6, sample_range(-1000.0f64..1000.0))))
        .bench_refs(|(a, ctx)| sum_v2(a, ctx));
}

fn f64_nulls_clustered() -> Nullable<Vec<f64>> {
    batch(
        7,
        draw_if(indexed(clustered), sample_range(-1000.0f64..1000.0)),
    )
}

#[divan::bench]
fn sum_f64_nulls_clustered(bencher: Bencher) {
    bencher
        .with_inputs(inputs(f64_nulls_clustered()))
        .bench_refs(|(a, ctx)| a.statistics().compute_as::<f64>(Stat::Sum, ctx));
}

#[divan::bench]
fn sum_v2_f64_nulls_clustered(bencher: Bencher) {
    bencher
        .with_inputs(inputs(f64_nulls_clustered()))
        .bench_refs(|(a, ctx)| sum_v2(a, ctx));
}

fn i32_nulls_clustered() -> Nullable<Vec<i32>> {
    batch(4, draw_if(indexed(clustered), sample_range(-1000i32..1000)))
}

#[divan::bench]
fn sum_i32_nulls_clustered(bencher: Bencher) {
    bencher
        .with_inputs(inputs(i32_nulls_clustered()))
        .bench_refs(|(a, ctx)| a.statistics().compute_as::<i64>(Stat::Sum, ctx));
}

#[divan::bench]
fn sum_v2_i32_nulls_clustered(bencher: Bencher) {
    bencher
        .with_inputs(inputs(i32_nulls_clustered()))
        .bench_refs(|(a, ctx)| sum_v2(a, ctx));
}

fn i32_nulls_scattered() -> Nullable<Vec<i32>> {
    batch(5, draw_if(bernoulli(0.5), sample_range(-1000i32..1000)))
}

#[divan::bench]
fn sum_i32_nulls_scattered(bencher: Bencher) {
    bencher
        .with_inputs(inputs(i32_nulls_scattered()))
        .bench_refs(|(a, ctx)| a.statistics().compute_as::<i64>(Stat::Sum, ctx));
}

#[divan::bench]
fn sum_v2_i32_nulls_scattered(bencher: Bencher) {
    bencher
        .with_inputs(inputs(i32_nulls_scattered()))
        .bench_refs(|(a, ctx)| sum_v2(a, ctx));
}
