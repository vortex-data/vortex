// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Aggregates over columns whose nulls are scattered at random, so their valid values form runs
//! of one or two rows rather than the long runs of the clustered benchmarks.
//!
//! Each iteration gets a fresh array, so no statistic cached by a previous iteration is reused.

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::prelude::*;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::min_max::min_max;
use vortex_array::aggregate_fn::fns::nan_count::nan_count;
use vortex_array::aggregate_fn::fns::sum::sum;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DecimalDType;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
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

/// Half the rows, chosen at random, are valid.
fn scattered_validity() -> Validity {
    let mut rng = StdRng::seed_from_u64(1);
    Validity::from_iter((0..N).map(|_| rng.random_bool(0.5)))
}

fn f64_values() -> Buffer<f64> {
    let mut rng = StdRng::seed_from_u64(2);
    (0..N).map(|_| rng.random_range(-1000.0..1000.0)).collect()
}

fn bench_with<O>(
    bencher: Bencher,
    input: impl Fn() -> ArrayRef + Sync,
    f: impl Fn(&ArrayRef) -> O + Sync,
) {
    bencher.with_inputs(&input).bench_refs(|array| f(array));
}

fn bench_min_max(bencher: Bencher, input: impl Fn() -> ArrayRef + Sync) {
    bench_with(bencher, input, |array| {
        min_max(
            array,
            &mut SESSION.create_execution_ctx(),
            NumericalAggregateOpts::default(),
        )
    });
}

fn bench_sum(bencher: Bencher, input: impl Fn() -> ArrayRef + Sync) {
    bench_with(bencher, input, |array| {
        sum(array, &mut SESSION.create_execution_ctx())
    });
}

#[divan::bench]
fn min_max_f64_all_valid(bencher: Bencher) {
    let values = f64_values();
    bench_min_max(bencher, || {
        PrimitiveArray::new(values.clone(), Validity::NonNullable).into_array()
    });
}

#[divan::bench]
fn min_max_f64_nulls_scattered(bencher: Bencher) {
    let (values, validity) = (f64_values(), scattered_validity());
    bench_min_max(bencher, || {
        PrimitiveArray::new(values.clone(), validity.clone()).into_array()
    });
}

#[divan::bench]
fn min_max_decimal_i64_nulls_scattered(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(3);
    let values: Buffer<i64> = (0..N)
        .map(|_| rng.random_range(-100_000..100_000))
        .collect();
    let validity = scattered_validity();
    bench_min_max(bencher, || {
        DecimalArray::new(values.clone(), DecimalDType::new(18, 2), validity.clone()).into_array()
    });
}

#[divan::bench]
fn sum_i32_nulls_scattered(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(4);
    let values: Buffer<i32> = (0..N).map(|_| rng.random_range(-1000..1000)).collect();
    let validity = scattered_validity();
    bench_sum(bencher, || {
        PrimitiveArray::new(values.clone(), validity.clone()).into_array()
    });
}

#[divan::bench]
fn sum_f64_nulls_scattered(bencher: Bencher) {
    let (values, validity) = (f64_values(), scattered_validity());
    bench_sum(bencher, || {
        PrimitiveArray::new(values.clone(), validity.clone()).into_array()
    });
}

#[divan::bench]
fn nan_count_f64_nulls_scattered(bencher: Bencher) {
    let (values, validity) = (f64_values(), scattered_validity());
    bench_with(
        bencher,
        || PrimitiveArray::new(values.clone(), validity.clone()).into_array(),
        |array| nan_count(array, &mut SESSION.create_execution_ctx()),
    );
}
