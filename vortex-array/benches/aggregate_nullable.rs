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
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::nan_count::nan_count;
use vortex_array::aggregate_fn::fns::null_count::null_count;
use vortex_array::aggregate_fn::fns::sum::sum;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DecimalDType;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::test_harness::mark_optimized;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
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
    input: impl Fn(&mut ExecutionCtx) -> ArrayRef + Sync,
    f: impl Fn(&ArrayRef, &mut ExecutionCtx) -> O + Sync,
) {
    bencher
        .with_inputs(|| {
            let mut ctx = SESSION.create_execution_ctx();
            let array = input(&mut ctx);
            mark_optimized(&array, &SESSION).vortex_expect("mark benchmark input as optimized");
            (array, ctx)
        })
        .bench_refs(|(array, ctx)| f(array, ctx));
}

fn bench_decimal_sum(bencher: Bencher, validity: Validity) {
    let values = i32_values();
    bench_with(
        bencher,
        |ctx| {
            DecimalArray::new(
                values.clone(),
                DecimalDType::new(9, 2),
                validity.clone(),
                ctx,
            )
            .into_array()
        },
        sum,
    );
}

#[divan::bench]
fn sum_decimal_i32_all_valid(bencher: Bencher) {
    bench_decimal_sum(bencher, Validity::NonNullable);
}

#[divan::bench]
fn sum_decimal_i32_nulls_scattered(bencher: Bencher) {
    bench_decimal_sum(bencher, scattered_validity());
}

#[divan::bench]
fn nan_count_f64_nulls_scattered(bencher: Bencher) {
    let (values, validity) = (f64_values(), scattered_validity());
    bench_with(
        bencher,
        |_| PrimitiveArray::new(values.clone(), validity.clone()).into_array(),
        nan_count,
    );
}

/// Mask a nullable column, as a parent's nulls mask a field, and count its nulls, which reads the
/// conjunction of the two validities.
#[divan::bench]
fn null_count_i32_masked(bencher: Bencher) {
    let values = i32_values();
    let validity = scattered_validity();
    let mut rng = StdRng::seed_from_u64(6);
    let mask = BoolArray::from_iter((0..N).map(|_| rng.random_bool(0.5))).into_array();
    mark_optimized(&mask, &SESSION).vortex_expect("mark benchmark mask as optimized");
    bench_with(
        bencher,
        |_| PrimitiveArray::new(values.clone(), validity.clone()).into_array(),
        |array, ctx| {
            let masked = array.clone().mask(mask.clone())?;
            null_count(&masked, ctx)
        },
    );
}

fn i32_values() -> Buffer<i32> {
    let mut rng = StdRng::seed_from_u64(7);
    (0..N).map(|_| rng.random_range(-1000..1000)).collect()
}
