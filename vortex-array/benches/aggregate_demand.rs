// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Accumulating the demanded rows of a batch, compared with accumulating a nullable batch whose
//! validity equals the demand, and with filtering the batch before accumulating it.

#![allow(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::prelude::*;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::count::Count;
use vortex_array::aggregate_fn::fns::is_constant::IsConstant;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::aggregate_fn::fns::min::Min;
use vortex_array::aggregate_fn::fns::null_count::NullCount;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_mask::Mask;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    check_results();
    divan::main();
}

// Sized to keep the CodSpeed simulation under 1ms per benchmark.
const N: usize = 15_000;

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

const AGGREGATES: &[&str] = &["sum", "min", "max", "count", "null_count", "is_constant"];

fn aggregate(name: &str) -> AggregateFnRef {
    let opts = NumericalAggregateOpts::default();
    match name {
        "sum" => Sum.bind(opts),
        "min" => Min.bind(opts),
        "max" => Max.bind(opts),
        "count" => Count.bind(opts),
        "null_count" => NullCount.bind(EmptyOptions),
        "is_constant" => IsConstant.bind(EmptyOptions),
        _ => unreachable!("unknown aggregate {name}"),
    }
}

fn values() -> Buffer<i32> {
    let mut rng = StdRng::seed_from_u64(1);
    (0..N).map(|_| rng.random_range(-1000..1000)).collect()
}

/// The field's own validity: a tenth of its values are null.
fn field_validity() -> Mask {
    let mut rng = StdRng::seed_from_u64(5);
    Mask::from_iter((0..N).map(|_| rng.random_bool(0.9)))
}

/// A nullable struct field, holding values in every row including those the struct nulls out.
fn field() -> ArrayRef {
    PrimitiveArray::new(
        values(),
        Validity::from_mask(field_validity(), Nullability::Nullable),
    )
    .into_array()
}

/// Half the rows, chosen at random, as the struct validity of a nested field would be.
fn demand() -> Mask {
    let mut rng = StdRng::seed_from_u64(2);
    Mask::from_iter((0..N).map(|_| rng.random_bool(0.5)))
}

/// The logical field: null wherever the struct or the field is null.
fn masked_field() -> ArrayRef {
    PrimitiveArray::new(
        values(),
        Validity::from_mask(&field_validity() & &demand(), Nullability::Nullable),
    )
    .into_array()
}

fn run(aggregate_fn: &AggregateFnRef, batch: &ArrayRef, demand: Option<&Mask>) -> Scalar {
    let mut ctx = SESSION.create_execution_ctx();
    let mut acc = aggregate_fn.accumulator(batch.dtype()).unwrap();
    match demand {
        Some(demand) => acc.accumulate_demanded(batch, demand, &mut ctx).unwrap(),
        None => acc.accumulate(batch, &mut ctx).unwrap(),
    }
    acc.finish().unwrap()
}

/// Check that every benchmark computes the right result before timing it.
///
/// The demanded rows give the same result as the filtered batch for every aggregate. The null count
/// counts only the field's nulls in demanded rows, while aggregates that skip nulls match the
/// logical field, whose undemanded rows are null.
fn check_results() {
    let (field, demand) = (field(), demand());
    let filtered = field.filter(demand.clone()).unwrap();
    for name in AGGREGATES {
        let aggregate_fn = aggregate(name);
        let demanded = run(&aggregate_fn, &field, Some(&demand));
        assert_eq!(demanded, run(&aggregate_fn, &filtered, None), "{name}");
        if matches!(*name, "sum" | "min" | "max" | "count") {
            assert_eq!(
                demanded,
                run(&aggregate_fn, &masked_field(), None),
                "{name}"
            );
        }
    }

    let expected_nulls = (&demand & &!&field_validity()).true_count() as u64;
    assert!(expected_nulls > 0);
    let null_count = run(&aggregate("null_count"), &field, Some(&demand));
    assert_eq!(
        null_count.as_primitive().typed_value::<u64>(),
        Some(expected_nulls)
    );
}

/// Accumulate the demanded rows of the field.
#[divan::bench(args = AGGREGATES)]
fn demanded(bencher: Bencher, name: &str) {
    let aggregate_fn = aggregate(name);
    let batch = field();
    let demand = demand();
    bencher
        .with_inputs(|| {
            (
                aggregate_fn.accumulator(batch.dtype()).unwrap(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(mut acc, mut ctx)| {
            acc.accumulate_demanded(&batch, &demand, &mut ctx).unwrap();
            acc.flush().unwrap()
        });
}

/// Accumulate the logical field, whose validity already excludes the undemanded rows.
#[divan::bench(args = AGGREGATES)]
fn nullable(bencher: Bencher, name: &str) {
    let aggregate_fn = aggregate(name);
    bencher
        .with_inputs(|| {
            // A fresh batch per iteration, so no statistic cached by a previous one is reused.
            let batch = masked_field();
            (
                aggregate_fn.accumulator(batch.dtype()).unwrap(),
                batch,
                SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(mut acc, batch, mut ctx)| {
            acc.accumulate(&batch, &mut ctx).unwrap();
            acc.flush().unwrap()
        });
}

/// Filter the field down to its demanded rows, then accumulate it.
#[divan::bench(args = AGGREGATES)]
fn filtered(bencher: Bencher, name: &str) {
    let aggregate_fn = aggregate(name);
    let batch = field();
    let demand = demand();
    bencher
        .with_inputs(|| {
            (
                aggregate_fn.accumulator(batch.dtype()).unwrap(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(mut acc, mut ctx)| {
            let filtered = batch.filter(demand.clone()).unwrap();
            acc.accumulate(&filtered, &mut ctx).unwrap();
            acc.flush().unwrap()
        });
}

const DENSITIES: &[f64] = &[0.01, 0.5, 0.99];

fn demand_with_density(density: f64) -> Mask {
    let mut rng = StdRng::seed_from_u64(3);
    Mask::from_iter((0..N).map(|_| rng.random_bool(density)))
}

/// The minimum of the demanded rows, by demand density, for a canonical and a dictionary batch.
#[divan::bench(args = DENSITIES, consts = [false, true])]
fn min_demanded_by_density<const DICT: bool>(bencher: Bencher, density: f64) {
    let batch = density_batch(DICT);
    let demand = demand_with_density(density);
    let aggregate_fn = aggregate("min");
    bencher
        .with_inputs(|| {
            (
                aggregate_fn.accumulator(batch.dtype()).unwrap(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(mut acc, mut ctx)| {
            acc.accumulate_demanded(&batch, &demand, &mut ctx).unwrap();
            acc.flush().unwrap()
        });
}

fn density_batch(dict: bool) -> ArrayRef {
    if dict {
        let mut rng = StdRng::seed_from_u64(4);
        let codes: Buffer<u8> = (0..N).map(|_| rng.random_range(0..64)).collect();
        let values: Buffer<i32> = (0..64).map(|v| v * 7 - 100).collect();
        vortex_array::arrays::DictArray::new(codes.into_array(), values.into_array()).into_array()
    } else {
        PrimitiveArray::new(values(), Validity::NonNullable).into_array()
    }
}
