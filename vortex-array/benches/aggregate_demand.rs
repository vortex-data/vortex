// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Accumulating a nested field with the demand of its nullable parent, compared with accumulating the
//! logical field, whose validity already excludes the rows the parent nulls out.

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
use vortex_array::aggregate_fn::demand::list_elements_demand;
use vortex_array::aggregate_fn::fns::null_count::NullCount;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::arrays::ListArray;
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

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

/// An aggregate that skips nulls and one that counts them.
const AGGREGATES: &[&str] = &["sum", "null_count"];

fn aggregate(name: &str) -> AggregateFnRef {
    match name {
        "sum" => Sum.bind(NumericalAggregateOpts::default()),
        "null_count" => NullCount.bind(EmptyOptions),
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

/// Nine in ten rows, chosen at random, as the struct validity of a nested field would be.
fn demand() -> Mask {
    let mut rng = StdRng::seed_from_u64(2);
    Mask::from_iter((0..N).map(|_| rng.random_bool(0.9)))
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
/// Undemanded rows are null, so the demanded field gives the same result as the logical field for
/// every aggregate. The null count counts the field's own nulls and the rows the struct nulls out.
fn check_results() {
    let (field, demand) = (field(), demand());
    for name in AGGREGATES {
        let aggregate_fn = aggregate(name);
        assert_eq!(
            run(&aggregate_fn, &field, Some(&demand)),
            run(&aggregate_fn, &masked_field(), None),
            "{name}"
        );
    }

    let expected_nulls = (!&(&field_validity() & &demand)).true_count() as u64;
    let null_count = run(&aggregate("null_count"), &field, Some(&demand));
    assert_eq!(
        null_count.as_primitive().typed_value::<u64>(),
        Some(expected_nulls)
    );
}

/// Accumulate the field with the demand of its parent.
///
/// Both this and [`nullable`] build their input outside the timed region, a fresh batch per
/// iteration so that no statistic cached by a previous iteration is reused, and run the same
/// accumulator over the same logical values. The demanded side also pays for merging the demand
/// into the field's validity, which the nullable side receives already merged.
#[divan::bench(args = AGGREGATES)]
fn demanded(bencher: Bencher, name: &str) {
    let aggregate_fn = aggregate(name);
    let demand = demand();
    bencher
        .with_inputs(|| {
            let batch = field();
            (
                aggregate_fn.accumulator(batch.dtype()).unwrap(),
                batch,
                SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(mut acc, batch, mut ctx)| {
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

const VALID_DENSITIES: &[f64] = &[0.5, 0.99];

/// A list column of `N` lists of up to eight elements, whose given fraction of lists is valid.
fn lists_with_validity(density: f64) -> ListArray {
    let mut rng = StdRng::seed_from_u64(7);
    let mut offsets = Vec::with_capacity(N + 1);
    offsets.push(0u32);
    for _ in 0..N {
        let last = *offsets.last().unwrap();
        offsets.push(last + rng.random_range(0..8u32));
    }
    let elements: Buffer<u32> = (0..*offsets.last().unwrap()).collect();
    let validity = Mask::from_iter((0..N).map(|_| rng.random_bool(density)));
    ListArray::try_new(
        elements.into_array(),
        Buffer::from(offsets).into_array(),
        Validity::from_mask(validity, Nullability::Nullable),
        &mut SESSION.create_execution_ctx(),
    )
    .unwrap()
}

/// Derive the demand on the elements of a list column from the lists' validity.
#[divan::bench(args = VALID_DENSITIES)]
fn list_elements_demand_by_validity(bencher: Bencher, density: f64) {
    let lists = lists_with_validity(density);
    let demand = Mask::new_true(N);
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_values(|mut ctx| list_elements_demand(&lists, &demand, &mut ctx).unwrap());
}
