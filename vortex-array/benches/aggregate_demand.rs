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
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_mask::Mask;
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

/// Half the rows, chosen at random, as the struct validity of a nested field would be.
fn demand() -> Mask {
    let mut rng = StdRng::seed_from_u64(2);
    Mask::from_iter((0..N).map(|_| rng.random_bool(0.5)))
}

/// Accumulate the demanded rows of a non-nullable batch.
#[divan::bench(args = AGGREGATES)]
fn demanded(bencher: Bencher, name: &str) {
    let aggregate_fn = aggregate(name);
    let batch: ArrayRef = PrimitiveArray::new(values(), Validity::NonNullable).into_array();
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

/// Accumulate a nullable batch whose validity is the demand.
#[divan::bench(args = AGGREGATES)]
fn nullable(bencher: Bencher, name: &str) {
    let aggregate_fn = aggregate(name);
    let validity = Validity::from_mask(demand(), vortex_array::dtype::Nullability::Nullable);
    let values = values();
    bencher
        .with_inputs(|| {
            // A fresh batch per iteration, so no statistic cached by a previous one is reused.
            let batch = PrimitiveArray::new(values.clone(), validity.clone()).into_array();
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

/// Filter the batch down to its demanded rows, then accumulate it.
#[divan::bench(args = AGGREGATES)]
fn filtered(bencher: Bencher, name: &str) {
    let aggregate_fn = aggregate(name);
    let batch: ArrayRef = PrimitiveArray::new(values(), Validity::NonNullable).into_array();
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
