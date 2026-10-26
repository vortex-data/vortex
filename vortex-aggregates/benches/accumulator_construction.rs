// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

//! Accumulator creation for this crate's aggregate functions.

use std::hint::black_box;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_aggregates::count::Count;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

const N: usize = 1024;

fn struct8_dtype() -> DType {
    DType::Struct(
        StructFields::new(
            FieldNames::from(["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]),
            vec![DType::Primitive(PType::I64, Nullability::NonNullable); 8],
        ),
        Nullability::NonNullable,
    )
}

#[divan::bench]
fn accumulator_count_struct8(bencher: Bencher) {
    let dtype = struct8_dtype();
    let aggregate_fn = Count.bind(NumericalAggregateOpts::default());
    bencher.counter(ItemsCount::new(N)).bench_local(|| {
        for _ in 0..N {
            black_box(aggregate_fn.accumulator(black_box(&dtype)).unwrap());
        }
    });
}
