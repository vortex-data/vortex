// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::distr::Uniform;
use rand::rngs::StdRng;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;

fn main() {
    divan::main();
}

const ARRAY_SIZE: usize = 100_000;
const TAKE_SIZE: usize = 1000;

#[divan::bench]
fn take_struct_simple(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(0);
    let range = Uniform::new(0i64, 100_000_000).unwrap();

    // Create single field for the struct
    let field = (0..ARRAY_SIZE)
        .map(|_| rng.sample(range))
        .collect::<Buffer<i64>>()
        .into_array();

    let struct_array = StructArray::try_new(
        FieldNames::from(["value"]),
        vec![field],
        ARRAY_SIZE,
        Validity::NonNullable,
    )
    .unwrap()
    .into_array();

    let indices: Buffer<u64> = (0..TAKE_SIZE)
        .map(|_| rng.random_range(0..ARRAY_SIZE) as u64)
        .collect();
    let indices_array = indices.into_array();

    bencher
        .with_inputs(|| {
            (
                &struct_array,
                &indices_array,
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, indices, ctx)| {
            array
                .take((*indices).clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
        });
}

#[divan::bench(args = [8])]
fn take_struct_wide(bencher: Bencher, width: usize) {
    let mut rng = StdRng::seed_from_u64(0);
    let range = Uniform::new(0i64, 100_000_000).unwrap();

    let fields: Vec<_> = (0..width)
        .map(|_| {
            (0..ARRAY_SIZE)
                .map(|_| rng.sample(range))
                .collect::<Buffer<i64>>()
                .into_array()
        })
        .collect();

    let field_names = FieldNames::from([
        "field1", "field2", "field3", "field4", "field5", "field6", "field7", "field8",
    ]);

    let struct_array = StructArray::try_new(field_names, fields, ARRAY_SIZE, Validity::NonNullable)
        .unwrap()
        .into_array();

    let indices: Buffer<u64> = (0..TAKE_SIZE)
        .map(|_| rng.random_range(0..ARRAY_SIZE) as u64)
        .collect();
    let indices_array = indices.into_array();

    bencher
        .with_inputs(|| {
            (
                &struct_array,
                &indices_array,
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, indices, ctx)| {
            array
                .take((*indices).clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
        });
}

#[divan::bench]
fn take_struct_sequential_indices(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(0);
    let range = Uniform::new(0i64, 100_000_000).unwrap();

    // Create single field for the struct
    let field = (0..ARRAY_SIZE)
        .map(|_| rng.sample(range))
        .collect::<Buffer<i64>>()
        .into_array();

    let struct_array = StructArray::try_new(
        FieldNames::from(["value"]),
        vec![field],
        ARRAY_SIZE,
        Validity::NonNullable,
    )
    .unwrap()
    .into_array();

    // Sequential indices for better cache performance
    let indices: Buffer<u64> = (0..TAKE_SIZE as u64).collect();
    let indices_array = indices.into_array();

    bencher
        .with_inputs(|| {
            (
                &struct_array,
                &indices_array,
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, indices, ctx)| {
            array
                .take((*indices).clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
        });
}
