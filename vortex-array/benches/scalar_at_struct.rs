// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::distr::Uniform;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;

fn main() {
    divan::main();
}

const ARRAY_SIZE: usize = 100_000;
const NUM_ACCESSES: usize = 1000;

#[divan::bench]
fn scalar_at_struct_simple(bencher: Bencher) {
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
    .unwrap();
    let struct_array: ArrayRef = struct_array.into_array();

    let indices: Vec<usize> = (0..NUM_ACCESSES)
        .map(|_| rng.random_range(0..ARRAY_SIZE))
        .collect();

    bencher
        .with_inputs(|| (&struct_array, &indices))
        .bench_refs(|(array, indices)| {
            for &idx in indices.iter() {
                divan::black_box(array.scalar_at(idx).unwrap());
            }
        });
}

#[divan::bench]
fn scalar_at_struct_wide(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(0);
    let range = Uniform::new(0i64, 100_000_000).unwrap();

    // Create a struct with many fields (8 fields)
    let fields: Vec<_> = (0..8)
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

    let struct_array: ArrayRef =
        StructArray::try_new(field_names, fields, ARRAY_SIZE, Validity::NonNullable)
            .unwrap()
            .into_array();

    let indices: Vec<usize> = (0..NUM_ACCESSES)
        .map(|_| rng.random_range(0..ARRAY_SIZE))
        .collect();

    bencher
        .with_inputs(|| (&struct_array, &indices))
        .bench_refs(|(array, indices)| {
            for &idx in indices.iter() {
                divan::black_box(array.scalar_at(idx).unwrap());
            }
        });
}
