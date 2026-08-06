// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Take on a canonical sparse union, swept over the variant count.
//!
//! Take gathers every sparse child, so cost is linear in the variant count. The nullable-indices
//! case also pays for the fill-null pass.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::UnionArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::UnionVariants;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

const ARRAY_SIZE: usize = 100_000;

/// Held low so the widest variant case stays inside the sub-millisecond microbenchmark budget.
const TAKE_SIZE: usize = 256;

const VARIANT_COUNTS: [usize; 3] = [2, 4, 8];

/// A sparse union of `variant_count` `i64` variants whose type IDs cycle through every variant.
fn union_array(variant_count: usize, rng: &mut StdRng) -> ArrayRef {
    let names: FieldNames = (0..variant_count).map(|i| format!("v{i}")).collect();
    let dtypes = vec![DType::Primitive(PType::I64, Nullability::NonNullable); variant_count];
    let variants = UnionVariants::new(names, dtypes).unwrap();

    let type_ids = PrimitiveArray::from_iter(
        (0..ARRAY_SIZE).map(|i| u8::try_from(i % variant_count).unwrap()),
    );
    let children = (0..variant_count)
        .map(|_| {
            (0..ARRAY_SIZE)
                .map(|_| rng.random::<i64>())
                .collect::<Buffer<i64>>()
                .into_array()
        })
        .collect::<Vec<_>>();

    UnionArray::new(type_ids.into_array(), variants, children).into_array()
}

#[divan::bench(args = VARIANT_COUNTS)]
fn take_union(bencher: Bencher, variant_count: usize) {
    let mut rng = StdRng::seed_from_u64(0);
    let array = union_array(variant_count, &mut rng);

    let indices = (0..TAKE_SIZE)
        .map(|_| rng.random_range(0..ARRAY_SIZE) as u64)
        .collect::<Buffer<u64>>()
        .into_array();

    bencher
        .with_inputs(|| (&array, &indices, SESSION.create_execution_ctx()))
        .bench_refs(|(array, indices, ctx)| {
            array
                .take((*indices).clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
        });
}

#[divan::bench(args = VARIANT_COUNTS)]
fn take_union_nullable_indices(bencher: Bencher, variant_count: usize) {
    let mut rng = StdRng::seed_from_u64(0);
    let array = union_array(variant_count, &mut rng);

    // Every tenth index is null, which produces an outer union null.
    let indices = PrimitiveArray::from_option_iter(
        (0..TAKE_SIZE).map(|i| (i % 10 != 0).then(|| rng.random_range(0..ARRAY_SIZE) as u64)),
    )
    .into_array();

    bencher
        .with_inputs(|| (&array, &indices, SESSION.create_execution_ctx()))
        .bench_refs(|(array, indices, ctx)| {
            array
                .take((*indices).clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
        });
}
