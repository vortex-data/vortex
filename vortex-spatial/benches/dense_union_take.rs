// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::UnionArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::UnionVariants;
use vortex_session::VortexSession;
use vortex_spatial::test_harness::DenseUnion;
use vortex_spatial::test_harness::spatial_session;

const LEN: usize = 65_536;
const N_VARIANTS: usize = 28;
const TAKE_LEN: usize = 4_096;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(spatial_session);

fn variants() -> UnionVariants {
    let names = FieldNames::from_iter((0..N_VARIANTS).map(|index| format!("variant_{index}")));
    let dtypes = vec![DType::Primitive(PType::I32, Nullability::NonNullable); N_VARIANTS];
    let type_ids = (1..=N_VARIANTS).map(|type_id| type_id as u8).collect();
    UnionVariants::try_new(names, dtypes, type_ids).unwrap()
}

fn selectors() -> (ArrayRef, ArrayRef, Vec<usize>) {
    let mut child_lengths = vec![0usize; N_VARIANTS];
    let mut type_ids = Vec::with_capacity(LEN);
    let mut offsets = Vec::with_capacity(LEN);
    for row in 0..LEN {
        let child_index = row % N_VARIANTS;
        type_ids.push((child_index + 1) as u8);
        offsets.push(child_lengths[child_index] as i32);
        child_lengths[child_index] += 1;
    }
    (
        PrimitiveArray::from_iter(type_ids).into_array(),
        PrimitiveArray::from_iter(offsets).into_array(),
        child_lengths,
    )
}

fn dense_union() -> ArrayRef {
    let (type_ids, offsets, child_lengths) = selectors();
    let children = child_lengths
        .into_iter()
        .map(|len| PrimitiveArray::from_iter(0..len as i32).into_array())
        .collect::<Vec<_>>();
    DenseUnion::try_new(type_ids, offsets, variants(), children)
        .unwrap()
        .into_array()
}

fn sparse_union() -> ArrayRef {
    let (type_ids, ..) = selectors();
    let children = (0..N_VARIANTS)
        .map(|child_index| {
            PrimitiveArray::from_iter((0..LEN).map(move |row| {
                if row % N_VARIANTS == child_index {
                    (row / N_VARIANTS) as i32
                } else {
                    0
                }
            }))
            .into_array()
        })
        .collect::<Vec<_>>();
    UnionArray::try_new(type_ids, variants(), children)
        .unwrap()
        .into_array()
}

fn indices() -> ArrayRef {
    PrimitiveArray::from_iter((0..TAKE_LEN).rev().map(|index| index as u32)).into_array()
}

fn bench_take(bencher: Bencher, array: ArrayRef, indices: ArrayRef) {
    bencher
        .with_inputs(|| (&array, &indices, SESSION.create_execution_ctx()))
        .bench_refs(|(array, indices, ctx)| {
            array
                .take((*indices).clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
        });
}

#[divan::bench]
fn dense_take(bencher: Bencher) {
    bench_take(bencher, dense_union(), indices());
}

#[divan::bench]
fn sparse_take(bencher: Bencher) {
    bench_take(bencher, sparse_union(), indices());
}
