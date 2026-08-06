// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Take on a canonical sparse union, swept over the variant count.
//!
//! Take gathers every sparse child, so a wider union costs more. The children mix encodings on
//! purpose, because each one carries its own gather cost and a union of identical primitives
//! understates the total. At this index count the per-child setup dominates the gather itself,
//! and that setup is exactly what the variant count multiplies.

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
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::UnionArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::UnionVariants;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

const ARRAY_SIZE: usize = 100_000;

/// Held low so the widest variant case stays inside the sub-millisecond microbenchmark budget.
const TAKE_SIZE: usize = 128;

/// One entry per pool variant, starting from a single-variant baseline that isolates the type IDs
/// gather from the child gathers.
const VARIANT_COUNTS: [usize; 3] = [1, 2, 3];

/// A `List<i32>` of `ARRAY_SIZE` short lists over one shared element buffer.
fn list_child(rng: &mut StdRng) -> ArrayRef {
    const MAX_LIST_LEN: i32 = 8;

    let sizes: Buffer<i32> = (0..ARRAY_SIZE)
        .map(|_| rng.random_range(0..MAX_LIST_LEN))
        .collect();
    let offsets: Buffer<i32> = sizes
        .iter()
        .scan(0i32, |offset, size| {
            let start = *offset;
            *offset += size;
            Some(start)
        })
        .collect();

    let total = offsets.last().unwrap() + sizes.last().unwrap();
    let elements = (0..total)
        .map(|_| rng.random::<i32>())
        .collect::<Buffer<i32>>()
        .into_array();

    ListViewArray::new(
        elements,
        offsets.into_array(),
        sizes.into_array(),
        Validity::NonNullable,
    )
    .into_array()
}

/// The `index`th variant of the pool, as a name and an `ARRAY_SIZE`-row child.
///
/// The pool runs `i64`, `list<i32>`, and `utf8`, so widening the union adds a child that gathers
/// differently rather than one more primitive gather.
fn variant_child(index: usize, rng: &mut StdRng) -> (String, ArrayRef) {
    let child = match index {
        0 => (0..ARRAY_SIZE)
            .map(|_| rng.random::<i64>())
            .collect::<Buffer<i64>>()
            .into_array(),
        1 => list_child(rng),
        _ => VarBinViewArray::from_iter_str((0..ARRAY_SIZE).map(|i| format!("value-{i}")))
            .into_array(),
    };

    (format!("v{index}"), child)
}

/// A sparse union over the first `variant_count` pool entries, with type IDs cycling through them.
fn union_array(variant_count: usize, rng: &mut StdRng) -> ArrayRef {
    let (names, children): (Vec<String>, Vec<ArrayRef>) = (0..variant_count)
        .map(|index| variant_child(index, rng))
        .unzip();

    let variants = UnionVariants::new(
        names.into_iter().collect(),
        children.iter().map(|child| child.dtype().clone()).collect(),
    )
    .unwrap();

    let type_ids = PrimitiveArray::from_iter(
        (0..ARRAY_SIZE).map(|i| u8::try_from(i % variant_count).unwrap()),
    );

    UnionArray::new(type_ids.into_array(), variants, children).into_array()
}

/// Take `indices` and execute the result, so that the child gathers actually run.
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

#[divan::bench(args = VARIANT_COUNTS)]
fn take_union(bencher: Bencher, variant_count: usize) {
    let mut rng = StdRng::seed_from_u64(0);
    let array = union_array(variant_count, &mut rng);

    let indices = (0..TAKE_SIZE)
        .map(|_| rng.random_range(0..ARRAY_SIZE) as u64)
        .collect::<Buffer<u64>>()
        .into_array();

    bench_take(bencher, array, indices);
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

    bench_take(bencher, array, indices);
}
