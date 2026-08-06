// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Take on a canonical sparse union of integers, strings, and lists.
//!
//! Take gathers every sparse child, not only the selected one, so the cost is the sum over all
//! three variants. The encodings differ on purpose: a primitive gather is close to a memcpy, a
//! `VarBinView` gather copies views and leaves the data buffers alone, and a `ListView` gather
//! rebuilds offsets and sizes over a shared element buffer. Nullable indices add a fill-null pass
//! per child on top of that.

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

/// Held low so that both cases stay inside the sub-millisecond microbenchmark budget. Per-child
/// setup dominates the gather at this size, and that setup is paid once per variant.
const TAKE_SIZE: usize = 128;

fn integer_child(rng: &mut StdRng) -> ArrayRef {
    (0..ARRAY_SIZE)
        .map(|_| rng.random::<i64>())
        .collect::<Buffer<i64>>()
        .into_array()
}

/// A `utf8` child mixing strings short enough to inline into a `VarBinView` view with longer ones
/// that have to point into the data buffer.
///
/// A `VarBinView` view inlines up to 12 bytes. A child of uniformly short strings would gather
/// without ever reading the data buffer, which is not what a real string column costs.
fn string_child() -> ArrayRef {
    VarBinViewArray::from_iter_str((0..ARRAY_SIZE).map(|i| match i % 4 {
        0 => format!("s{i}"),
        _ => format!("a considerably longer string value, number {i}"),
    }))
    .into_array()
}

/// A `list<i64>` child of variable-length lists over one shared element buffer.
///
/// The lengths vary on purpose. Equal-length lists are what `FixedSizeList` is for, and they would
/// not exercise the offsets and sizes that a `ListView` gather has to rebuild.
fn list_child(rng: &mut StdRng) -> ArrayRef {
    let sizes: Buffer<i32> = (0..ARRAY_SIZE).map(|_| rng.random_range(0..16)).collect();
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
        .map(|_| rng.random::<i64>())
        .collect::<Buffer<i64>>()
        .into_array();

    ListViewArray::new(
        elements,
        offsets.into_array(),
        sizes.into_array(),
        Validity::NonNullable,
    )
    .into_array()
}

/// A sparse union whose rows cycle through an integer, a string, and a list.
fn union_array(rng: &mut StdRng) -> ArrayRef {
    let children = vec![integer_child(rng), string_child(), list_child(rng)];
    let variants = UnionVariants::new(
        ["ints", "strings", "lists"].into(),
        children.iter().map(|child| child.dtype().clone()).collect(),
    )
    .unwrap();

    let type_ids = PrimitiveArray::from_iter(
        (0..ARRAY_SIZE).map(|i| u8::try_from(i % children.len()).unwrap()),
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

#[divan::bench]
fn take_union(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(0);
    let array = union_array(&mut rng);

    let indices = (0..TAKE_SIZE)
        .map(|_| rng.random_range(0..ARRAY_SIZE) as u64)
        .collect::<Buffer<u64>>()
        .into_array();

    bench_take(bencher, array, indices);
}

#[divan::bench]
fn take_union_nullable_indices(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(0);
    let array = union_array(&mut rng);

    // Every tenth index is null, which produces an outer union null.
    let indices = PrimitiveArray::from_option_iter(
        (0..TAKE_SIZE).map(|i| (i % 10 != 0).then(|| rng.random_range(0..ARRAY_SIZE) as u64)),
    )
    .into_array();

    bench_take(bencher, array, indices);
}
