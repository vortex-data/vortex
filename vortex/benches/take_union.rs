// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Take on a canonical sparse union of integers, strings, and lists.
//!
//! Take gathers every sparse child, not only the one each row selects, so the cost is the sum over
//! all three variants. That is the price of the sparse layout, and it is also what makes the layout
//! affordable: Vortex does not require the children of a canonical array to be canonical, so the
//! inactive slots of a child can be encoded away.
//!
//! Rows are skewed: 98% are integers and the strings and lists are 1% each. That skew is the case
//! the layout is designed for, and it is what makes a rare variant worth encoding sparsely.
//!
//! The `dense_children` cases hold every child materialized at the union's length, which is what a
//! writer produces before compression. The `compressed_children` cases keep the dominant integer
//! child canonical and store each rare child as a `SparseArray` holding only the rows its variant
//! selects, which is what a compressor produces.
//!
//! The `compressed_children` cases run over the sub-millisecond budget the other microbenchmarks
//! hold to. What they measure is cache-missing binary searches in `Patches::take`, which only
//! appears once a child's patch indices outgrow the cache. Shrinking the array to fit the budget
//! closes the gap between the two shapes to noise and measures nothing.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::RecursiveCanonical;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ListViewArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::UnionArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::dtype::UnionVariants;
use vortex::array::scalar::Scalar;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::encodings::sparse::Sparse;
use vortex_session::VortexSession;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::default);

const ARRAY_SIZE: usize = 100_000;

const TAKE_SIZE: usize = 128;

const VARIANT_NAMES: [&str; 3] = ["ints", "strings", "lists"];

/// The variant that almost every row selects.
const DOMINANT_VARIANT: usize = 0;

/// The variant that `row` selects. Integers dominate, and strings and lists are rare.
fn type_id_for(row: usize) -> u8 {
    match row % 100 {
        98 => 1,
        99 => 2,
        _ => 0,
    }
}

fn integer_child(len: usize, rng: &mut StdRng) -> ArrayRef {
    (0..len)
        .map(|_| rng.random::<i64>())
        .collect::<Buffer<i64>>()
        .into_array()
}

/// A `utf8` child mixing strings short enough to inline into a `VarBinView` view with longer ones
/// that have to point into the data buffer.
///
/// A `VarBinView` view inlines up to 12 bytes. A child of uniformly short strings would gather
/// without ever reading the data buffer, which is not what a real string column costs.
fn string_child(len: usize) -> ArrayRef {
    VarBinViewArray::from_iter_str((0..len).map(|i| match i % 4 {
        0 => format!("s{i}"),
        _ => format!("a considerably longer string value, number {i}"),
    }))
    .into_array()
}

/// A `list<i64>` child of variable-length lists over one shared element buffer.
///
/// The lengths vary on purpose. Equal-length lists are what `FixedSizeList` is for, and they would
/// not exercise the offsets and sizes that a `ListView` gather has to rebuild.
fn list_child(len: usize, rng: &mut StdRng) -> ArrayRef {
    let sizes: Buffer<i32> = (0..len).map(|_| rng.random_range(0..16)).collect();
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

fn variant_child(variant: usize, len: usize, rng: &mut StdRng) -> ArrayRef {
    match variant {
        0 => integer_child(len, rng),
        1 => string_child(len),
        _ => list_child(len, rng),
    }
}

/// The rows whose type ID selects `variant`.
fn variant_indices(variant: usize) -> Buffer<u64> {
    (0..ARRAY_SIZE)
        .filter(|&row| usize::from(type_id_for(row)) == variant)
        .map(|row| row as u64)
        .collect()
}

fn union_array(children: Vec<ArrayRef>) -> ArrayRef {
    let variants = UnionVariants::new(
        VARIANT_NAMES.into(),
        children.iter().map(|child| child.dtype().clone()).collect(),
    )
    .unwrap();

    let type_ids = PrimitiveArray::from_iter((0..ARRAY_SIZE).map(type_id_for));

    UnionArray::new(type_ids.into_array(), variants, children).into_array()
}

/// A union whose children are each materialized at the union's full length.
fn dense_children_union(rng: &mut StdRng) -> ArrayRef {
    union_array(
        (0..VARIANT_NAMES.len())
            .map(|variant| variant_child(variant, ARRAY_SIZE, rng))
            .collect(),
    )
}

/// A union whose rare children store only the rows their variant selects, leaving the inactive
/// slots to the sparse fill value.
///
/// The dominant integer child stays canonical. At 98% density a sparse child would store a patch
/// index for nearly every row and buy nothing.
fn compressed_children_union(rng: &mut StdRng) -> ArrayRef {
    union_array(
        (0..VARIANT_NAMES.len())
            .map(|variant| {
                if variant == DOMINANT_VARIANT {
                    return variant_child(variant, ARRAY_SIZE, rng);
                }

                let indices = variant_indices(variant);
                let values = variant_child(variant, indices.len(), rng);
                let fill = Scalar::default_value(values.dtype());

                Sparse::try_new(indices.into_array(), values, ARRAY_SIZE, fill)
                    .unwrap()
                    .into_array()
            })
            .collect(),
    )
}

fn random_indices(rng: &mut StdRng) -> ArrayRef {
    (0..TAKE_SIZE)
        .map(|_| rng.random_range(0..ARRAY_SIZE) as u64)
        .collect::<Buffer<u64>>()
        .into_array()
}

/// Random indices where every tenth one is null, which produces an outer union null.
fn random_nullable_indices(rng: &mut StdRng) -> ArrayRef {
    PrimitiveArray::from_option_iter(
        (0..TAKE_SIZE).map(|i| (i % 10 != 0).then(|| rng.random_range(0..ARRAY_SIZE) as u64)),
    )
    .into_array()
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
fn take_union_dense_children(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(0);
    let array = dense_children_union(&mut rng);
    let indices = random_indices(&mut rng);

    bench_take(bencher, array, indices);
}

#[divan::bench]
fn take_union_compressed_children(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(0);
    let array = compressed_children_union(&mut rng);
    let indices = random_indices(&mut rng);

    bench_take(bencher, array, indices);
}

#[divan::bench]
fn take_union_dense_children_nullable_indices(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(0);
    let array = dense_children_union(&mut rng);
    let indices = random_nullable_indices(&mut rng);

    bench_take(bencher, array, indices);
}

#[divan::bench]
fn take_union_compressed_children_nullable_indices(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(0);
    let array = compressed_children_union(&mut rng);
    let indices = random_nullable_indices(&mut rng);

    bench_take(bencher, array, indices);
}
