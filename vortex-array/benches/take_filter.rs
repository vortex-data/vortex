// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for taking from a lazy [`FilterArray`].
//!
//! Parameterized over:
//! - Number of indices to take
//! - Number of rows retained by the filter
//! - Filter mask layout (single contiguous slice vs random positions)
//! - Take index layout (sequential vs random ranks)
//! - Nullable vs non-null take indices

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::FieldNames;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_mask::Mask;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

const ARRAY_LEN: usize = 100_000;
const FILTERED_LENS: &[usize] = &[16384, 65536];
const NUM_INDICES: &[usize] = &[1_000];
const LARGE_TAKE_CASES: &[(usize, usize)] = &[(10_000, 100_000), (50_000, 100_000)];
const MASK_SEED: u64 = 42;
const INDEX_SEED: u64 = 43;
const LIST_SIZE: usize = 4;
const NULL_INDEX_INTERVAL: usize = 8;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

fn primitive_array() -> ArrayRef {
    PrimitiveArray::from_iter(0..ARRAY_LEN as u32).into_array()
}

fn struct_array() -> ArrayRef {
    StructArray::try_new(
        FieldNames::from(["id", "value"]),
        vec![
            PrimitiveArray::from_iter(0..ARRAY_LEN as u32).into_array(),
            PrimitiveArray::from_iter((0..ARRAY_LEN).map(|idx| (idx as u64) * 3)).into_array(),
        ],
        ARRAY_LEN,
        Validity::NonNullable,
    )
    .unwrap()
    .into_array()
}

fn list_array() -> ArrayRef {
    let elements = PrimitiveArray::from_iter(0..(ARRAY_LEN * LIST_SIZE) as u32).into_array();
    let offsets =
        Buffer::from_iter((0..=ARRAY_LEN).map(|idx| (idx * LIST_SIZE) as u32)).into_array();

    ListArray::try_new(elements, offsets, Validity::NonNullable)
        .unwrap()
        .into_array()
}

fn string_array() -> ArrayRef {
    let strings: Vec<String> = (0..ARRAY_LEN)
        .map(|idx| {
            if idx % 4 == 0 {
                format!("long-string-value-{idx:06}")
            } else {
                format!("s{idx}")
            }
        })
        .collect();

    VarBinViewArray::from_iter_str(strings.iter().map(String::as_str)).into_array()
}

fn slice_mask(filtered_len: usize) -> Mask {
    let start = (ARRAY_LEN - filtered_len) / 2;
    Mask::from_buffer(BitBuffer::from_iter(
        (0..ARRAY_LEN).map(|idx| (start..start + filtered_len).contains(&idx)),
    ))
}

fn random_mask(filtered_len: usize) -> Mask {
    let mut indices: Vec<usize> = (0..ARRAY_LEN).collect();
    indices.shuffle(&mut StdRng::seed_from_u64(MASK_SEED));
    indices.truncate(filtered_len);

    let mut buffer = BitBufferMut::new_unset(ARRAY_LEN);
    for idx in indices {
        buffer.set(idx);
    }

    Mask::from_buffer(buffer.freeze())
}

fn sequential_indices(num_indices: usize) -> ArrayRef {
    Buffer::from_iter(0..num_indices as u64).into_array()
}

fn random_indices(num_indices: usize, filtered_len: usize) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(INDEX_SEED);
    Buffer::from_iter((0..num_indices).map(|_| rng.random_range(0..filtered_len as u64)))
        .into_array()
}

fn nullable_random_indices(num_indices: usize, filtered_len: usize) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(INDEX_SEED);
    PrimitiveArray::from_option_iter((0..num_indices).map(|idx| {
        if idx % NULL_INDEX_INTERVAL == 0 {
            None
        } else {
            Some(rng.random_range(0..filtered_len as u64))
        }
    }))
    .into_array()
}

fn bench_take_filter(bencher: Bencher, array: ArrayRef, indices: ArrayRef) {
    bencher
        .with_inputs(|| (&array, &indices, SESSION.create_execution_ctx()))
        .bench_refs(|(array, indices, ctx)| {
            array
                .take(indices.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
                .unwrap()
        });
}

#[divan::bench(args = NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_primitive_slice_mask_sequential_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = primitive_array().filter(slice_mask(FILTERED_LEN)).unwrap();
    let indices = sequential_indices(num_indices);

    bench_take_filter(bencher, array, indices);
}

#[divan::bench(args = NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_primitive_slice_mask_random_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = primitive_array().filter(slice_mask(FILTERED_LEN)).unwrap();
    let indices = random_indices(num_indices, FILTERED_LEN);

    bench_take_filter(bencher, array, indices);
}

#[divan::bench(args = NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_primitive_random_mask_sequential_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = primitive_array().filter(random_mask(FILTERED_LEN)).unwrap();
    let indices = sequential_indices(num_indices);

    bench_take_filter(bencher, array, indices);
}

#[divan::bench(args = NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_primitive_random_mask_random_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = primitive_array().filter(random_mask(FILTERED_LEN)).unwrap();
    let indices = random_indices(num_indices, FILTERED_LEN);

    bench_take_filter(bencher, array, indices);
}

#[divan::bench(args = NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_struct_slice_mask_sequential_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = struct_array().filter(slice_mask(FILTERED_LEN)).unwrap();
    let indices = sequential_indices(num_indices);

    bench_take_filter(bencher, array, indices);
}

#[divan::bench(args = NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_struct_slice_mask_random_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = struct_array().filter(slice_mask(FILTERED_LEN)).unwrap();
    let indices = random_indices(num_indices, FILTERED_LEN);

    bench_take_filter(bencher, array, indices);
}

#[divan::bench(args = NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_struct_random_mask_sequential_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = struct_array().filter(random_mask(FILTERED_LEN)).unwrap();
    let indices = sequential_indices(num_indices);

    bench_take_filter(bencher, array, indices);
}

#[divan::bench(args = NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_struct_random_mask_random_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = struct_array().filter(random_mask(FILTERED_LEN)).unwrap();
    let indices = random_indices(num_indices, FILTERED_LEN);

    bench_take_filter(bencher, array, indices);
}

#[divan::bench(args = NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_list_slice_mask_sequential_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = list_array().filter(slice_mask(FILTERED_LEN)).unwrap();
    let indices = sequential_indices(num_indices);

    bench_take_filter(bencher, array, indices);
}

#[divan::bench(args = NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_list_slice_mask_random_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = list_array().filter(slice_mask(FILTERED_LEN)).unwrap();
    let indices = random_indices(num_indices, FILTERED_LEN);

    bench_take_filter(bencher, array, indices);
}

#[divan::bench(args = NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_list_random_mask_sequential_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = list_array().filter(random_mask(FILTERED_LEN)).unwrap();
    let indices = sequential_indices(num_indices);

    bench_take_filter(bencher, array, indices);
}

#[divan::bench(args = NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_list_random_mask_random_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = list_array().filter(random_mask(FILTERED_LEN)).unwrap();
    let indices = random_indices(num_indices, FILTERED_LEN);

    bench_take_filter(bencher, array, indices);
}

#[divan::bench(args = NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_string_slice_mask_sequential_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = string_array().filter(slice_mask(FILTERED_LEN)).unwrap();
    let indices = sequential_indices(num_indices);

    bench_take_filter(bencher, array, indices);
}

#[divan::bench(args = NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_string_slice_mask_random_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = string_array().filter(slice_mask(FILTERED_LEN)).unwrap();
    let indices = random_indices(num_indices, FILTERED_LEN);

    bench_take_filter(bencher, array, indices);
}

#[divan::bench(args = NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_string_random_mask_sequential_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = string_array().filter(random_mask(FILTERED_LEN)).unwrap();
    let indices = sequential_indices(num_indices);

    bench_take_filter(bencher, array, indices);
}

#[divan::bench(args = NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_string_random_mask_random_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = string_array().filter(random_mask(FILTERED_LEN)).unwrap();
    let indices = random_indices(num_indices, FILTERED_LEN);

    bench_take_filter(bencher, array, indices);
}

#[divan::bench(args = NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_primitive_nullable_slice_mask_random_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = primitive_array().filter(slice_mask(FILTERED_LEN)).unwrap();
    let indices = nullable_random_indices(num_indices, FILTERED_LEN);

    bench_take_filter(bencher, array, indices);
}

#[divan::bench(args = NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_primitive_nullable_random_mask_random_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = primitive_array().filter(random_mask(FILTERED_LEN)).unwrap();
    let indices = nullable_random_indices(num_indices, FILTERED_LEN);

    bench_take_filter(bencher, array, indices);
}

#[divan::bench(args = NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_struct_nullable_random_mask_random_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = struct_array().filter(random_mask(FILTERED_LEN)).unwrap();
    let indices = nullable_random_indices(num_indices, FILTERED_LEN);

    bench_take_filter(bencher, array, indices);
}

#[divan::bench(args = NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_list_nullable_random_mask_random_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = list_array().filter(random_mask(FILTERED_LEN)).unwrap();
    let indices = nullable_random_indices(num_indices, FILTERED_LEN);

    bench_take_filter(bencher, array, indices);
}

#[divan::bench(args = NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_string_nullable_random_mask_random_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = string_array().filter(random_mask(FILTERED_LEN)).unwrap();
    let indices = nullable_random_indices(num_indices, FILTERED_LEN);

    bench_take_filter(bencher, array, indices);
}

#[divan::bench(args = LARGE_TAKE_CASES)]
fn take_filter_primitive_large_random_mask_random_indices(
    bencher: Bencher,
    (filtered_len, num_indices): (usize, usize),
) {
    let array = primitive_array().filter(random_mask(filtered_len)).unwrap();
    let indices = random_indices(num_indices, filtered_len);

    bench_take_filter(bencher, array, indices);
}

#[divan::bench(args = LARGE_TAKE_CASES)]
fn take_filter_struct_large_random_mask_random_indices(
    bencher: Bencher,
    (filtered_len, num_indices): (usize, usize),
) {
    let array = struct_array().filter(random_mask(filtered_len)).unwrap();
    let indices = random_indices(num_indices, filtered_len);

    bench_take_filter(bencher, array, indices);
}
