// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for taking from a lazy [`FilterArray`].
//!
//! Parameterized over:
//! - Element type (primitive, struct, list, string)
//! - Number of indices to take
//! - Number of rows retained by the filter
//! - Filter mask layout (single contiguous slice vs random positions)
//! - Take index layout (sequential vs random ranks)
//! - Nullable vs non-null take indices
//!
//! Nested types (struct, list) build much smaller arrays and take far fewer indices than the
//! primitive case so the suite stays manageable.

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

const ARRAY_LEN: usize = 25_000;
const FILTERED_LENS: &[usize] = &[4_096, 16_384];
const NUM_INDICES: &[usize] = &[1_000];
const SMALL_NUM_INDICES: &[usize] = &[10];
const LARGE_TAKE_CASES: &[(usize, usize)] = &[(2_500, 25_000), (12_500, 25_000)];

/// Nested types are heavier per element, so they use a much smaller array and take fewer indices.
const NESTED_ARRAY_LEN: usize = 1_024;
const NESTED_FILTERED_LENS: &[usize] = &[256, 768];
const NESTED_NUM_INDICES: &[usize] = &[50];

const MASK_SEED: u64 = 42;
const INDEX_SEED: u64 = 43;
const LIST_SIZE: usize = 4;
const NULL_INDEX_INTERVAL: usize = 8;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

fn primitive_array() -> ArrayRef {
    PrimitiveArray::from_iter(0..ARRAY_LEN as u32).into_array()
}

fn list_array(len: usize) -> ArrayRef {
    let elements = PrimitiveArray::from_iter(0..(len * LIST_SIZE) as u32).into_array();
    let offsets = Buffer::from_iter((0..=len).map(|idx| (idx * LIST_SIZE) as u32)).into_array();

    ListArray::try_new(elements, offsets, Validity::NonNullable)
        .unwrap()
        .into_array()
}

fn slice_mask(array_len: usize, filtered_len: usize) -> Mask {
    let start = (array_len - filtered_len) / 2;
    Mask::from_buffer(BitBuffer::from_iter(
        (0..array_len).map(|idx| (start..start + filtered_len).contains(&idx)),
    ))
}

fn random_mask(array_len: usize, filtered_len: usize) -> Mask {
    Mask::from_buffer(random_mask_buffer(array_len, filtered_len))
}

fn random_mask_buffer(array_len: usize, filtered_len: usize) -> BitBuffer {
    let mut indices: Vec<usize> = (0..array_len).collect();
    indices.shuffle(&mut StdRng::seed_from_u64(MASK_SEED));
    indices.truncate(filtered_len);

    let mut buffer = BitBufferMut::new_unset(array_len);
    for idx in indices {
        buffer.set(idx);
    }

    buffer.freeze()
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

fn bench_take_filter_uncached_mask(
    bencher: Bencher,
    child: ArrayRef,
    mask: BitBuffer,
    indices: ArrayRef,
) {
    bencher
        .with_inputs(|| {
            let array = child
                .clone()
                .filter(Mask::from_buffer(mask.clone()))
                .unwrap();
            (array, indices.clone(), SESSION.create_execution_ctx())
        })
        .bench_values(|(array, indices, mut ctx)| {
            array
                .take(indices)
                .unwrap()
                .execute::<RecursiveCanonical>(&mut ctx)
                .unwrap()
        });
}

#[divan::bench(args = NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_primitive_slice_mask_sequential_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = primitive_array()
        .filter(slice_mask(ARRAY_LEN, FILTERED_LEN))
        .unwrap();
    bench_take_filter(bencher, array, sequential_indices(num_indices));
}

#[divan::bench(args = NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_primitive_slice_mask_random_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = primitive_array()
        .filter(slice_mask(ARRAY_LEN, FILTERED_LEN))
        .unwrap();
    bench_take_filter(bencher, array, random_indices(num_indices, FILTERED_LEN));
}

#[divan::bench(args = NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_primitive_random_mask_sequential_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = primitive_array()
        .filter(random_mask(ARRAY_LEN, FILTERED_LEN))
        .unwrap();
    bench_take_filter(bencher, array, sequential_indices(num_indices));
}

#[divan::bench(args = NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_primitive_random_mask_random_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = primitive_array()
        .filter(random_mask(ARRAY_LEN, FILTERED_LEN))
        .unwrap();
    bench_take_filter(bencher, array, random_indices(num_indices, FILTERED_LEN));
}

#[divan::bench(args = SMALL_NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_primitive_small_random_mask_random_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = primitive_array()
        .filter(random_mask(ARRAY_LEN, FILTERED_LEN))
        .unwrap();
    bench_take_filter(bencher, array, random_indices(num_indices, FILTERED_LEN));
}

#[divan::bench(args = SMALL_NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_primitive_small_uncached_random_mask_random_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    bench_take_filter_uncached_mask(
        bencher,
        primitive_array(),
        random_mask_buffer(ARRAY_LEN, FILTERED_LEN),
        random_indices(num_indices, FILTERED_LEN),
    );
}

#[divan::bench(args = NESTED_NUM_INDICES, consts = NESTED_FILTERED_LENS)]
fn take_filter_list_slice_mask_sequential_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = list_array(NESTED_ARRAY_LEN)
        .filter(slice_mask(NESTED_ARRAY_LEN, FILTERED_LEN))
        .unwrap();
    bench_take_filter(bencher, array, sequential_indices(num_indices));
}

#[divan::bench(args = NESTED_NUM_INDICES, consts = NESTED_FILTERED_LENS)]
fn take_filter_list_slice_mask_random_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = list_array(NESTED_ARRAY_LEN)
        .filter(slice_mask(NESTED_ARRAY_LEN, FILTERED_LEN))
        .unwrap();
    bench_take_filter(bencher, array, random_indices(num_indices, FILTERED_LEN));
}

#[divan::bench(args = NESTED_NUM_INDICES, consts = NESTED_FILTERED_LENS)]
fn take_filter_list_random_mask_sequential_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = list_array(NESTED_ARRAY_LEN)
        .filter(random_mask(NESTED_ARRAY_LEN, FILTERED_LEN))
        .unwrap();
    bench_take_filter(bencher, array, sequential_indices(num_indices));
}

#[divan::bench(args = NESTED_NUM_INDICES, consts = NESTED_FILTERED_LENS)]
fn take_filter_list_random_mask_random_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = list_array(NESTED_ARRAY_LEN)
        .filter(random_mask(NESTED_ARRAY_LEN, FILTERED_LEN))
        .unwrap();
    bench_take_filter(bencher, array, random_indices(num_indices, FILTERED_LEN));
}

#[divan::bench(args = SMALL_NUM_INDICES, consts = NESTED_FILTERED_LENS)]
fn take_filter_list_small_random_mask_random_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = list_array(NESTED_ARRAY_LEN)
        .filter(random_mask(NESTED_ARRAY_LEN, FILTERED_LEN))
        .unwrap();
    bench_take_filter(bencher, array, random_indices(num_indices, FILTERED_LEN));
}

#[divan::bench(args = SMALL_NUM_INDICES, consts = NESTED_FILTERED_LENS)]
fn take_filter_list_small_uncached_random_mask_random_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    bench_take_filter_uncached_mask(
        bencher,
        list_array(NESTED_ARRAY_LEN),
        random_mask_buffer(NESTED_ARRAY_LEN, FILTERED_LEN),
        random_indices(num_indices, FILTERED_LEN),
    );
}

#[divan::bench(args = NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_primitive_nullable_slice_mask_random_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = primitive_array()
        .filter(slice_mask(ARRAY_LEN, FILTERED_LEN))
        .unwrap();
    bench_take_filter(
        bencher,
        array,
        nullable_random_indices(num_indices, FILTERED_LEN),
    );
}

#[divan::bench(args = NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_primitive_nullable_random_mask_random_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = primitive_array()
        .filter(random_mask(ARRAY_LEN, FILTERED_LEN))
        .unwrap();
    bench_take_filter(
        bencher,
        array,
        nullable_random_indices(num_indices, FILTERED_LEN),
    );
}

#[divan::bench(args = NESTED_NUM_INDICES, consts = NESTED_FILTERED_LENS)]
fn take_filter_list_nullable_random_mask_random_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = list_array(NESTED_ARRAY_LEN)
        .filter(random_mask(NESTED_ARRAY_LEN, FILTERED_LEN))
        .unwrap();
    bench_take_filter(
        bencher,
        array,
        nullable_random_indices(num_indices, FILTERED_LEN),
    );
}

#[divan::bench(args = LARGE_TAKE_CASES)]
fn take_filter_primitive_large_random_mask_random_indices(
    bencher: Bencher,
    (filtered_len, num_indices): (usize, usize),
) {
    let array = primitive_array()
        .filter(random_mask(ARRAY_LEN, filtered_len))
        .unwrap();
    bench_take_filter(bencher, array, random_indices(num_indices, filtered_len));
}
