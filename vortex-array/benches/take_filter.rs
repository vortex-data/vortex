// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for taking from a lazy [`FilterArray`].
//!
//! Parameterized over:
//! - Number of indices to take
//! - Number of rows retained by the filter
//! - Filter mask layout (single contiguous slice vs random positions)
//! - Take index layout (sequential vs random ranks)

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::FilterArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_buffer::Buffer;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

const ARRAY_LEN: usize = 100_000;
const FILTERED_LENS: &[usize] = &[10_000, 50_000, 90_000];
const NUM_INDICES: &[usize] = &[1_000, 10_000];
const MASK_SEED: u64 = 42;
const INDEX_SEED: u64 = 43;

fn primitive_array() -> ArrayRef {
    PrimitiveArray::from_iter(0..ARRAY_LEN as u32).into_array()
}

fn slice_mask(filtered_len: usize) -> Mask {
    let start = (ARRAY_LEN - filtered_len) / 2;
    Mask::from_slices(ARRAY_LEN, vec![(start, start + filtered_len)])
}

fn random_mask(filtered_len: usize) -> Mask {
    let mut indices: Vec<usize> = (0..ARRAY_LEN).collect();
    indices.shuffle(&mut StdRng::seed_from_u64(MASK_SEED));
    indices.truncate(filtered_len);
    indices.sort_unstable();
    Mask::from_indices(ARRAY_LEN, indices)
}

fn sequential_indices(num_indices: usize) -> ArrayRef {
    Buffer::from_iter(0..num_indices as u64).into_array()
}

fn random_indices(num_indices: usize, filtered_len: usize) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(INDEX_SEED);
    Buffer::from_iter((0..num_indices).map(|_| rng.random_range(0..filtered_len as u64)))
        .into_array()
}

#[divan::bench(args = NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_slice_mask_sequential_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = primitive_array().filter(slice_mask(FILTERED_LEN)).unwrap();
    let indices = sequential_indices(num_indices);

    bencher
        .with_inputs(|| (&array, &indices, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(array, indices, ctx)| {
            array
                .take(indices.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
                .unwrap()
        });
}

#[divan::bench(args = NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_slice_mask_random_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = primitive_array().filter(slice_mask(FILTERED_LEN)).unwrap();
    let indices = random_indices(num_indices, FILTERED_LEN);

    bencher
        .with_inputs(|| (&array, &indices, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(array, indices, ctx)| {
            array
                .take(indices.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
                .unwrap()
        });
}

#[divan::bench(args = NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_random_mask_sequential_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = primitive_array().filter(random_mask(FILTERED_LEN)).unwrap();
    let indices = sequential_indices(num_indices);

    bencher
        .with_inputs(|| (&array, &indices, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(array, indices, ctx)| {
            array
                .take(indices.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
                .unwrap()
        });
}

#[divan::bench(args = NUM_INDICES, consts = FILTERED_LENS)]
fn take_filter_random_mask_random_indices<const FILTERED_LEN: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let array = primitive_array().filter(random_mask(FILTERED_LEN)).unwrap();
    let indices = random_indices(num_indices, FILTERED_LEN);

    bencher
        .with_inputs(|| (&array, &indices, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(array, indices, ctx)| {
            array
                .take(indices.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
                .unwrap()
        });
}
