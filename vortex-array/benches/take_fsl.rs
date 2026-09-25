// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for take operations on [`FixedSizeListArray`].
//!
//! Parameterized over:
//! - Number of indices to take
//! - Fixed size list length (elements per list)
//! - Element byte width

#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::BytesCount;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

/// Number of lists in the source array.
const NUM_LISTS: usize = 500;

/// Creates random indices for taking from the array.
fn create_random_indices(num_indices: usize, max_index: usize) -> Buffer<u64> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..num_indices)
        .map(|_| rng.random_range(0..max_index) as u64)
        .collect()
}

/// Number of chunks used for the chunked-elements and chunked-of-FSL benchmarks.
const NUM_CHUNKS: usize = 8;

/// Take widths for the chunked benchmarks.
///
/// Together with [`CHUNKED_LIST_SIZES`] these are sized so that the slowest current take path
/// (per-element expansion through the generic chunked take) stays under 1ms per iteration under
/// codspeed simulation, which reports roughly an order of magnitude above local wall time.
const CHUNKED_NUM_INDICES: &[usize] = &[64];

/// Fixed size list lengths for the chunked benchmarks. See [`CHUNKED_NUM_INDICES`].
const CHUNKED_LIST_SIZES: &[usize] = &[8, 16, 32];

/// Creates a `FixedSizeListArray` whose elements child is a `ChunkedArray` with `NUM_CHUNKS`
/// chunks whose boundaries do not line up with list boundaries.
fn create_fsl_with_chunked_elements(list_size: usize, num_lists: usize) -> FixedSizeListArray {
    let total_elements = list_size * num_lists;
    let chunk_len = total_elements.div_ceil(NUM_CHUNKS);
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < total_elements {
        let end = (start + chunk_len).min(total_elements);
        let chunk: Buffer<i64> = (start as i64..end as i64).collect();
        chunks.push(chunk.into_array());
        start = end;
    }
    let dtype = chunks[0].dtype().clone();
    let elements = ChunkedArray::try_new(chunks, dtype).unwrap().into_array();
    FixedSizeListArray::new(elements, list_size as u32, Validity::NonNullable, num_lists)
}

/// Creates a `ChunkedArray` of `NUM_CHUNKS` `FixedSizeListArray` chunks.
fn create_chunked_fsl(list_size: usize, num_lists: usize) -> ArrayRef {
    let lists_per_chunk = num_lists.div_ceil(NUM_CHUNKS);
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < num_lists {
        let end = (start + lists_per_chunk).min(num_lists);
        let elements: Buffer<i64> =
            ((start * list_size) as i64..(end * list_size) as i64).collect();
        chunks.push(
            FixedSizeListArray::new(
                elements.into_array(),
                list_size as u32,
                Validity::NonNullable,
                end - start,
            )
            .into_array(),
        );
        start = end;
    }
    let dtype = chunks[0].dtype().clone();
    ChunkedArray::try_new(chunks, dtype).unwrap().into_array()
}

fn bench_take_array<const LIST_SIZE: usize>(
    bencher: Bencher,
    num_indices: usize,
    array: ArrayRef,
    sorted: bool,
) {
    let mut indices = create_random_indices(num_indices, NUM_LISTS)
        .as_ref()
        .to_vec();
    if sorted {
        indices.sort_unstable();
    }
    let indices_array = Buffer::from(indices).into_array();

    bencher
        .counter(BytesCount::of_many::<i64>(num_indices * LIST_SIZE))
        .with_inputs(|| (&array, &indices_array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, indices, execution_ctx)| {
            array
                .take((*indices).clone())
                .unwrap()
                .execute::<RecursiveCanonical>(execution_ctx)
                .unwrap()
        });
}

#[divan::bench(args = CHUNKED_NUM_INDICES, consts = CHUNKED_LIST_SIZES)]
fn take_fsl_chunked_elements_random<const LIST_SIZE: usize>(bencher: Bencher, num_indices: usize) {
    let fsl = create_fsl_with_chunked_elements(LIST_SIZE, NUM_LISTS);
    bench_take_array::<LIST_SIZE>(bencher, num_indices, fsl.into_array(), false);
}

#[divan::bench(args = CHUNKED_NUM_INDICES, consts = CHUNKED_LIST_SIZES)]
fn take_fsl_chunked_elements_sorted<const LIST_SIZE: usize>(bencher: Bencher, num_indices: usize) {
    let fsl = create_fsl_with_chunked_elements(LIST_SIZE, NUM_LISTS);
    bench_take_array::<LIST_SIZE>(bencher, num_indices, fsl.into_array(), true);
}

#[divan::bench(args = CHUNKED_NUM_INDICES, consts = CHUNKED_LIST_SIZES)]
fn take_chunked_fsl_random<const LIST_SIZE: usize>(bencher: Bencher, num_indices: usize) {
    let array = create_chunked_fsl(LIST_SIZE, NUM_LISTS);
    bench_take_array::<LIST_SIZE>(bencher, num_indices, array, false);
}

#[divan::bench(args = CHUNKED_NUM_INDICES, consts = CHUNKED_LIST_SIZES)]
fn take_chunked_fsl_sorted<const LIST_SIZE: usize>(bencher: Bencher, num_indices: usize) {
    let array = create_chunked_fsl(LIST_SIZE, NUM_LISTS);
    bench_take_array::<LIST_SIZE>(bencher, num_indices, array, true);
}
