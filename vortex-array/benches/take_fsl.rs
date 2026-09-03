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
use num_traits::FromPrimitive;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PiecewiseSequenceArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::half::f16;
use vortex_array::match_smallest_offset_type;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_session::VortexSession;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

/// Number of lists in the source array.
const NUM_LISTS: usize = 500;

/// Number of indices to take.
///
/// Together with [`LIST_SIZES`] these are sized so the widest type at the largest list size stays
/// under 1ms per iteration under codspeed simulation.
const NUM_INDICES: &[usize] = &[10, 100];

/// Fixed size list lengths (elements per list). See [`NUM_INDICES`].
const LIST_SIZES: &[usize] = &[16, 64, 128, 256];

/// F16 list lengths for isolating the per-index, piecewise, and manual range-copy strategies.
const F16_STRATEGY_LIST_SIZES: &[usize] = &[1, 2, 4, 8, 16, 64, 128, 256, 512, 1024, 2048];

/// F16 strategy benchmarks keep a smaller take width so the forced slow strategies stay cheap.
const F16_STRATEGY_NUM_INDICES: &[usize] = &[10];

/// Creates a FixedSizeListArray with the given list size and number of lists.
fn create_fsl<T>(list_size: usize, num_lists: usize) -> FixedSizeListArray
where
    T: NativePType + FromPrimitive,
{
    create_fsl_with_validity::<T>(list_size, num_lists, Validity::NonNullable)
}

fn create_fsl_with_validity<T>(
    list_size: usize,
    num_lists: usize,
    validity: Validity,
) -> FixedSizeListArray
where
    T: NativePType + FromPrimitive,
{
    let total_elements = list_size * num_lists;
    let elements: Buffer<T> = (0..total_elements)
        .map(|idx| T::from_u16((idx % 251) as u16).unwrap())
        .collect();
    FixedSizeListArray::new(elements.into_array(), list_size as u32, validity, num_lists)
}

fn create_i64_fsl_with_validity(
    list_size: usize,
    num_lists: usize,
    validity: Validity,
) -> FixedSizeListArray {
    let total_elements = list_size * num_lists;
    let elements: Buffer<i64> = (0..total_elements as i64).collect();
    FixedSizeListArray::new(elements.into_array(), list_size as u32, validity, num_lists)
}

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

#[divan::bench(args = NUM_INDICES, consts = LIST_SIZES)]
fn take_fsl_random<const LIST_SIZE: usize>(bencher: Bencher, num_indices: usize) {
    let fsl = create_i64_fsl_with_validity(LIST_SIZE, NUM_LISTS, Validity::NonNullable);
    bench_take_fsl_random::<i64, LIST_SIZE>(bencher, num_indices, fsl);
}

#[divan::bench(args = NUM_INDICES, consts = LIST_SIZES)]
fn take_fsl_f16_random<const LIST_SIZE: usize>(bencher: Bencher, num_indices: usize) {
    take_fsl_random_typed::<f16, LIST_SIZE>(bencher, num_indices);
}

#[divan::bench(args = NUM_INDICES, consts = LIST_SIZES)]
fn take_fsl_u8_random<const LIST_SIZE: usize>(bencher: Bencher, num_indices: usize) {
    take_fsl_random_typed::<u8, LIST_SIZE>(bencher, num_indices);
}

#[divan::bench(args = NUM_INDICES, consts = LIST_SIZES)]
fn take_fsl_u32_random<const LIST_SIZE: usize>(bencher: Bencher, num_indices: usize) {
    take_fsl_random_typed::<u32, LIST_SIZE>(bencher, num_indices);
}

#[divan::bench(args = NUM_INDICES, consts = LIST_SIZES)]
fn take_fsl_u64_random<const LIST_SIZE: usize>(bencher: Bencher, num_indices: usize) {
    take_fsl_random_typed::<u64, LIST_SIZE>(bencher, num_indices);
}

fn take_fsl_random_typed<T, const LIST_SIZE: usize>(bencher: Bencher, num_indices: usize)
where
    T: NativePType + FromPrimitive,
{
    let fsl = create_fsl::<T>(LIST_SIZE, NUM_LISTS);
    bench_take_fsl_random::<T, LIST_SIZE>(bencher, num_indices, fsl);
}

fn bench_take_fsl_random<T, const LIST_SIZE: usize>(
    bencher: Bencher,
    num_indices: usize,
    fsl: FixedSizeListArray,
) where
    T: NativePType,
{
    let indices = create_random_indices(num_indices, NUM_LISTS);
    let indices_array = indices.into_array();

    bencher
        .counter(BytesCount::of_many::<T>(num_indices * LIST_SIZE))
        .with_inputs(|| (&fsl, &indices_array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, indices, execution_ctx)| {
            array
                .clone()
                .take(indices.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(execution_ctx)
                .unwrap()
        });
}

#[divan::bench(args = F16_STRATEGY_NUM_INDICES, consts = F16_STRATEGY_LIST_SIZES)]
fn take_fsl_f16_force_per_index<const LIST_SIZE: usize>(bencher: Bencher, num_indices: usize) {
    let fsl = create_fsl::<f16>(LIST_SIZE, NUM_LISTS);
    let indices = create_random_indices(num_indices, NUM_LISTS);

    bencher
        .counter(BytesCount::of_many::<f16>(num_indices * LIST_SIZE))
        .with_inputs(|| (&fsl, &indices, SESSION.create_execution_ctx()))
        .bench_refs(|(array, indices, execution_ctx)| {
            match_smallest_offset_type!(array.elements().len(), |E| {
                take_fsl_f16_per_index_strategy::<LIST_SIZE, E>(array, indices)
            })
            .into_array()
            .execute::<RecursiveCanonical>(execution_ctx)
            .unwrap()
        });
}

#[divan::bench(args = F16_STRATEGY_NUM_INDICES, consts = F16_STRATEGY_LIST_SIZES)]
fn take_fsl_f16_force_piecewise_sequence<const LIST_SIZE: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let fsl = create_fsl::<f16>(LIST_SIZE, NUM_LISTS);
    let indices = create_random_indices(num_indices, NUM_LISTS);

    bencher
        .counter(BytesCount::of_many::<f16>(num_indices * LIST_SIZE))
        .with_inputs(|| (&fsl, &indices, SESSION.create_execution_ctx()))
        .bench_refs(|(array, indices, execution_ctx)| {
            take_fsl_f16_piecewise_sequence_strategy::<LIST_SIZE>(array, indices)
                .into_array()
                .execute::<RecursiveCanonical>(execution_ctx)
                .unwrap()
        });
}

#[divan::bench(args = F16_STRATEGY_NUM_INDICES, consts = F16_STRATEGY_LIST_SIZES)]
fn take_fsl_f16_force_manual_range_copy<const LIST_SIZE: usize>(
    bencher: Bencher,
    num_indices: usize,
) {
    let fsl = create_fsl::<f16>(LIST_SIZE, NUM_LISTS);
    let indices = create_random_indices(num_indices, NUM_LISTS);

    bencher
        .counter(BytesCount::of_many::<f16>(num_indices * LIST_SIZE))
        .with_inputs(|| (&fsl, &indices, SESSION.create_execution_ctx()))
        .bench_refs(|(array, indices, execution_ctx)| {
            take_fsl_f16_manual_range_copy_strategy::<LIST_SIZE>(array, indices, execution_ctx)
                .into_array()
                .execute::<RecursiveCanonical>(execution_ctx)
                .unwrap()
        });
}

fn take_fsl_f16_per_index_strategy<const LIST_SIZE: usize, E: IntegerPType>(
    array: &FixedSizeListArray,
    indices: &Buffer<u64>,
) -> FixedSizeListArray {
    let mut element_indices = BufferMut::<E>::with_capacity(indices.len() * LIST_SIZE);
    for &idx in indices.as_ref() {
        let start = idx as usize * LIST_SIZE;
        let end = start + LIST_SIZE;
        for element_idx in start..end {
            // SAFETY: capacity is exactly `indices.len() * LIST_SIZE`, and this loop appends
            // exactly `LIST_SIZE` element indices for each input index.
            unsafe { element_indices.push_unchecked(E::from_usize(element_idx).unwrap()) };
        }
    }

    let element_indices =
        PrimitiveArray::new(element_indices.freeze(), Validity::NonNullable).into_array();
    let elements = array.elements().take(element_indices).unwrap();

    // SAFETY: `elements` was built by taking exactly `LIST_SIZE` elements per input index, so its
    // length is `indices.len() * LIST_SIZE`; the output is non-nullable by construction.
    unsafe {
        FixedSizeListArray::new_unchecked(
            elements,
            LIST_SIZE as u32,
            Validity::NonNullable,
            indices.len(),
        )
    }
}

fn take_fsl_f16_piecewise_sequence_strategy<const LIST_SIZE: usize>(
    array: &FixedSizeListArray,
    indices: &Buffer<u64>,
) -> FixedSizeListArray {
    let starts = indices
        .as_ref()
        .iter()
        .map(|&idx| idx * LIST_SIZE as u64)
        .collect::<Vec<_>>();
    let run_count = starts.len();
    let starts = PrimitiveArray::from_iter(starts).into_array();
    let lengths = ConstantArray::new(LIST_SIZE as u64, run_count).into_array();
    let multipliers = ConstantArray::new(1u64, run_count).into_array();

    // SAFETY: benchmark indices are generated in-bounds; lengths and multiplier 1 are
    // non-nullable unsigned constants; output length is exactly `indices.len() * LIST_SIZE`.
    let element_indices = unsafe {
        PiecewiseSequenceArray::new_unchecked(
            starts,
            lengths,
            multipliers,
            indices.len() * LIST_SIZE,
        )
    }
    .into_array();
    let elements = array.elements().take(element_indices).unwrap();

    // SAFETY: each generated run has width `LIST_SIZE`, and there is one run per input index,
    // so `elements.len() == indices.len() * LIST_SIZE`.
    unsafe {
        FixedSizeListArray::new_unchecked(
            elements,
            LIST_SIZE as u32,
            Validity::NonNullable,
            indices.len(),
        )
    }
}

fn take_fsl_f16_manual_range_copy_strategy<const LIST_SIZE: usize>(
    array: &FixedSizeListArray,
    indices: &Buffer<u64>,
    execution_ctx: &mut ExecutionCtx,
) -> FixedSizeListArray {
    let elements = array
        .elements()
        .clone()
        .execute::<PrimitiveArray>(execution_ctx)
        .unwrap();
    let source = elements.as_slice::<f16>();
    let mut values = BufferMut::<f16>::with_capacity(indices.len() * LIST_SIZE);

    for &idx in indices.as_ref() {
        let start = idx as usize * LIST_SIZE;
        values.extend_from_slice(&source[start..start + LIST_SIZE]);
    }

    // SAFETY: the buffer was filled with exactly `LIST_SIZE` copied values per input index, so it
    // has the element length required by the constructed FSL.
    unsafe {
        FixedSizeListArray::new_unchecked(
            PrimitiveArray::new(values.freeze(), Validity::NonNullable).into_array(),
            LIST_SIZE as u32,
            Validity::NonNullable,
            indices.len(),
        )
    }
}

#[divan::bench(args = NUM_INDICES, consts = LIST_SIZES)]
fn take_fsl_nullable_random<const LIST_SIZE: usize>(bencher: Bencher, num_indices: usize) {
    // Create validity with ~10% nulls
    let mut rng = StdRng::seed_from_u64(123);
    let validity = Validity::from_iter((0..NUM_LISTS).map(|_| rng.random_ratio(9, 10)));

    let fsl = create_i64_fsl_with_validity(LIST_SIZE, NUM_LISTS, validity);
    bench_take_fsl_random::<i64, LIST_SIZE>(bencher, num_indices, fsl);
}

#[divan::bench(args = NUM_INDICES, consts = LIST_SIZES)]
fn take_fsl_f16_nullable_random<const LIST_SIZE: usize>(bencher: Bencher, num_indices: usize) {
    take_fsl_nullable_random_typed::<f16, LIST_SIZE>(bencher, num_indices);
}

fn take_fsl_nullable_random_typed<T, const LIST_SIZE: usize>(bencher: Bencher, num_indices: usize)
where
    T: NativePType + FromPrimitive,
{
    // Create validity with ~10% nulls
    let mut rng = StdRng::seed_from_u64(123);
    let validity = Validity::from_iter((0..NUM_LISTS).map(|_| rng.random_ratio(9, 10)));

    let fsl = create_fsl_with_validity::<T>(LIST_SIZE, NUM_LISTS, validity);
    bench_take_fsl_random::<T, LIST_SIZE>(bencher, num_indices, fsl);
}
