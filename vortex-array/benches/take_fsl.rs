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
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PiecewiseSequenceArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
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

/// Number of indices to take. This keeps even the widest, longest cases below one millisecond in
/// CodSpeed's instruction-count simulation.
const NUM_INDICES: &[usize] = &[10];

/// Fixed size list lengths (elements per list).
const LIST_SIZES: &[usize] = &[16, 64, 128, 256, 512, 1024, 2048, 4096];

/// F16 list lengths for isolating the per-index, piecewise, and manual range-copy strategies.
const F16_STRATEGY_LIST_SIZES: &[usize] = &[1, 2, 4, 8, 16, 64, 128, 256, 512, 1024, 2048, 4096];

/// Creates a FixedSizeListArray with the given list size and number of lists.
fn create_fsl<T>(list_size: usize, num_lists: usize) -> FixedSizeListArray
where
    T: NativePType + FromPrimitive,
{
    let total_elements = list_size * num_lists;
    let elements: Buffer<T> = (0..total_elements)
        .map(|idx| T::from_u16((idx % 251) as u16).unwrap())
        .collect();
    FixedSizeListArray::new(
        elements.into_array(),
        list_size as u32,
        Validity::NonNullable,
        num_lists,
    )
}

/// Creates random indices for taking from the array.
fn create_random_indices(num_indices: usize, max_index: usize) -> Buffer<u64> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..num_indices)
        .map(|_| rng.random_range(0..max_index) as u64)
        .collect()
}

#[divan::bench(args = NUM_INDICES, consts = LIST_SIZES)]
fn take_fsl_f16_random<const LIST_SIZE: usize>(bencher: Bencher, num_indices: usize) {
    take_fsl_random::<f16, LIST_SIZE>(bencher, num_indices);
}

#[divan::bench(args = NUM_INDICES, consts = LIST_SIZES)]
fn take_fsl_u8_random<const LIST_SIZE: usize>(bencher: Bencher, num_indices: usize) {
    take_fsl_random::<u8, LIST_SIZE>(bencher, num_indices);
}

#[divan::bench(args = NUM_INDICES, consts = LIST_SIZES)]
fn take_fsl_u32_random<const LIST_SIZE: usize>(bencher: Bencher, num_indices: usize) {
    take_fsl_random::<u32, LIST_SIZE>(bencher, num_indices);
}

#[divan::bench(args = NUM_INDICES, consts = LIST_SIZES)]
fn take_fsl_u64_random<const LIST_SIZE: usize>(bencher: Bencher, num_indices: usize) {
    take_fsl_random::<u64, LIST_SIZE>(bencher, num_indices);
}

fn take_fsl_random<T, const LIST_SIZE: usize>(bencher: Bencher, num_indices: usize)
where
    T: NativePType + FromPrimitive,
{
    let fsl = create_fsl::<T>(LIST_SIZE, NUM_LISTS);
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

#[divan::bench(args = NUM_INDICES, consts = F16_STRATEGY_LIST_SIZES)]
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

#[divan::bench(args = NUM_INDICES, consts = F16_STRATEGY_LIST_SIZES)]
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

#[divan::bench(args = NUM_INDICES, consts = F16_STRATEGY_LIST_SIZES)]
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
fn take_fsl_f16_nullable_random<const LIST_SIZE: usize>(bencher: Bencher, num_indices: usize) {
    let total_elements = LIST_SIZE * NUM_LISTS;
    let elements: Buffer<f16> = (0..total_elements)
        .map(|idx| f16::from_u16((idx % 251) as u16).unwrap())
        .collect();

    // Create validity with ~10% nulls
    let mut rng = StdRng::seed_from_u64(123);
    let validity = Validity::from_iter((0..NUM_LISTS).map(|_| rng.random_ratio(9, 10)));

    let fsl = FixedSizeListArray::new(elements.into_array(), LIST_SIZE as u32, validity, NUM_LISTS);

    let indices = create_random_indices(num_indices, NUM_LISTS);
    let indices_array = indices.into_array();

    bencher
        .counter(BytesCount::of_many::<f16>(num_indices * LIST_SIZE))
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
