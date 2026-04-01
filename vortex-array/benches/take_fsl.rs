// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for take operations on [`FixedSizeListArray`].
//!
//! Parameterized over:
//! - Number of indices to take
//! - Fixed size list length (elements per list)

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::unwrap_used)]

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;

fn main() {
    divan::main();
}

/// Number of lists in the source array.
const NUM_LISTS: usize = 500;

/// Number of indices to take.
const NUM_INDICES: &[usize] = &[100, 1_000];

/// Fixed size list lengths (elements per list).
const LIST_SIZES: &[usize] = &[16, 64, 256, 1024, 4096];

/// Creates a FixedSizeListArray with the given list size and number of lists.
fn create_fsl(list_size: usize, num_lists: usize) -> FixedSizeListArray {
    let total_elements = list_size * num_lists;
    let elements: Buffer<i64> = (0..total_elements as i64).collect();
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
fn take_fsl_random<const LIST_SIZE: usize>(bencher: Bencher, num_indices: usize) {
    let fsl = create_fsl(LIST_SIZE, NUM_LISTS);
    let indices = create_random_indices(num_indices, NUM_LISTS);
    let indices_array = indices.into_array();

    bencher
        .with_inputs(|| (&fsl, &indices_array, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(array, indices, execution_ctx)| {
            array
                .clone()
                .take(indices.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(execution_ctx)
                .unwrap()
        });
}

#[divan::bench(args = NUM_INDICES, consts = LIST_SIZES)]
fn take_fsl_nullable_random<const LIST_SIZE: usize>(bencher: Bencher, num_indices: usize) {
    let total_elements = LIST_SIZE * NUM_LISTS;
    let elements: Buffer<i64> = (0..total_elements as i64).collect();

    // Create validity with ~10% nulls
    let mut rng = StdRng::seed_from_u64(123);
    let validity = Validity::from_iter((0..NUM_LISTS).map(|_| rng.random_ratio(9, 10)));

    let fsl = FixedSizeListArray::new(elements.into_array(), LIST_SIZE as u32, validity, NUM_LISTS);

    let indices = create_random_indices(num_indices, NUM_LISTS);
    let indices_array = indices.into_array();

    bencher
        .with_inputs(|| (&fsl, &indices_array, LEGACY_SESSION.create_execution_ctx()))
        .bench_refs(|(array, indices, execution_ctx)| {
            array
                .clone()
                .take(indices.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(execution_ctx)
                .unwrap()
        });
}
