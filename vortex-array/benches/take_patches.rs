// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]

use divan::Bencher;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vortex_array::patches::Patches;
use vortex_array::{ArrayRef, IntoArray, ToCanonical};
use vortex_buffer::Buffer;

fn main() {
    divan::main();
}

const BENCH_ARGS: &[(f64, f64)] = &[
    // patches_sparsity, index_multiple
    (0.1, 1.0),
    (0.1, 0.5),
    (0.1, 0.1),
    (0.1, 0.05),
    (0.05, 1.0),
    (0.05, 0.5),
    (0.05, 0.1),
    (0.05, 0.05),
    (0.01, 1.0),
    (0.01, 0.5),
    (0.01, 0.1),
    (0.01, 0.05),
    (0.005, 1.0),
    (0.005, 0.5),
    (0.005, 0.1),
    (0.005, 0.05),
];

#[divan::bench(args = BENCH_ARGS)]
fn take_search(bencher: Bencher, (patches_sparsity, index_multiple): (f64, f64)) {
    let mut rng = StdRng::seed_from_u64(0);
    let patches = fixture(65536, patches_sparsity, &mut rng);
    let indices = indices(
        patches.array_len(),
        (patches.array_len() as f64 * index_multiple) as usize,
        &mut rng,
    );

    bencher
        .with_inputs(|| (&patches, indices.clone()))
        .bench_values(|(patches, indices)| patches.take_search(indices.to_primitive(), false));
}

#[divan::bench(args = BENCH_ARGS)]
fn take_search_chunked(bencher: Bencher, (patches_sparsity, index_multiple): (f64, f64)) {
    let mut rng = StdRng::seed_from_u64(0);
    let patches = fixture_with_chunk_offsets(65536, patches_sparsity, &mut rng);
    let indices = indices(
        patches.array_len(),
        (patches.array_len() as f64 * index_multiple) as usize,
        &mut rng,
    );

    bencher
        .with_inputs(|| (&patches, indices.clone()))
        .bench_values(|(patches, indices)| patches.take_search(indices.to_primitive(), false));
}

#[divan::bench(args = BENCH_ARGS)]
fn take_map(bencher: Bencher, (patches_sparsity, index_multiple): (f64, f64)) {
    let mut rng = StdRng::seed_from_u64(0);
    let patches = fixture(65536, patches_sparsity, &mut rng);
    let indices = indices(
        patches.array_len(),
        (patches.array_len() as f64 * index_multiple) as usize,
        &mut rng,
    );

    bencher
        .with_inputs(|| (&patches, indices.clone()))
        .bench_values(|(patches, indices)| patches.take_map(indices.to_primitive(), false));
}

fn fixture(len: usize, sparsity: f64, rng: &mut StdRng) -> Patches {
    let indices = (0..len)
        .filter(|_| rng.random_bool(sparsity))
        .map(|x| x as u64)
        .collect::<Buffer<u64>>();
    let sparse_len = indices.len();
    let values = Buffer::from_iter((0..sparse_len).map(|x| x as u64)).into_array();
    Patches::new(
        len,
        0,
        indices.into_array(),
        values,
        // TODO(0ax1): handle chunk offsets
        None,
    )
}

fn fixture_with_chunk_offsets(len: usize, sparsity: f64, rng: &mut StdRng) -> Patches {
    let patch_indices = (0..len)
        .filter(|_| rng.random_bool(sparsity))
        .map(|x| x as u64)
        .collect::<Vec<u64>>();

    let sparse_len = patch_indices.len();
    let values = Buffer::from_iter((0..sparse_len).map(|x| x as u64)).into_array();

    const PATCH_CHUNK_SIZE: usize = 1024;
    let chunk_offsets: Vec<u64> = (0..len)
        .step_by(PATCH_CHUNK_SIZE)
        .map(|chunk_start| {
            patch_indices.partition_point(|&idx| (idx as usize) < chunk_start) as u64
        })
        .collect();

    Patches::new(
        len,
        0,
        Buffer::from(patch_indices).into_array(),
        values,
        Some(Buffer::from(chunk_offsets).into_array()),
    )
}

fn indices(array_len: usize, n_indices: usize, rng: &mut StdRng) -> ArrayRef {
    Buffer::from_iter((0..n_indices).map(|_| rng.random_range(0..(array_len as u64)))).into_array()
}
