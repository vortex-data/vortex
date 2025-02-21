#![allow(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]

use divan::Bencher;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vortex_array::patches::Patches;
use vortex_array::{Array, IntoArray, IntoArrayVariant};
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
    let patches = fixture(16384, patches_sparsity, &mut rng);
    let indices = indices(
        patches.array_len(),
        (patches.array_len() as f64 * index_multiple) as usize,
        &mut rng,
    );

    bencher
        .with_inputs(|| (&patches, indices.clone()))
        .bench_values(|(patches, indices)| patches.take_search(indices.into_primitive().unwrap()));
}

#[divan::bench(args = BENCH_ARGS)]
fn take_map(bencher: Bencher, (patches_sparsity, index_multiple): (f64, f64)) {
    let mut rng = StdRng::seed_from_u64(0);
    let patches = fixture(16384, patches_sparsity, &mut rng);
    let indices = indices(
        patches.array_len(),
        (patches.array_len() as f64 * index_multiple) as usize,
        &mut rng,
    );

    bencher
        .with_inputs(|| (&patches, indices.clone()))
        .bench_values(|(patches, indices)| patches.take_map(indices.into_primitive().unwrap()));
}

fn fixture(len: usize, sparsity: f64, rng: &mut StdRng) -> Patches {
    let indices = (0..len)
        .filter(|_| rng.gen_bool(sparsity))
        .map(|x| x as u64)
        .collect::<Buffer<u64>>();
    let sparse_len = indices.len();
    let values = Buffer::from_iter((0..sparse_len).map(|x| x as u64)).into_array();
    Patches::new(len, 0, indices.into_array(), values)
}

fn indices(array_len: usize, n_indices: usize, rng: &mut StdRng) -> Array {
    Buffer::from_iter((0..n_indices).map(|_| rng.gen_range(0..(array_len as u64)))).into_array()
}
