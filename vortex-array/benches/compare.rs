#![allow(clippy::unwrap_used)]

use divan::Bencher;
use rand::distr::Uniform;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use vortex_array::arrays::BoolArray;
use vortex_array::compute::{compare, Operator};
use vortex_array::{Array, IntoArray};
use vortex_buffer::Buffer;

fn main() {
    divan::main();
}

const ARRAY_SIZE: usize = 10_000_000;

#[divan::bench]
fn compare_bool(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(0);
    let range = Uniform::new(0u8, 1).unwrap();

    let arr1 = BoolArray::from_iter((0..ARRAY_SIZE).map(|_| rng.sample(range) == 0)).into_array();
    let arr2 = BoolArray::from_iter((0..ARRAY_SIZE).map(|_| rng.sample(range) == 0)).into_array();

    bencher
        .with_inputs(|| (&arr1, &arr2))
        .bench_refs(|(arr1, arr2)| compare(*arr1, *arr2, Operator::Gte).unwrap());
}

#[divan::bench]
fn compare_int(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(0);
    let range = Uniform::new(0i64, 100_000_000).unwrap();

    let arr1 = (0..ARRAY_SIZE)
        .map(|_| rng.sample(range))
        .collect::<Buffer<_>>()
        .into_array();

    let arr2 = (0..ARRAY_SIZE)
        .map(|_| rng.sample(range))
        .collect::<Buffer<_>>()
        .into_array();

    bencher
        .with_inputs(|| (&arr1, &arr2))
        .bench_refs(|(arr1, arr2)| compare(*arr1, *arr2, Operator::Gte).unwrap());
}
