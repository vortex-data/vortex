#![allow(clippy::unwrap_used)]

use divan::counter::{BytesCount, ItemsCount};
use divan::Bencher;
use rand::prelude::StdRng;
use rand::{RngCore, SeedableRng};
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::{Array, IntoArray, IntoArrayVariant};
use vortex_btrblocks::integer::IntCompressor;
use vortex_btrblocks::Compressor;
use vortex_buffer::buffer_mut;
use vortex_sampling_compressor::SamplingCompressor;

fn make_clickbench_window_name() -> Array {
    // A test that's meant to mirror the WindowName column from ClickBench.
    let mut values = buffer_mut![-1i32; 1_000_000];
    let mut visited = HashSet::new();
    let mut rng = StdRng::seed_from_u64(1u64);
    while visited.len() < 223 {
        let random = (rng.next_u32() as usize) % 1_000_000;
        if visited.contains(&random) {
            continue;
        }
        visited.insert(random);
        // Pick 100 random values to insert.
        values[random] = 5 * (rng.next_u64() % 100) as i32;
    }

    // Ok, now let's compress
    values.freeze().into_array()
}

#[divan::bench]
fn btrblocks(bencher: Bencher) {
    bencher
        .with_inputs(|| make_clickbench_window_name().into_primitive().unwrap())
        .input_counter(|array| ItemsCount::new(array.len()))
        .input_counter(|array| BytesCount::of_many::<i32>(array.len()))
        .bench_local_values(|array| IntCompressor::compress(&array, false, 3, &[]).unwrap());
}

#[divan::bench]
fn sampling_compressor(bencher: Bencher) {
    let compressor = SamplingCompressor::default();
    bencher
        .with_inputs(make_clickbench_window_name)
        .input_counter(|array| ItemsCount::new(array.len()))
        .input_counter(|array| BytesCount::of_many::<i32>(array.len()))
        .bench_local_values(|array| compressor.compress(&array, None).unwrap());
}

fn main() {
    divan::main()
}
