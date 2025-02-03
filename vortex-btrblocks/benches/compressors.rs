#![allow(clippy::unwrap_used)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::prelude::StdRng;
use rand::{RngCore, SeedableRng};
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::{Array, IntoArray, IntoArrayVariant};
use vortex_btrblocks::integer::IntCompressor;
use vortex_btrblocks::Compressor;
use vortex_buffer::buffer_mut;
use vortex_sampling_compressor::SamplingCompressor;

fn clickbench_window_name() -> Array {
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

fn compress_btrblocks(c: &mut Criterion) {
    let window_name = clickbench_window_name().into_primitive().unwrap();

    c.bench_function("btrblocks", |b| {
        b.iter(|| IntCompressor::compress(&window_name, false, 3, &[]).unwrap())
    });
}

fn compress_sampling(c: &mut Criterion) {
    let compressor = SamplingCompressor::default();

    c.bench_function("sampling", |b| {
        b.iter(|| {
            black_box(
                compressor
                    .compress(&clickbench_window_name(), None)
                    .unwrap(),
            )
        })
    });
}

criterion_group!(benches, compress_btrblocks, compress_sampling);
criterion_main!(benches);
