// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::IntoArray;
use vortex_array::patches::Patches;
use vortex_buffer::Buffer;

fn main() {
    divan::main();
}

const ARRAY_LEN: usize = 1_000_000;
const NUM_PATCHES: usize = 100;
const NUM_QUERIES: usize = 1_000;
const PATCH_LOW: usize = 100_000;
const PATCH_HIGH: usize = 110_000;

fn narrow_band_patches() -> Patches {
    let mut rng = StdRng::seed_from_u64(42);
    let mut indices: Vec<u64> = (0..NUM_PATCHES)
        .map(|_| rng.random_range((PATCH_LOW as u64)..(PATCH_HIGH as u64)))
        .collect();
    indices.sort();
    indices.dedup();
    let values: Buffer<i32> = (0..indices.len() as i32).collect();
    Patches::new(
        ARRAY_LEN,
        0,
        Buffer::from(indices).into_array(),
        values.into_array(),
        None,
    )
    .unwrap()
}

fn full_range_patches() -> Patches {
    let mut rng = StdRng::seed_from_u64(43);
    let mut indices: Vec<u64> = (0..NUM_PATCHES)
        .map(|_| rng.random_range(0..(ARRAY_LEN as u64)))
        .collect();
    indices.sort();
    indices.dedup();
    let values: Buffer<i32> = (0..indices.len() as i32).collect();
    Patches::new(
        ARRAY_LEN,
        0,
        Buffer::from(indices).into_array(),
        values.into_array(),
        None,
    )
    .unwrap()
}

#[divan::bench]
fn search_index_below_min(bencher: Bencher) {
    let patches = narrow_band_patches();
    let queries: Vec<usize> = (0..NUM_QUERIES).collect();

    bencher.bench_local(|| {
        for &q in &queries {
            std::hint::black_box(patches.search_index(q).unwrap());
        }
    });
}

#[divan::bench]
fn search_index_above_max(bencher: Bencher) {
    let patches = narrow_band_patches();
    let queries: Vec<usize> = (PATCH_HIGH..(PATCH_HIGH + NUM_QUERIES)).collect();

    bencher.bench_local(|| {
        for &q in &queries {
            std::hint::black_box(patches.search_index(q).unwrap());
        }
    });
}

#[divan::bench]
fn search_index_mixed_out_of_range(bencher: Bencher) {
    let patches = narrow_band_patches();
    let queries: Vec<usize> = (0..NUM_QUERIES / 2)
        .map(|i| i * 100)
        .chain((0..NUM_QUERIES / 2).map(|i| PATCH_HIGH + i * 50))
        .collect();

    bencher.bench_local(|| {
        for &q in &queries {
            std::hint::black_box(patches.search_index(q).unwrap());
        }
    });
}

#[divan::bench]
fn search_index_in_range(bencher: Bencher) {
    let patches = narrow_band_patches();
    let mut rng = StdRng::seed_from_u64(7);
    let queries: Vec<usize> = (0..NUM_QUERIES)
        .map(|_| rng.random_range(PATCH_LOW..PATCH_HIGH))
        .collect();

    bencher.bench_local(|| {
        for &q in &queries {
            std::hint::black_box(patches.search_index(q).unwrap());
        }
    });
}

#[divan::bench]
fn search_index_full_range_random(bencher: Bencher) {
    let patches = full_range_patches();
    let mut rng = StdRng::seed_from_u64(11);
    let queries: Vec<usize> = (0..NUM_QUERIES)
        .map(|_| rng.random_range(0..ARRAY_LEN))
        .collect();

    bencher.bench_local(|| {
        for &q in &queries {
            std::hint::black_box(patches.search_index(q).unwrap());
        }
    });
}

#[divan::bench]
fn get_patched_above_max(bencher: Bencher) {
    let patches = narrow_band_patches();
    let queries: Vec<usize> = (PATCH_HIGH..(PATCH_HIGH + NUM_QUERIES)).collect();

    bencher.bench_local(|| {
        for &q in &queries {
            std::hint::black_box(patches.get_patched(q).unwrap());
        }
    });
}
