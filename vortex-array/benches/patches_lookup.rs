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

// Patch indices for `narrow_band_patches` are sampled from this window.
const PATCH_LOW: usize = 100_000;
const PATCH_HIGH: usize = 110_000;

/// Build a `Patches` whose indices are sampled from `index_iter`.
///
/// Indices are sorted and deduplicated; the values column is a dense
/// `i32` sequence and is incidental to the benchmarks (which target
/// index lookup, not value materialization).
fn patches_from_indices(index_iter: impl Iterator<Item = u64>) -> Patches {
    let mut indices: Vec<u64> = index_iter.collect();
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

/// All patches clustered in `PATCH_LOW..PATCH_HIGH` — models a localized burst.
fn narrow_band_patches() -> Patches {
    let mut rng = StdRng::seed_from_u64(42);
    patches_from_indices(
        (0..NUM_PATCHES).map(|_| rng.random_range((PATCH_LOW as u64)..(PATCH_HIGH as u64))),
    )
}

/// Patches spread uniformly across the full array.
fn full_range_patches() -> Patches {
    let mut rng = StdRng::seed_from_u64(43);
    patches_from_indices((0..NUM_PATCHES).map(|_| rng.random_range(0..(ARRAY_LEN as u64))))
}

fn bench_search_index(bencher: Bencher, patches: Patches, queries: Vec<usize>) {
    bencher
        .with_inputs(|| (&patches, &queries))
        .bench_refs(|(patches, queries)| {
            for &q in queries.iter() {
                divan::black_box(patches.search_index(q).unwrap());
            }
        });
}

#[divan::bench]
fn search_index_below_min(bencher: Bencher) {
    let queries = (0..NUM_QUERIES).collect();
    bench_search_index(bencher, narrow_band_patches(), queries);
}

#[divan::bench]
fn search_index_above_max(bencher: Bencher) {
    let queries = (PATCH_HIGH..(PATCH_HIGH + NUM_QUERIES)).collect();
    bench_search_index(bencher, narrow_band_patches(), queries);
}

#[divan::bench]
fn search_index_mixed_out_of_range(bencher: Bencher) {
    let queries: Vec<usize> = (0..NUM_QUERIES / 2)
        .map(|i| i * 100)
        .chain((0..NUM_QUERIES / 2).map(|i| PATCH_HIGH + i * 50))
        .collect();
    bench_search_index(bencher, narrow_band_patches(), queries);
}

#[divan::bench]
fn search_index_in_range(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(7);
    let queries: Vec<usize> = (0..NUM_QUERIES)
        .map(|_| rng.random_range(PATCH_LOW..PATCH_HIGH))
        .collect();
    bench_search_index(bencher, narrow_band_patches(), queries);
}

#[divan::bench]
fn search_index_full_range_random(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(11);
    let queries: Vec<usize> = (0..NUM_QUERIES)
        .map(|_| rng.random_range(0..ARRAY_LEN))
        .collect();
    bench_search_index(bencher, full_range_patches(), queries);
}

#[divan::bench]
fn get_patched_above_max(bencher: Bencher) {
    let patches = narrow_band_patches();
    let queries: Vec<usize> = (PATCH_HIGH..(PATCH_HIGH + NUM_QUERIES)).collect();

    bencher
        .with_inputs(|| (&patches, &queries))
        .bench_refs(|(patches, queries)| {
            for &q in queries.iter() {
                divan::black_box(patches.get_patched(q).unwrap());
            }
        });
}
