// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::patches::PATCH_CHUNK_SIZE;
use vortex_array::patches::Patches;
use vortex_array::patches_v2::PatchesV2;
use vortex_buffer::Buffer;

fn main() {
    divan::main();
}

const ARRAY_LEN: usize = 1_000_000;
const NUM_PATCHES: usize = 100;
// Sized to keep CodSpeed simulation under 1ms per benchmark.
const NUM_QUERIES: usize = 100;

const PATCH_LOW: usize = 100_000;
const PATCH_HIGH: usize = 110_000;

fn patches_from_indices(index_iter: impl Iterator<Item = u64>, chunked: bool) -> Patches {
    let mut indices: Vec<u64> = index_iter.collect();
    indices.sort();
    indices.dedup();
    let values: Buffer<i32> = (0..indices.len() as i32).collect();

    let chunk_offsets = chunked.then(|| {
        let offsets: Vec<u64> = (0..ARRAY_LEN)
            .step_by(PATCH_CHUNK_SIZE)
            .map(|chunk_start| indices.partition_point(|&idx| (idx as usize) < chunk_start) as u64)
            .collect();
        Buffer::from(offsets).into_array()
    });

    let patches = Patches::new(
        ARRAY_LEN,
        0,
        Buffer::from(indices).into_array(),
        values.into_array(),
        chunk_offsets,
    )
    .unwrap();
    if chunked {
        assert!(patches.chunk_offsets().is_some());
    }
    patches
}

fn narrow_band_patches(chunked: bool) -> Patches {
    let mut rng = StdRng::seed_from_u64(42);
    patches_from_indices(
        (0..NUM_PATCHES).map(|_| rng.random_range((PATCH_LOW as u64)..(PATCH_HIGH as u64))),
        chunked,
    )
}

fn full_range_patches(chunked: bool) -> Patches {
    let mut rng = StdRng::seed_from_u64(43);
    patches_from_indices(
        (0..NUM_PATCHES).map(|_| rng.random_range(0..(ARRAY_LEN as u64))),
        chunked,
    )
}

fn queries_below_min() -> Vec<usize> {
    (0..NUM_QUERIES).collect()
}

fn queries_above_max() -> Vec<usize> {
    (PATCH_HIGH..(PATCH_HIGH + NUM_QUERIES)).collect()
}

fn queries_mixed_out_of_range() -> Vec<usize> {
    (0..NUM_QUERIES / 2)
        .map(|i| i * 100)
        .chain((0..NUM_QUERIES / 2).map(|i| PATCH_HIGH + i * 50))
        .collect()
}

fn queries_in_range() -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(7);
    (0..NUM_QUERIES)
        .map(|_| rng.random_range(PATCH_LOW..PATCH_HIGH))
        .collect()
}

fn queries_full_range() -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(11);
    (0..NUM_QUERIES)
        .map(|_| rng.random_range(0..ARRAY_LEN))
        .collect()
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
    bench_search_index(bencher, narrow_band_patches(false), queries_below_min());
}

#[divan::bench]
fn search_index_below_min_chunked(bencher: Bencher) {
    bench_search_index(bencher, narrow_band_patches(true), queries_below_min());
}

#[divan::bench]
fn search_index_above_max(bencher: Bencher) {
    bench_search_index(bencher, narrow_band_patches(false), queries_above_max());
}

#[divan::bench]
fn search_index_above_max_chunked(bencher: Bencher) {
    bench_search_index(bencher, narrow_band_patches(true), queries_above_max());
}

#[divan::bench]
fn search_index_mixed_out_of_range(bencher: Bencher) {
    bench_search_index(
        bencher,
        narrow_band_patches(false),
        queries_mixed_out_of_range(),
    );
}

#[divan::bench]
fn search_index_mixed_out_of_range_chunked(bencher: Bencher) {
    bench_search_index(
        bencher,
        narrow_band_patches(true),
        queries_mixed_out_of_range(),
    );
}

#[divan::bench]
fn search_index_in_range(bencher: Bencher) {
    bench_search_index(bencher, narrow_band_patches(false), queries_in_range());
}

#[divan::bench]
fn search_index_in_range_chunked(bencher: Bencher) {
    bench_search_index(bencher, narrow_band_patches(true), queries_in_range());
}

#[divan::bench]
fn search_index_full_range_random(bencher: Bencher) {
    bench_search_index(bencher, full_range_patches(false), queries_full_range());
}

#[divan::bench]
fn search_index_full_range_random_chunked(bencher: Bencher) {
    bench_search_index(bencher, full_range_patches(true), queries_full_range());
}

fn patches_v2_from(patches: &Patches) -> PatchesV2 {
    let mut ctx = array_session().create_execution_ctx();
    PatchesV2::from_patches(patches, &mut ctx).unwrap()
}

fn bench_search_index_v2(bencher: Bencher, patches: PatchesV2, queries: Vec<usize>) {
    let mut ctx = array_session().create_execution_ctx();
    bencher
        .with_inputs(|| (&patches, &queries))
        .bench_local_refs(|(patches, queries)| {
            for &q in queries.iter() {
                divan::black_box(patches.search_index(q, &mut ctx).unwrap());
            }
        });
}

#[divan::bench]
fn search_index_below_min_v2(bencher: Bencher) {
    bench_search_index_v2(
        bencher,
        patches_v2_from(&narrow_band_patches(false)),
        queries_below_min(),
    );
}

#[divan::bench]
fn search_index_above_max_v2(bencher: Bencher) {
    bench_search_index_v2(
        bencher,
        patches_v2_from(&narrow_band_patches(false)),
        queries_above_max(),
    );
}

#[divan::bench]
fn search_index_mixed_out_of_range_v2(bencher: Bencher) {
    bench_search_index_v2(
        bencher,
        patches_v2_from(&narrow_band_patches(false)),
        queries_mixed_out_of_range(),
    );
}

#[divan::bench]
fn search_index_in_range_v2(bencher: Bencher) {
    bench_search_index_v2(
        bencher,
        patches_v2_from(&narrow_band_patches(false)),
        queries_in_range(),
    );
}

#[divan::bench]
fn search_index_full_range_random_v2(bencher: Bencher) {
    bench_search_index_v2(
        bencher,
        patches_v2_from(&full_range_patches(false)),
        queries_full_range(),
    );
}

fn bench_apply_v1(bencher: Bencher, patches: Patches) {
    let mut ctx = array_session().create_execution_ctx();
    let indices = patches
        .indices()
        .clone()
        .execute::<vortex_array::arrays::PrimitiveArray>(&mut ctx)
        .unwrap();
    bencher
        .with_inputs(|| vec![0i64; ARRAY_LEN])
        .bench_local_values(|mut dst| {
            for &index in indices.as_slice::<u64>() {
                dst[index as usize] = 1;
            }
            divan::black_box(dst);
        });
}

fn bench_apply_v2(bencher: Bencher, patches: PatchesV2) {
    let mut ctx = array_session().create_execution_ctx();
    bencher
        .with_inputs(|| vec![0i64; ARRAY_LEN])
        .bench_local_values(|mut dst| {
            patches
                .apply_each(&mut ctx, |logical, _ordinal| dst[logical] = 1)
                .unwrap();
            divan::black_box(dst);
        });
}

#[divan::bench]
fn apply_full_range(bencher: Bencher) {
    bench_apply_v1(bencher, full_range_patches(false));
}

#[divan::bench]
fn apply_full_range_v2(bencher: Bencher) {
    bench_apply_v2(bencher, patches_v2_from(&full_range_patches(false)));
}

#[divan::bench]
fn slice_unaligned(bencher: Bencher) {
    let patches = full_range_patches(true);
    bencher.bench(|| divan::black_box(patches.slice(1_000..900_000).unwrap()));
}

#[divan::bench]
fn slice_unaligned_v2(bencher: Bencher) {
    let patches = patches_v2_from(&full_range_patches(true));
    let mut ctx = array_session().create_execution_ctx();
    bencher.bench_local(|| divan::black_box(patches.slice(1_000..900_000, &mut ctx).unwrap()));
}
