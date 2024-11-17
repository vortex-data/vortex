#![allow(clippy::unwrap_used)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use itertools::Itertools;
use rand::distributions::Uniform;
use rand::{thread_rng, Rng};
use vortex_array::array::{PrimitiveArray, SparseArray};
use vortex_array::compute::{take, TakeOptions};
use vortex_fastlanes::{find_best_bit_width, BitPackedArray};

fn values(len: usize, bits: usize) -> Vec<u32> {
    let rng = thread_rng();
    let range = Uniform::new(0_u32, 2_u32.pow(bits as u32));
    rng.sample_iter(range).take(len).collect()
}

fn bench_take(c: &mut Criterion) {
    let values = values(1_000_000, 8);
    let uncompressed = PrimitiveArray::from(values.clone());

    let packed = BitPackedArray::encode(
        uncompressed.as_ref(),
        find_best_bit_width(&uncompressed).unwrap(),
    )
    .unwrap();

    let stratified_indices: PrimitiveArray = (0..10).map(|i| i * 10_000).collect::<Vec<_>>().into();
    c.bench_function("take_10_stratified", |b| {
        b.iter(|| {
            black_box(
                take(
                    packed.as_ref(),
                    stratified_indices.as_ref(),
                    TakeOptions::default(),
                )
                .unwrap(),
            )
        });
    });

    let contiguous_indices: PrimitiveArray = (0..10).collect::<Vec<_>>().into();
    c.bench_function("take_10_contiguous", |b| {
        b.iter(|| {
            black_box(
                take(
                    packed.as_ref(),
                    contiguous_indices.as_ref(),
                    TakeOptions::default(),
                )
                .unwrap(),
            )
        });
    });

    let rng = thread_rng();
    let range = Uniform::new(0, values.len());
    let random_indices: PrimitiveArray = rng
        .sample_iter(range)
        .take(10_000)
        .map(|i| i as u32)
        .collect_vec()
        .into();
    c.bench_function("take_10K_random", |b| {
        b.iter(|| {
            black_box(
                take(
                    packed.as_ref(),
                    random_indices.as_ref(),
                    TakeOptions::default(),
                )
                .unwrap(),
            )
        });
    });

    let contiguous_indices: PrimitiveArray = (0..10_000).collect::<Vec<_>>().into();
    c.bench_function("take_10K_contiguous", |b| {
        b.iter(|| {
            black_box(
                take(
                    packed.as_ref(),
                    contiguous_indices.as_ref(),
                    TakeOptions::default(),
                )
                .unwrap(),
            )
        });
    });

    let lots_of_indices: PrimitiveArray = (0..200_000)
        .map(|i| (i * 42) % values.len() as u64)
        .collect::<Vec<_>>()
        .into();
    c.bench_function("take_200K_dispersed", |b| {
        b.iter(|| {
            black_box(
                take(
                    packed.as_ref(),
                    lots_of_indices.as_ref(),
                    TakeOptions::default(),
                )
                .unwrap(),
            )
        });
    });

    let lots_of_indices: PrimitiveArray = (0..200_000)
        .map(|i| ((i * 42) % 1024) as u64)
        .collect::<Vec<_>>()
        .into();
    c.bench_function("take_200K_first_chunk_only", |b| {
        b.iter(|| {
            black_box(
                take(
                    packed.as_ref(),
                    lots_of_indices.as_ref(),
                    TakeOptions::default(),
                )
                .unwrap(),
            )
        });
    });
}

fn bench_patched_take(c: &mut Criterion) {
    let big_base2 = 1048576;
    let num_exceptions = 10000;
    let values = (0u32..big_base2 + num_exceptions).collect_vec();

    let uncompressed = PrimitiveArray::from(values.clone());
    let packed = BitPackedArray::encode(
        uncompressed.as_ref(),
        find_best_bit_width(&uncompressed).unwrap(),
    )
    .unwrap();
    assert!(packed.patches().is_some());
    assert_eq!(
        SparseArray::try_from(packed.patches().unwrap())
            .unwrap()
            .values()
            .len(),
        num_exceptions as usize
    );

    let stratified_indices: PrimitiveArray = (0..10).map(|i| i * 10_000).collect::<Vec<_>>().into();
    c.bench_function("patched_take_10_stratified", |b| {
        b.iter(|| {
            black_box(
                take(
                    packed.as_ref(),
                    stratified_indices.as_ref(),
                    TakeOptions::default(),
                )
                .unwrap(),
            )
        });
    });

    let contiguous_indices: PrimitiveArray = (0..10).collect::<Vec<_>>().into();
    c.bench_function("patched_take_10_contiguous", |b| {
        b.iter(|| {
            black_box(
                take(
                    packed.as_ref(),
                    contiguous_indices.as_ref(),
                    TakeOptions::default(),
                )
                .unwrap(),
            )
        });
    });

    let rng = thread_rng();
    let range = Uniform::new(0, values.len());
    let random_indices: PrimitiveArray = rng
        .sample_iter(range)
        .take(10_000)
        .map(|i| i as u32)
        .collect_vec()
        .into();
    c.bench_function("patched_take_10K_random", |b| {
        b.iter(|| {
            black_box(
                take(
                    packed.as_ref(),
                    random_indices.as_ref(),
                    TakeOptions::default(),
                )
                .unwrap(),
            )
        });
    });

    let not_patch_indices: PrimitiveArray = (0u32..num_exceptions)
        .cycle()
        .take(10000)
        .collect_vec()
        .into();
    c.bench_function("patched_take_10K_contiguous_not_patches", |b| {
        b.iter(|| {
            black_box(
                take(
                    packed.as_ref(),
                    not_patch_indices.as_ref(),
                    TakeOptions::default(),
                )
                .unwrap(),
            )
        });
    });

    let patch_indices: PrimitiveArray = (big_base2..big_base2 + num_exceptions)
        .cycle()
        .take(10000)
        .collect_vec()
        .into();
    c.bench_function("patched_take_10K_contiguous_patches", |b| {
        b.iter(|| {
            black_box(
                take(
                    packed.as_ref(),
                    patch_indices.as_ref(),
                    TakeOptions::default(),
                )
                .unwrap(),
            )
        });
    });

    let lots_of_indices: PrimitiveArray = (0..200_000)
        .map(|i| (i * 42) % values.len() as u64)
        .collect::<Vec<_>>()
        .into();
    c.bench_function("patched_take_200K_dispersed", |b| {
        b.iter(|| {
            black_box(
                take(
                    packed.as_ref(),
                    lots_of_indices.as_ref(),
                    TakeOptions::default(),
                )
                .unwrap(),
            )
        });
    });

    let lots_of_indices: PrimitiveArray = (0..200_000)
        .map(|i| ((i * 42) % 1024) as u64)
        .collect::<Vec<_>>()
        .into();
    c.bench_function("patched_take_200K_first_chunk_only", |b| {
        b.iter(|| {
            black_box(
                take(
                    packed.as_ref(),
                    lots_of_indices.as_ref(),
                    TakeOptions::default(),
                )
                .unwrap(),
            )
        });
    });

    // There are currently 2 magic parameters of note:
    // 1. the threshold at which sparse take will switch from search_sorted to map (currently 128)
    // 2. the threshold at which bitpacked take will switch from bulk patching to per chunk patching (currently 64)
    //
    // There are thus 3 cases to consider:
    // 1. N < 64 per chunk, covered by patched_take_10K_random
    // 2. N > 128 per chunk, covered by patched_take_10K_contiguous_*
    // 3. 64 < N < 128 per chunk, which is what we're trying to cover here (with 100 per chunk).
    //
    // As a result of the above, we get both search_sorted and per chunk patching, almost entirely on patches.
    // I've iterated on both thresholds (1) and (2) using this collection of benchmarks, and those
    // were roughly the best values that I found.
    let per_chunk_count = 100;
    let adversarial_indices: PrimitiveArray = (0..(num_exceptions + 1024) / 1024)
        .cycle()
        .map(|chunk_idx| big_base2 - 1024 + chunk_idx * 1024)
        .flat_map(|base_idx| (base_idx..(base_idx + per_chunk_count)))
        .take(10000)
        .collect_vec()
        .into();
    c.bench_function("patched_take_10K_adversarial", |b| {
        b.iter(|| {
            black_box(
                take(
                    packed.as_ref(),
                    adversarial_indices.as_ref(),
                    TakeOptions::default(),
                )
                .unwrap(),
            )
        });
    });
}

criterion_group!(benches, bench_take, bench_patched_take);
criterion_main!(benches);
