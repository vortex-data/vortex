#![allow(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]

use divan::Bencher;
use rand::distributions::Uniform;
use rand::prelude::StdRng;
use rand::{thread_rng, Rng, SeedableRng};
use vortex_array::array::PrimitiveArray;
use vortex_array::compute::take;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_fastlanes::{find_best_bit_width, BitPackedArray};

fn main() {
    divan::main();
}

#[divan::bench]
fn take_10_stratified(bencher: Bencher) {
    let values = fixture(1_000_000, 8);
    let uncompressed = PrimitiveArray::new(values, Validity::NonNullable);
    let packed = BitPackedArray::encode(
        uncompressed.as_ref(),
        find_best_bit_width(&uncompressed).unwrap(),
    )
    .unwrap();
    let indices = PrimitiveArray::from_iter((0..10).map(|i| i * 10_000));

    bencher
        .with_inputs(|| (&packed, &indices))
        .bench_refs(|(packed, indices)| take(packed.as_ref(), indices.as_ref()).unwrap())
}

#[divan::bench]
fn take_10_contiguous(bencher: Bencher) {
    let values = fixture(1_000_000, 8);
    let uncompressed = PrimitiveArray::new(values, Validity::NonNullable);
    let packed = BitPackedArray::encode(
        uncompressed.as_ref(),
        find_best_bit_width(&uncompressed).unwrap(),
    )
    .unwrap();
    let indices = PrimitiveArray::from_iter(0..10);

    bencher
        .with_inputs(|| (&packed, &indices))
        .bench_refs(|(packed, indices)| take(packed.as_ref(), indices.as_ref()).unwrap())
}

#[divan::bench]
fn take_10k_random(bencher: Bencher) {
    let values = fixture(1_000_000, 8);
    let range = Uniform::new(0, values.len());
    let uncompressed = PrimitiveArray::new(values, Validity::NonNullable);
    let packed = BitPackedArray::encode(
        uncompressed.as_ref(),
        find_best_bit_width(&uncompressed).unwrap(),
    )
    .unwrap();

    let rng = StdRng::seed_from_u64(0);
    let indices = PrimitiveArray::from_iter(rng.sample_iter(range).take(10_000).map(|i| i as u32));

    bencher
        .with_inputs(|| (&packed, &indices))
        .bench_refs(|(packed, indices)| take(packed.as_ref(), indices.as_ref()).unwrap())
}

#[divan::bench]
fn take_10k_contiguous(bencher: Bencher) {
    let values = fixture(1_000_000, 8);
    let uncompressed = PrimitiveArray::new(values, Validity::NonNullable);
    let packed = BitPackedArray::encode(
        uncompressed.as_ref(),
        find_best_bit_width(&uncompressed).unwrap(),
    )
    .unwrap();
    let indices = PrimitiveArray::from_iter(0..10_000);

    bencher
        .with_inputs(|| (&packed, &indices))
        .bench_refs(|(packed, indices)| take(packed.as_ref(), indices.as_ref()).unwrap())
}

#[divan::bench]
fn take_200k_dispersed(bencher: Bencher) {
    let values = fixture(1_000_000, 8);
    let uncompressed = PrimitiveArray::new(values.clone(), Validity::NonNullable);
    let packed = BitPackedArray::encode(
        uncompressed.as_ref(),
        find_best_bit_width(&uncompressed).unwrap(),
    )
    .unwrap();
    let indices = PrimitiveArray::from_iter((0..200_000).map(|i| (i * 42) % values.len() as u64));

    bencher
        .with_inputs(|| (&packed, &indices))
        .bench_refs(|(packed, indices)| take(packed.as_ref(), indices.as_ref()).unwrap())
}

#[divan::bench]
fn take_200k_first_chunk_only(bencher: Bencher) {
    let values = fixture(1_000_000, 8);
    let uncompressed = PrimitiveArray::new(values, Validity::NonNullable);
    let packed = BitPackedArray::encode(
        uncompressed.as_ref(),
        find_best_bit_width(&uncompressed).unwrap(),
    )
    .unwrap();
    let indices = PrimitiveArray::from_iter((0..200_000).map(|i| ((i * 42) % 1024) as u64));

    bencher
        .with_inputs(|| (&packed, &indices))
        .bench_refs(|(packed, indices)| take(packed.as_ref(), indices.as_ref()).unwrap())
}

fn fixture(len: usize, bits: usize) -> Buffer<u32> {
    let rng = thread_rng();
    let range = Uniform::new(0_u32, 2_u32.pow(bits as u32));
    rng.sample_iter(range).take(len).collect()
}

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

const BIG_BASE2: u32 = 1048576;
const NUM_EXCEPTIONS: u32 = 10000;

#[divan::bench]
fn patched_take_10_stratified(bencher: Bencher) {
    let values = (0u32..BIG_BASE2 + NUM_EXCEPTIONS).collect::<Buffer<u32>>();
    let uncompressed = PrimitiveArray::new(values, Validity::NonNullable);
    let packed = BitPackedArray::encode(
        uncompressed.as_ref(),
        find_best_bit_width(&uncompressed).unwrap(),
    )
    .unwrap();

    assert!(packed.patches().is_some());
    assert_eq!(
        packed.patches().unwrap().num_patches(),
        NUM_EXCEPTIONS as usize
    );

    let indices = PrimitiveArray::from_iter((0..10).map(|i| i * 10_000));

    bencher
        .with_inputs(|| (&packed, &indices))
        .bench_refs(|(packed, indices)| take(packed.as_ref(), indices.as_ref()).unwrap())
}

#[divan::bench]
fn patched_take_10_contiguous(bencher: Bencher) {
    let values = (0u32..BIG_BASE2 + NUM_EXCEPTIONS).collect::<Buffer<u32>>();
    let uncompressed = PrimitiveArray::new(values, Validity::NonNullable);
    let packed = BitPackedArray::encode(
        uncompressed.as_ref(),
        find_best_bit_width(&uncompressed).unwrap(),
    )
    .unwrap();

    assert!(packed.patches().is_some());
    assert_eq!(
        packed.patches().unwrap().num_patches(),
        NUM_EXCEPTIONS as usize
    );

    let indices = PrimitiveArray::from_iter(0..10);

    bencher
        .with_inputs(|| (&packed, &indices))
        .bench_refs(|(packed, indices)| take(packed.as_ref(), indices.as_ref()).unwrap())
}

#[divan::bench]
fn patched_take_10k_random(bencher: Bencher) {
    let values = (0u32..BIG_BASE2 + NUM_EXCEPTIONS).collect::<Buffer<u32>>();
    let uncompressed = PrimitiveArray::new(values.clone(), Validity::NonNullable);
    let packed = BitPackedArray::encode(
        uncompressed.as_ref(),
        find_best_bit_width(&uncompressed).unwrap(),
    )
    .unwrap();

    let rng = StdRng::seed_from_u64(0);
    let range = Uniform::new(0, values.len());
    let indices = PrimitiveArray::from_iter(rng.sample_iter(range).take(10_000).map(|i| i as u32));

    bencher
        .with_inputs(|| (&packed, &indices))
        .bench_refs(|(packed, indices)| take(packed.as_ref(), indices.as_ref()).unwrap())
}

#[divan::bench]
fn patched_take_10k_contiguous_not_patches(bencher: Bencher) {
    let values = (0u32..BIG_BASE2 + NUM_EXCEPTIONS).collect::<Buffer<u32>>();
    let uncompressed = PrimitiveArray::new(values, Validity::NonNullable);
    let packed = BitPackedArray::encode(
        uncompressed.as_ref(),
        find_best_bit_width(&uncompressed).unwrap(),
    )
    .unwrap();
    let indices = PrimitiveArray::from_iter((0u32..NUM_EXCEPTIONS).cycle().take(10000));

    bencher
        .with_inputs(|| (&packed, &indices))
        .bench_refs(|(packed, indices)| take(packed.as_ref(), indices.as_ref()).unwrap())
}

#[divan::bench]
fn patched_take_10k_contiguous_patches(bencher: Bencher) {
    let values = (0u32..BIG_BASE2 + NUM_EXCEPTIONS).collect::<Buffer<u32>>();
    let uncompressed = PrimitiveArray::new(values, Validity::NonNullable);
    let packed = BitPackedArray::encode(
        uncompressed.as_ref(),
        find_best_bit_width(&uncompressed).unwrap(),
    )
    .unwrap();

    assert!(packed.patches().is_some());
    assert_eq!(
        packed.patches().unwrap().num_patches(),
        NUM_EXCEPTIONS as usize
    );

    let indices =
        PrimitiveArray::from_iter((BIG_BASE2..BIG_BASE2 + NUM_EXCEPTIONS).cycle().take(10000));

    bencher
        .with_inputs(|| (&packed, &indices))
        .bench_refs(|(packed, indices)| take(packed.as_ref(), indices.as_ref()).unwrap())
}

#[divan::bench]
fn patched_take_200k_dispersed(bencher: Bencher) {
    let values = (0u32..BIG_BASE2 + NUM_EXCEPTIONS).collect::<Buffer<u32>>();
    let uncompressed = PrimitiveArray::new(values.clone(), Validity::NonNullable);
    let packed = BitPackedArray::encode(
        uncompressed.as_ref(),
        find_best_bit_width(&uncompressed).unwrap(),
    )
    .unwrap();
    let indices = PrimitiveArray::from_iter((0..200_000).map(|i| (i * 42) % values.len() as u64));

    bencher
        .with_inputs(|| (&packed, &indices))
        .bench_refs(|(packed, indices)| take(packed.as_ref(), indices.as_ref()).unwrap())
}

#[divan::bench]
fn patched_take_200k_first_chunk_only(bencher: Bencher) {
    let values = (0u32..BIG_BASE2 + NUM_EXCEPTIONS).collect::<Buffer<u32>>();
    let uncompressed = PrimitiveArray::new(values, Validity::NonNullable);
    let packed = BitPackedArray::encode(
        uncompressed.as_ref(),
        find_best_bit_width(&uncompressed).unwrap(),
    )
    .unwrap();
    let indices = PrimitiveArray::from_iter((0..200_000).map(|i| ((i * 42) % 1024) as u64));

    bencher
        .with_inputs(|| (&packed, &indices))
        .bench_refs(|(packed, indices)| take(packed.as_ref(), indices.as_ref()).unwrap())
}

#[divan::bench]
fn patched_take_10k_adversarial(bencher: Bencher) {
    let values = (0u32..BIG_BASE2 + NUM_EXCEPTIONS).collect::<Buffer<u32>>();
    let uncompressed = PrimitiveArray::new(values, Validity::NonNullable);
    let packed = BitPackedArray::encode(
        uncompressed.as_ref(),
        find_best_bit_width(&uncompressed).unwrap(),
    )
    .unwrap();
    let per_chunk_count = 100;
    let indices = PrimitiveArray::from_iter(
        (0..(NUM_EXCEPTIONS + 1024) / 1024)
            .cycle()
            .map(|chunk_idx| BIG_BASE2 - 1024 + chunk_idx * 1024)
            .flat_map(|base_idx| (base_idx..(base_idx + per_chunk_count)))
            .take(10000),
    );

    bencher
        .with_inputs(|| (&packed, &indices))
        .bench_refs(|(packed, indices)| take(packed.as_ref(), indices.as_ref()).unwrap())
}
