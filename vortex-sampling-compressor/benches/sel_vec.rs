#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use divan::Bencher;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vortex_alp::ALPEncoding;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::compute::filter;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{Array, Encoding, IntoArray};
use vortex_dtype::PType;
use vortex_mask::Mask;
use vortex_sampling_compressor::SamplingCompressor;
use vortex_sampling_compressor::compressors::EncodingCompressor;
use vortex_sampling_compressor::compressors::alp::ALPCompressor;
use vortex_sampling_compressor::compressors::bitpacked::BITPACK_NO_PATCHES;

fn main() {
    divan::main();
}

const BENCH_ARGS: &[(usize, f64)] = &[
    // array length, selectivity ratios
    (65536, 0.001), // 0.1%
    (65536, 0.01),  // 1%
    (65536, 0.1),   // 10%
    (65536, 0.5),   // 50%
    (65536, 0.9),   // 90%
    (65536, 0.99),  // 99%
    (65536, 0.999), // 99.9%
    (65536, 1.0),   // 100%
];

/// Benchmark the filter-then-canonical approach
/// This tests performance when filtering is done before decompressing
#[divan::bench(args = BENCH_ARGS)]
fn filter_then_canonical(bencher: Bencher, (max, selectivity): (usize, f64)) {
    // Create a low-precision primitive array of f64
    let arr = PrimitiveArray::from_iter((0..=65535).map(|x| (x as f64) * 0.2f64));
    assert_eq!(arr.ptype(), PType::F64);

    // Setup compressor with ALP + BitPacking
    let compressor = SamplingCompressor::default().including_only(&[
        &ALPCompressor as &dyn EncodingCompressor,
        &BITPACK_NO_PATCHES,
    ]);

    // Compress the array using ALP encoding
    let arr = compressor
        .compress(&arr.into_array(), None)
        .unwrap()
        .into_array();

    assert_eq!(arr.encoding(), ALPEncoding::ID);

    // Create mask with given selectivity
    let true_count = (selectivity * max as f64) as usize;
    let mask = create_mask(max, true_count);
    assert_eq!(mask.len(), max);
    assert_eq!(mask.true_count(), true_count);

    bencher
        .with_inputs(|| (arr.clone(), mask.clone()))
        .bench_refs(|(arr, mask)| {
            let filtered = filter(arr, mask).unwrap();
            filtered.to_canonical().unwrap().into_array()
        });
}

/// Benchmarks when decompression happens before filtering.
#[divan::bench(args = BENCH_ARGS)]
fn canonical_then_filter(bencher: Bencher, (max, selectivity): (usize, f64)) {
    // Create test array and compress it
    let arr = PrimitiveArray::from_iter((0..=65535).map(|x| (x as f64) * 0.2f64));
    let compressor = SamplingCompressor::default().including_only(&[
        &ALPCompressor as &dyn EncodingCompressor,
        &BITPACK_NO_PATCHES,
    ]);

    let arr = compressor
        .compress(&arr.into_array(), None)
        .unwrap()
        .into_array();

    // Create filter mask with desired selectivity
    let true_count = (selectivity * max as f64) as usize;
    let mask = create_mask(max, true_count);

    bencher
        .with_inputs(|| (arr.clone(), &mask))
        .bench_values(|(arr, mask)| {
            let canonical = arr.to_canonical().unwrap().into_array();
            filter(&canonical, mask)
        });
}

/// Create a mask with randomly distributed true values.
///
/// # Arguments
/// * `len` - Length of the mask
/// * `true_count` - Number of true values to include in the mask
fn create_mask(len: usize, true_count: usize) -> Mask {
    let mut mask = vec![false; len];
    let mut rng = StdRng::seed_from_u64(0);
    let mut set = 0;
    // Randomly distribute true values until we reach the desired count
    while set < true_count {
        let index = rng.random_range(0..len);
        if !mask[index] {
            mask[index] = true;
            set += 1;
        }
    }
    Mask::from_iter(mask)
}
