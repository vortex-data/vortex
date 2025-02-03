#![allow(unused_imports, unused, dead_code)]
//! Various tests for the selection vector being present.

use criterion::{BenchmarkId, Criterion};
use rand::Rng;
use vortex::array::PrimitiveArray;
use vortex::compute::filter;
use vortex::dtype::{DType, Nullability, PType};
use vortex::encodings::alp::{ALPArray, ALPEncoding};
use vortex::mask::Mask;
use vortex::sampling_compressor::compressors::alp::ALPCompressor;
use vortex::sampling_compressor::compressors::bitpacked::{
    BitPackedCompressor, BITPACK_NO_PATCHES, BITPACK_WITH_PATCHES,
};
use vortex::sampling_compressor::compressors::r#for::FoRCompressor;
use vortex::sampling_compressor::compressors::EncodingCompressor;
use vortex::sampling_compressor::SamplingCompressor;
use vortex::variants::PrimitiveArrayTrait;
use vortex::{Array, Encoding, IntoArray, IntoCanonical};

// criterion benchmark setup:
fn bench_sel_vec(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_then_canonical");

    // Run ALP + BitPacking.
    let compressor = SamplingCompressor::default().including_only(&[
        &ALPCompressor as &dyn EncodingCompressor,
        &BITPACK_NO_PATCHES,
        // &FoRCompressor,
    ]);

    // Create a low-precision primitive array of f64
    let arr = PrimitiveArray::from_iter((0..=65535).map(|x| (x as f64) * 0.2f64));
    assert_eq!(arr.ptype(), PType::F64);

    let arr = compressor
        .compress(&arr.into_array(), None)
        .unwrap()
        .into_array();
    assert_eq!(arr.encoding(), ALPEncoding::ID);

    println!("tree: {}", arr.tree_display());

    // Try for various mask
    let max = 65536;
    for selectivity in [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0] {
        // Create a random mask of the given size
        let true_count = (selectivity * max as f64) as usize;
        // Create a randomized mask with the correct length and true_count.
        let mask = create_mask(max, true_count);
        assert_eq!(mask.len(), max);
        assert_eq!(mask.true_count(), true_count);
        group.bench_with_input(
            BenchmarkId::from_parameter(selectivity),
            &mask,
            |b, mask| {
                // Filter then into_canonical
                b.iter(|| filter_then_canonical(&arr, mask))
            },
        );
    }
    group.finish();

    let mut group = c.benchmark_group("canonical_then_filter");
    for selectivity in [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0] {
        // Create a random mask of the given size
        let true_count = (selectivity * max as f64) as usize;
        // Create a randomized mask with the correct length and true_count.
        let mask = create_mask(max, true_count);
        group.bench_with_input(
            BenchmarkId::from_parameter(selectivity),
            &mask,
            |b, mask| {
                // Filter then into_canonical
                b.iter(|| canonical_then_filter(&arr, mask))
            },
        );
    }
    group.finish();
}

fn filter_then_canonical(array: &Array, mask: &Mask) -> Array {
    let filtered = filter(array, mask).unwrap();
    filtered.into_canonical().unwrap().into_array()
}

fn canonical_then_filter(array: &Array, mask: &Mask) -> Array {
    let canonical = array.clone().into_canonical().unwrap().into_array();
    filter(&canonical, mask).unwrap()
}

fn create_mask(len: usize, true_count: usize) -> Mask {
    let mut mask = vec![false; len];
    // randomly distribute true_count true values
    let mut rng = rand::thread_rng();
    let mut set = 0;
    while set < true_count {
        let index = rng.gen_range(0..len);
        if !mask[index] {
            mask[index] = true;
            set += 1;
        }
    }
    Mask::from_iter(mask)
}

criterion::criterion_group!(sel_vec, bench_sel_vec);
criterion::criterion_main!(sel_vec);
