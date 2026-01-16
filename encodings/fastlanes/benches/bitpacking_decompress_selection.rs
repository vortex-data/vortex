// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarking filter-then-canonicalize versus canonicalize-then-filter.
//!
//! Before running these benchmarks, modify filter_primitive to unconditionally call
//! filter_primitive_chunk_by_chunk. This ensures we are actually comparing early filtering to late
//! filtering.
#![allow(clippy::unwrap_used)]

use divan::Bencher;
use rand::Rng as _;
use rand::SeedableRng as _;
use rand::rngs::StdRng;
use vortex_array::Array;
use vortex_array::IntoArray as _;
use vortex_array::ToCanonical;
use vortex_array::compute::filter;
use vortex_array::compute::warm_up_vtables;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_dtype::IntegerPType;
use vortex_fastlanes::bitpack_compress::bitpack_to_best_bit_width;
use vortex_mask::Mask;

fn main() {
    warm_up_vtables();
    divan::main();
}

// #[divan::bench(types = [i8, i16, i32, i64], args = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999])]
#[divan::bench(types = [i8, i16, i32, i64], args = [0.005, 0.01, 0.0105, 0.02, 0.03, 0.04, 0.05])]
fn decompress_bitpacking_early_filter<T: IntegerPType>(bencher: Bencher, fraction_kept: f64) {
    let mut rng = StdRng::seed_from_u64(0);
    let values = (0..100_000)
        .map(|_| T::from(rng.random_range(0..100)).unwrap())
        .collect::<BufferMut<T>>()
        .into_array()
        .to_primitive();
    let array = bitpack_to_best_bit_width(&values).unwrap();
    let mask = (0..100_000)
        .map(|_| rng.random_bool(fraction_kept))
        .collect::<BitBuffer>();

    bencher
        // Be sure to reconstruct the mask to avoid cached set_indices
        .with_inputs(|| (&array, Mask::from_buffer(mask.clone())))
        .bench_refs(|(array, mask)| filter(array.as_ref(), mask).unwrap().to_canonical());
}

// #[divan::bench(types = [i8, i16, i32, i64], args = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999])]
#[divan::bench(types = [i8, i16, i32, i64], args = [0.005, 0.01, 0.0105, 0.02, 0.03, 0.04, 0.05])]
fn decompress_bitpacking_late_filter<T: IntegerPType>(bencher: Bencher, fraction_kept: f64) {
    let mut rng = StdRng::seed_from_u64(0);
    let values = (0..100_000)
        .map(|_| T::from(rng.random_range(0..100)).unwrap())
        .collect::<BufferMut<T>>()
        .into_array()
        .to_primitive();

    let array = bitpack_to_best_bit_width(&values).unwrap();

    let mask = (0..100_000)
        .map(|_| rng.random_bool(fraction_kept))
        .collect::<BitBuffer>();

    bencher
        // Be sure to reconstruct the mask to avoid cached set_indices
        .with_inputs(|| (&array, Mask::from_buffer(mask.clone())))
        .bench_refs(|(array, mask)| filter(array.to_canonical().unwrap().as_ref(), mask).unwrap());
}
