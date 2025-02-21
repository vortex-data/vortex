//! Benchmarking filter-then-canonicalize versus canonicalize-then-filter.
//!
//! Before running these benchmarks, modify filter_primitive to unconditionally call
//! filter_primitive_chunk_by_chunk. This ensures we are actually comparing early filtering to late
//! filtering.
#![allow(clippy::unwrap_used)]

use divan::Bencher;
use rand::rngs::StdRng;
use rand::{Rng as _, SeedableRng as _};
use vortex_array::arrays::BooleanBuffer;
use vortex_array::compute::filter;
use vortex_array::{IntoArray as _, IntoArrayVariant as _, IntoCanonical};
use vortex_buffer::BufferMut;
use vortex_dtype::NativePType;
use vortex_fastlanes::bitpack_to_best_bit_width;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

// #[divan::bench(types = [i8, i16, i32, i64], args = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999])]
#[divan::bench(types = [i8, i16, i32, i64], args = [0.005, 0.01, 0.0105, 0.02, 0.03, 0.04, 0.05])]
fn decompress_bitpacking_early_filter<T: NativePType>(bencher: Bencher, fraction_kept: f64) {
    let mut rng = StdRng::seed_from_u64(0);
    let values = (0..10_000)
        .map(|_| T::from(rng.gen_range(0..100)).unwrap())
        .collect::<BufferMut<T>>()
        .into_array()
        .into_primitive()
        .unwrap();

    let array = bitpack_to_best_bit_width(values).unwrap();
    let array = array.as_ref();

    let mask = (0..10000)
        .map(|_| rng.gen_bool(fraction_kept))
        .collect::<BooleanBuffer>();
    let mask = &Mask::from_buffer(mask);

    bencher.bench(|| filter(array, mask).unwrap().into_canonical().unwrap());
}

// #[divan::bench(types = [i8, i16, i32, i64], args = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999])]
#[divan::bench(types = [i8, i16, i32, i64], args = [0.005, 0.01, 0.0105, 0.02, 0.03, 0.04, 0.05])]
fn decompress_bitpacking_late_filter<T: NativePType>(bencher: Bencher, fraction_kept: f64) {
    let mut rng = StdRng::seed_from_u64(0);
    let values = (0..10_000)
        .map(|_| T::from(rng.gen_range(0..100)).unwrap())
        .collect::<BufferMut<T>>()
        .into_array()
        .into_primitive()
        .unwrap();

    let array = bitpack_to_best_bit_width(values).unwrap();
    let array = array.as_ref();

    let mask = (0..10000)
        .map(|_| rng.gen_bool(fraction_kept))
        .collect::<BooleanBuffer>();
    let mask = &Mask::from_buffer(mask);

    bencher
        .with_inputs(|| array.clone())
        .bench_values(|array| filter(array.into_canonical().unwrap().as_ref(), mask).unwrap());
}
