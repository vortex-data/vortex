// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
#![allow(unexpected_cfgs)]

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::Rng;
use rand::SeedableRng;
use rand::prelude::StdRng;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::compute::filter;
use vortex_array::compute::warm_up_vtables;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_dtype::NativePType;
use vortex_fastlanes::bitpack_compress::bitpack_to_best_bit_width;
use vortex_mask::Mask;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub fn main() {
    warm_up_vtables();
    divan::main();
}

const TRUE_COUNT: &[f64] = &[0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999];
const LENGTH: usize = 100_000;

#[divan::bench(types = [i8, i16, i32, i64], args = TRUE_COUNT)]
pub fn decompress_bitpacking_early_filter<T: NativePType>(bencher: Bencher, fraction_kept: f64) {
    let mut rng = StdRng::seed_from_u64(0);
    let values = (0..LENGTH)
        .map(|_| T::from(rng.random_range(0..100)).unwrap())
        .collect::<BufferMut<T>>()
        .into_array()
        .to_primitive();
    let array = bitpack_to_best_bit_width(&values).unwrap();

    let mask = (0..LENGTH)
        .map(|_| rng.random_bool(fraction_kept))
        .collect::<BitBuffer>();

    bencher
        // Be sure to reconstruct the mask to avoid cached set_indices
        .with_inputs(|| Mask::from_buffer(mask.clone()))
        .bench_refs(|mask| filter(array.as_ref(), mask).unwrap().to_canonical());
}

#[divan::bench(types = [i8, i16, i32, i64], args = TRUE_COUNT)]
pub fn decompress_bitpacking_late_filter<T: NativePType>(bencher: Bencher, fraction_kept: f64) {
    let mut rng = StdRng::seed_from_u64(0);
    let values = (0..LENGTH)
        .map(|_| T::from(rng.random_range(0..100)).unwrap())
        .collect::<BufferMut<T>>()
        .into_array()
        .to_primitive();

    let array = bitpack_to_best_bit_width(&values).unwrap();

    let mask = (0..LENGTH)
        .map(|_| rng.random_bool(fraction_kept))
        .collect::<BitBuffer>();

    bencher
        // Be sure to reconstruct the mask to avoid cached set_indices
        .with_inputs(|| Mask::from_buffer(mask.clone()))
        .bench_refs(|mask| filter(array.to_canonical().as_ref(), mask).unwrap());
}

#[divan::bench(types = [i8, i16, i32, i64], args = TRUE_COUNT)]
pub fn decompress_bitpacking_pipeline_filter<T: NativePType>(bencher: Bencher, fraction_kept: f64) {
    let mut rng = StdRng::seed_from_u64(0);
    let values = (0..LENGTH)
        .map(|_| T::from(rng.random_range(0..100)).unwrap())
        .collect::<BufferMut<T>>()
        .into_array()
        .to_primitive();
    let array = bitpack_to_best_bit_width(&values).unwrap();

    let mask = (0..LENGTH)
        .map(|_| rng.random_bool(fraction_kept))
        .collect::<BitBuffer>();

    bencher
        // Be sure to reconstruct the mask to avoid cached set_indices
        .with_inputs(|| Mask::from(mask.clone()))
        .bench_refs(|mask| array.execute_with_selection(mask).unwrap());
}
