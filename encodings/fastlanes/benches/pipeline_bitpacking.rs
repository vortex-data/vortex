// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
#![allow(unexpected_cfgs)]

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use vortex_array::compute::{filter, warm_up_vtables};
use vortex_array::{IntoArray, ToCanonical};
use vortex_buffer::{BitBuffer, BufferMut};
use vortex_dtype::NativePType;
use vortex_fastlanes::bitpack_to_best_bit_width;
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
        .bench_local_values(|mask| filter(array.as_ref(), &mask).unwrap().to_canonical());
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
        .with_inputs(|| Mask::from_buffer(mask.clone()))
        .bench_values(|mask| filter(array.to_canonical().as_ref(), &mask).unwrap());
}

// TODO(ngates): bring back benchmarks once operator API is stable.
// #[divan::bench(types = [i8, i16, i32, i64], args = TRUE_COUNT)]
// pub fn decompress_bitpacking_pipeline_filter<T: Element + NativePType>(
//     bencher: Bencher,
//     fraction_kept: f64,
// ) {
//     let mut rng = StdRng::seed_from_u64(0);
//     let values = (0..LENGTH)
//         .map(|_| T::from(rng.random_range(0..100)).unwrap())
//         .collect::<BufferMut<T>>()
//         .into_array()
//         .to_primitive();
//     let array = bitpack_to_best_bit_width(&values).unwrap();
//
//     let mask = (0..LENGTH)
//         .map(|_| rng.random_bool(fraction_kept))
//         .collect::<BooleanBuffer>();
//
//     bencher
//         .with_inputs(|| Mask::from_buffer(mask.clone()))
//         .bench_local_values(|mask| {
//             export_canonical_pipeline_expr(
//                 array.dtype(),
//                 array.len(),
//                 array.to_operator().unwrap().unwrap().as_ref(),
//                 &mask,
//             )
//             .unwrap()
//         });
// }
