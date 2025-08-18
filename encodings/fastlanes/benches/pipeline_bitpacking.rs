// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
#![allow(unexpected_cfgs)]

use arrow_buffer::BooleanBuffer;
use divan::Bencher;
use mimalloc::MiMalloc;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use vortex_array::compute::filter;
use vortex_array::pipeline::canonical::export_canonical_pipeline_expr;
use vortex_array::pipeline::types::Element;
use vortex_array::{IntoArray, ToCanonical};
use vortex_buffer::BufferMut;
use vortex_dtype::NativePType;
use vortex_fastlanes::bitpack_to_best_bit_width;
use vortex_mask::Mask;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub fn main() {
    divan::main();
}

#[divan::bench(types = [i8, i16, i32, i64], args = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999])]
// #[divan::bench(types = [i8, i16, i32, i64], args = [0.005, 0.01, 0.0105, 0.02, 0.03, 0.04, 0.05])]
pub fn decompress_bitpacking_early_filter<T: NativePType>(bencher: Bencher, fraction_kept: f64) {
    let mut rng = StdRng::seed_from_u64(0);
    let values = (0..100_000)
        .map(|_| T::from(rng.random_range(0..100)).unwrap())
        .collect::<BufferMut<T>>()
        .into_array()
        .to_primitive()
        .unwrap();
    let array = bitpack_to_best_bit_width(&values).unwrap();

    let mask = (0..100_000)
        .map(|_| rng.random_bool(fraction_kept))
        .collect::<BooleanBuffer>();

    bencher
        // Be sure to reconstruct the mask to avoid cached set_indices
        .with_inputs(|| Mask::from_buffer(mask.clone()))
        .bench_local_values(|mask| {
            filter(array.as_ref(), &mask)
                .unwrap()
                .to_canonical()
                .unwrap()
        });
}

#[divan::bench(types = [i8, i16, i32, i64], args = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999])]
// #[divan::bench(types = [i8, i16, i32, i64], args = [0.005, 0.01, 0.0105, 0.02, 0.03, 0.04, 0.05])]
pub fn decompress_bitpacking_late_filter<T: NativePType>(bencher: Bencher, fraction_kept: f64) {
    let mut rng = StdRng::seed_from_u64(0);
    let values = (0..100_000)
        .map(|_| T::from(rng.random_range(0..100)).unwrap())
        .collect::<BufferMut<T>>()
        .into_array()
        .to_primitive()
        .unwrap();

    let array = bitpack_to_best_bit_width(&values).unwrap();

    let mask = (0..100_000)
        .map(|_| rng.random_bool(fraction_kept))
        .collect::<BooleanBuffer>();

    bencher
        .with_inputs(|| Mask::from_buffer(mask.clone()))
        .bench_values(|mask| filter(array.to_canonical().unwrap().as_ref(), &mask).unwrap());
}

#[divan::bench(types = [i8, i16, i32, i64], args = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999])]
// #[divan::bench(types = [i8, i16, i32, i64], args = [0.005, 0.01, 0.0105, 0.02, 0.03, 0.04, 0.05])]
pub fn decompress_bitpacking_pipeline_filter<T: Element + NativePType>(
    bencher: Bencher,
    fraction_kept: f64,
) {
    let mut rng = StdRng::seed_from_u64(0);
    let values = (0..100_000)
        .map(|_| T::from(rng.random_range(0..100)).unwrap())
        .collect::<BufferMut<T>>()
        .into_array()
        .to_primitive()
        .unwrap();
    let array = bitpack_to_best_bit_width(&values).unwrap();

    let mask = (0..100_000)
        .map(|_| rng.random_bool(fraction_kept))
        .collect::<BooleanBuffer>();

    let expect = filter(
        array.to_canonical().unwrap().as_ref(),
        &Mask::from_buffer(mask.clone()),
    )
    .unwrap();

    bencher
        .with_inputs(|| Mask::from_buffer(mask.clone()))
        .bench_local_values(|mask| {
            export_canonical_pipeline_expr(
                array.dtype(),
                array.len(),
                array.to_operator().unwrap().as_ref(),
                &mask,
            )
            .unwrap()
        });

    let array = export_canonical_pipeline_expr(
        array.dtype(),
        array.len(),
        array.to_operator().unwrap().as_ref(),
        &Mask::from_buffer(mask.clone()),
    )
    .unwrap()
    .into_primitive()
    .unwrap();
    assert_eq!(array.len(), mask.count_set_bits());

    assert_eq!(
        array.into_buffer::<T>(),
        expect.to_primitive().unwrap().into_buffer::<T>()
    );
}
