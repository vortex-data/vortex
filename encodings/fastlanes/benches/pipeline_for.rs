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
use vortex_array::pipeline::{Element, export_canonical_pipeline_expr};
use vortex_array::{IntoArray, ToCanonical};
use vortex_buffer::BufferMut;
use vortex_dtype::NativePType;
use vortex_fastlanes::{FoRArray, bitpack_to_best_bit_width};
use vortex_mask::Mask;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub fn main() {
    divan::main();
}

fn create_for_bitpacked_array<T: NativePType>(
    values: BufferMut<T>,
) -> vortex_error::VortexResult<vortex_array::ArrayRef> {
    let primitive_array = values.into_array().to_primitive();

    // First apply FoR encoding
    let for_array = FoRArray::encode(primitive_array)?;

    // Then bitpack the residuals
    let residuals = for_array.encoded().to_primitive();
    let bitpacked = bitpack_to_best_bit_width(&residuals)?;

    // Create a new FoR array with bitpacked residuals
    Ok(
        FoRArray::try_new(bitpacked.into_array(), for_array.reference_scalar().clone())?
            .into_array(),
    )
}

const TRUE_COUNT: &[f64] = &[
    0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.00,
];

const LENGTH: usize = 102_400;

#[divan::bench(types = [i8, i16, i32, i64], args = TRUE_COUNT)]
pub fn decompress_for_early_filter<T: NativePType>(bencher: Bencher, fraction_kept: f64) {
    let mut rng = StdRng::seed_from_u64(0);
    let values = (0..LENGTH)
        .map(|_| T::from(rng.random_range(26..127)).unwrap())
        .collect::<BufferMut<T>>();

    let array = create_for_bitpacked_array(values).unwrap();
    // let mask = generate_mask_with_runs(102_400, fraction_kept, &mut rng);
    let mask = (0..LENGTH)
        .map(|_| rng.random_bool(fraction_kept))
        .collect::<BooleanBuffer>();

    bencher
        .with_inputs(|| Mask::from_buffer(mask.clone()))
        .bench_local_values(|mask| filter(array.as_ref(), &mask).unwrap().to_canonical());
}

#[divan::bench(types = [i8, i16, i32, i64], args = TRUE_COUNT)]
#[allow(dead_code)]
pub fn decompress_for_pipeline_plan_filter<T: Element + NativePType>(
    bencher: Bencher,
    fraction_kept: f64,
) {
    let mut rng = StdRng::seed_from_u64(0);
    let values = (0..LENGTH)
        .map(|_| T::from(rng.random_range(26..127)).unwrap())
        .collect::<BufferMut<T>>();

    let array = create_for_bitpacked_array(values).unwrap();
    let mask = (0..LENGTH)
        .map(|_| rng.random_bool(fraction_kept))
        .collect::<BooleanBuffer>();

    bencher
        .with_inputs(|| Mask::from_buffer(mask.clone()))
        .bench_local_values(|mask| {
            export_canonical_pipeline_expr(
                array.dtype(),
                array.len(),
                array.to_operator().unwrap().unwrap().as_ref(),
                &mask,
            )
            .unwrap()
        });
}
