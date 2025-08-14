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
use vortex_array::pipeline::canonical::{
    export_canonical_pipeline, export_canonical_pipeline_expr,
};
use vortex_array::pipeline::query::Pipeline;
use vortex_array::pipeline::types::Element;
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

fn generate_mask_with_runs(len: usize, fraction_kept: f64, rng: &mut StdRng) -> BooleanBuffer {
    if fraction_kept == 0.0 {
        // All false
        return BooleanBuffer::new_unset(len);
    } else if fraction_kept == 1.0 {
        // All true
        return BooleanBuffer::new_set(len);
    }

    let mut mask = Vec::with_capacity(len);
    let mut current_idx = 0;

    // Generate runs of true and false
    // The average run length will be inversely proportional to how far we are from 0.5
    // Near 0.5: shorter runs (more alternation)
    // Near 0 or 1: longer runs (less alternation)
    let run_frequency = 1.0 - 2.0 * (fraction_kept - 0.5).abs();
    let avg_run_length = ((1.0 / run_frequency.max(0.01)) * 10.0).max(1.0) as usize;

    while current_idx < len {
        // Decide if this run should be true or false based on fraction_kept
        let is_true_run = rng.random_bool(fraction_kept);

        // Generate run length with some variability
        let run_length = (rng
            .random_range(1..=(avg_run_length * 2))
            .min(len - current_idx))
        .max(1);

        for _ in 0..run_length {
            if current_idx >= len {
                break;
            }
            mask.push(is_true_run);
            current_idx += 1;
        }
    }

    mask.into_iter().collect()
}

fn create_for_bitpacked_array<T: NativePType>(
    values: BufferMut<T>,
) -> vortex_error::VortexResult<vortex_array::ArrayRef> {
    let primitive_array = values.into_array().to_primitive().unwrap();

    // First apply FoR encoding
    let for_array = FoRArray::encode(primitive_array)?;

    // Then bitpack the residuals
    let residuals = for_array.encoded().to_primitive()?;
    let bitpacked = bitpack_to_best_bit_width(&residuals)?;

    // Create a new FoR array with bitpacked residuals
    Ok(
        FoRArray::try_new(bitpacked.into_array(), for_array.reference_scalar().clone())?
            .into_array(),
    )
}

#[divan::bench(types = [i64], args = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999])]
pub fn decompress_for_early_filter<T: NativePType>(bencher: Bencher, fraction_kept: f64) {
    let mut rng = StdRng::seed_from_u64(0);
    let values = (0..102_400)
        .map(|_| T::from(rng.random_range(50..150)).unwrap())
        .collect::<BufferMut<T>>();

    let array = create_for_bitpacked_array(values).unwrap();
    // let mask = generate_mask_with_runs(102_400, fraction_kept, &mut rng);
    let mask = (0..102_400)
        .map(|_| rng.random_bool(fraction_kept))
        .collect::<BooleanBuffer>();

    bencher
        .with_inputs(|| Mask::from_buffer(mask.clone()))
        .bench_local_values(|mask| {
            filter(array.as_ref(), &mask)
                .unwrap()
                .to_canonical()
                .unwrap()
        });
}

// #[divan::bench(types = [i16, i32, i64], args = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999])]
// pub fn decompress_for_late_filter<T: NativePType>(bencher: Bencher, fraction_kept: f64) {
//     let mut rng = StdRng::seed_from_u64(0);
//     let values = (0..102_400)
//         .map(|_| T::from(rng.random_range(50..150)).unwrap())
//         .collect::<BufferMut<T>>();
//
//     let array = create_for_bitpacked_array(values).unwrap();
//     let mask = generate_mask_with_runs(102_400, fraction_kept, &mut rng);
//
//     bencher
//         .with_inputs(|| Mask::from_buffer(mask.clone()))
//         .bench_values(|mask| filter(array.to_canonical().unwrap().as_ref(), &mask).unwrap());
// }

// Pipeline filter is commented out because FoR::to_pipeline is not yet implemented
#[divan::bench(types = [i64], args = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999])]
#[allow(dead_code)]
pub fn decompress_for_pipeline_plan_filter<T: Element + NativePType>(
    bencher: Bencher,
    fraction_kept: f64,
) {
    let len = 102_400;
    let mut rng = StdRng::seed_from_u64(0);
    let values = (0..len)
        .map(|_| T::from(rng.random_range(50..150)).unwrap())
        .collect::<BufferMut<T>>();

    let array = create_for_bitpacked_array(values).unwrap();
    // let mask = generate_mask_with_runs(100, fraction_kept, &mut rng);
    let mask = (0..len)
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
                array.to_pipeline_plan().unwrap().as_ref(),
                &mask,
            )
            .unwrap()
        });

    let result = export_canonical_pipeline_expr(
        array.dtype(),
        array.len(),
        array.to_pipeline_plan().unwrap().as_ref(),
        &Mask::from_buffer(mask.clone()),
    )
    .unwrap()
    .into_primitive()
    .unwrap();
    assert_eq!(result.len(), mask.count_set_bits());

    for i in 0..mask.count_set_bits() {
        assert_eq!(
            result.scalar_at(i).unwrap(),
            expect.scalar_at(i).unwrap(),
            "{}, {}",
            i,
            fraction_kept
        );
    }
}

#[divan::bench(types = [i64], args = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999])]
#[allow(dead_code)]
pub fn decompress_for_pipeline_filter<T: Element + NativePType>(
    bencher: Bencher,
    fraction_kept: f64,
) {
    let len = 102_400;
    let mut rng = StdRng::seed_from_u64(0);
    let values = (0..len)
        .map(|_| T::from(rng.random_range(50..150)).unwrap())
        .collect::<BufferMut<T>>();

    let array = create_for_bitpacked_array(values).unwrap();
    // let mask = generate_mask_with_runs(100, fraction_kept, &mut rng);
    let mask = (0..len)
        .map(|_| rng.random_bool(fraction_kept))
        .collect::<BooleanBuffer>();

    let expect = filter(
        array.to_canonical().unwrap().as_ref(),
        &Mask::from_buffer(mask.clone()),
    )
    .unwrap();

    let plan = array.to_pipeline_plan().unwrap();

    bencher
        .with_inputs(|| {
            (
                Mask::from_buffer(mask.clone()),
                Pipeline::new(plan.as_ref()).unwrap(),
            )
        })
        .bench_local_values(|(mask, mut pipeline)| {
            export_canonical_pipeline(array.dtype(), array.len(), &mut pipeline, &mask).unwrap()
        });

    let result = export_canonical_pipeline_expr(
        array.dtype(),
        array.len(),
        array.to_pipeline_plan().unwrap().as_ref(),
        &Mask::from_buffer(mask.clone()),
    )
    .unwrap()
    .into_primitive()
    .unwrap();
    assert_eq!(result.len(), mask.count_set_bits());

    for i in 0..mask.count_set_bits() {
        assert_eq!(
            result.scalar_at(i).unwrap(),
            expect.scalar_at(i).unwrap(),
            "{}, {}",
            i,
            fraction_kept
        );
    }
}
