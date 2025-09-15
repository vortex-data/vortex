// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
#![allow(unexpected_cfgs)]

use arrow_buffer::BooleanBuffer;
use divan::Bencher;
use mimalloc::MiMalloc;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use vortex_array::compute::{filter, warm_up_vtables};
use vortex_array::pipeline::{Element, export_canonical_pipeline_expr};
use vortex_array::{Array, ArrayRef, IntoArray, ToCanonical};
use vortex_buffer::BufferMut;
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{DType, NativePType};
use vortex_error::VortexResult;
use vortex_expr::{Scope, VortexExprExt, lit, lt, root};
use vortex_fastlanes::{FoRArray, bitpack_to_best_bit_width};
use vortex_mask::Mask;
use vortex_scalar::Scalar;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub fn main() {
    warm_up_vtables();
    divan::main();
}

const TRUE_COUNT: &[f64] = &[
    0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.00,
];

fn create_for_bitpacked_array<T: NativePType>(values: BufferMut<T>) -> VortexResult<ArrayRef> {
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

#[divan::bench(types = [u8, u16, u32, u64], args = TRUE_COUNT)]
pub fn eval<T: NativePType + Into<Scalar>>(bencher: Bencher, fraction_kept: f64) {
    let mut rng = StdRng::seed_from_u64(0);
    let values = (0..100_000)
        .map(|_| T::from(rng.random_range(10..100)).unwrap())
        .collect::<BufferMut<T>>();
    let array = create_for_bitpacked_array(values).unwrap();

    let mask = (0..100_000)
        .map(|_| rng.random_bool(fraction_kept))
        .collect::<BooleanBuffer>();

    let expr = lt(root(), lit(T::from_i32(2).unwrap()));

    bencher
        // Be sure to reconstruct the mask to avoid cached set_indices
        .with_inputs(|| (Mask::from_buffer(mask.clone()), array.clone()))
        .bench_local_values(|(mask, array)| {
            // We run the filter first, then compare.
            let array = filter(array.as_ref(), &mask).unwrap();
            expr.evaluate(&Scope::new(array)).unwrap().to_canonical()
        });
}

#[divan::bench(types = [u8, u16, u32, u64], args = TRUE_COUNT)]
pub fn pipeline<T: Element + NativePType + Into<Scalar>>(bencher: Bencher, fraction_kept: f64) {
    let mut rng = StdRng::seed_from_u64(0);
    let values = (0..100_000)
        .map(|_| T::from(rng.random_range(10..100)).unwrap())
        .collect::<BufferMut<T>>();
    let array = create_for_bitpacked_array(values).unwrap();

    let mask = (0..100_000)
        .map(|_| rng.random_bool(fraction_kept))
        .collect::<BooleanBuffer>();

    let expr = lt(root(), lit(T::from_i32(2).unwrap()));
    let operator = expr.to_operator_unoptimized(&array).unwrap().unwrap();

    bencher
        .with_inputs(|| Mask::from_buffer(mask.clone()))
        .bench_local_values(|mask| {
            export_canonical_pipeline_expr(
                &DType::Bool(NonNullable),
                array.len(),
                operator.as_ref(),
                &mask,
            )
            .unwrap()
        });
}

#[divan::bench(types = [u8, u16, u32, u64], args = TRUE_COUNT)]
pub fn pipeline_opt<T: Element + NativePType + Into<Scalar>>(bencher: Bencher, fraction_kept: f64) {
    let mut rng = StdRng::seed_from_u64(0);
    let values = (0..100_000)
        .map(|_| T::from(rng.random_range(10..100)).unwrap())
        .collect::<BufferMut<T>>();
    let array = create_for_bitpacked_array(values).unwrap();

    let mask = (0..100_000)
        .map(|_| rng.random_bool(fraction_kept))
        .collect::<BooleanBuffer>();

    let expr = lt(root(), lit(T::from_i32(2).unwrap()));
    let operator = expr.to_operator(&array).unwrap().unwrap();

    bencher
        .with_inputs(|| (Mask::from_buffer(mask.clone()), operator.clone()))
        .bench_local_values(|(mask, operator)| {
            export_canonical_pipeline_expr(
                &DType::Bool(NonNullable),
                array.len(),
                operator.as_ref(),
                &mask,
            )
            .unwrap()
        });
}
