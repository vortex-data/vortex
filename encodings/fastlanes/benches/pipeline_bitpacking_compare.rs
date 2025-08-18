// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
#![allow(unexpected_cfgs)]

use arrow_buffer::BooleanBuffer;
use divan::Bencher;
use mimalloc::MiMalloc;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use vortex_array::compute::{Operator, compare, filter};
use vortex_array::pipeline::canonical::export_canonical_pipeline_expr;
use vortex_array::{Array, IntoArray, ToCanonical};
use vortex_buffer::BufferMut;
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{DType, NativePType};
use vortex_expr::{Scope, VortexExprExt, gt, root};
use vortex_fastlanes::bitpack_to_best_bit_width;
use vortex_mask::Mask;
use vortex_vector::types::Element;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub fn main() {
    divan::main();
}

const TRUE_COUNT: &[f64] = &[
    0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.00,
];

#[divan::bench(types = [i8, i16, i32, i64], args = TRUE_COUNT)]
pub fn early_filter<T: NativePType>(bencher: Bencher, fraction_kept: f64) {
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

    let expr = gt(root(), root());

    bencher
        // Be sure to reconstruct the mask to avoid cached set_indices
        .with_inputs(|| Mask::from_buffer(mask.clone()))
        .bench_local_values(|mask| {
            // We run the filter first, then compare.
            let array = filter(array.as_ref(), &mask).unwrap();
            expr.evaluate(&Scope::new(array))
                .unwrap()
                .to_canonical()
                .unwrap()
        });
}

#[divan::bench(types = [i8, i16, i32, i64], args = TRUE_COUNT)]
pub fn late_filter<T: NativePType>(bencher: Bencher, fraction_kept: f64) {
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
        .bench_values(|mask| {
            // We run the compute first, then filter.
            filter(
                &compare(array.as_ref(), array.as_ref(), Operator::Gt).unwrap(),
                &mask,
            )
            .unwrap()
        });
}

#[divan::bench(types = [i8, i16, i32, i64], args = TRUE_COUNT)]
pub fn pipeline<T: Element + NativePType>(bencher: Bencher, fraction_kept: f64) {
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

    let expr = gt(root(), root());

    let operator = expr.to_operator(array.as_ref()).unwrap().unwrap();

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
