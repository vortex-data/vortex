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
use vortex_array::pipeline::canonical::{export_canonical, export_canonical_pipeline};
use vortex_array::pipeline::operators::compare::CompareOperator;
use vortex_array::pipeline::types::Element;
use vortex_array::{Array, IntoArray, ToCanonical};
use vortex_buffer::BufferMut;
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{DType, NativePType};
use vortex_expr::{Scope, root};
use vortex_fastlanes::bitpack_to_best_bit_width;
use vortex_mask::Mask;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub fn main() {
    divan::main();
}

const TRUE_COUNT: &[f64] = &[
    0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.00,
];
//
// const TRUE_COUNT: &[f64] = &[1.];

#[divan::bench(types = [ i64], args = TRUE_COUNT)]
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

    let expr = vortex_expr::gt(root(), root());

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

#[divan::bench(types = [i64], args = TRUE_COUNT)]
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
//
// #[divan::bench(types = [i8, i16, i32, i64], args = [0.005, 0.01, 0.025, 0.05])]
// pub fn fused<T: Element + NativePType>(bencher: Bencher, fraction_kept: f64) {
//     let mut rng = StdRng::seed_from_u64(0);
//     let values = (0..100_000)
//         .map(|_| T::from(rng.random_range(0..100)).unwrap())
//         .collect::<BufferMut<T>>()
//         .into_array()
//         .to_primitive()
//         .unwrap();
//     let array = bitpack_to_best_bit_width(&values).unwrap();
//
//     let mask = (0..100_000)
//         .map(|_| rng.random_bool(fraction_kept))
//         .collect::<BooleanBuffer>();
//
//     bencher
//         .with_inputs(|| Mask::from_buffer(mask.clone()))
//         .bench_local_values(|mask| export_canonical(array.as_ref(), &mask).unwrap());
//
//     let array = export_canonical(array.as_ref(), &Mask::from_buffer(mask.clone()))
//         .unwrap()
//         .into_primitive()
//         .unwrap();
//     assert_eq!(array.len(), mask.count_set_bits());
// }

#[divan::bench(types = [i64], args = TRUE_COUNT)]
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

    let expr1 = array.to_pipeline_plan().unwrap();
    let expr2 = array.to_pipeline_plan().unwrap();
    let expr = CompareOperator::new(expr1, expr2, Operator::Gt);

    bencher
        .with_inputs(|| Mask::from_buffer(mask.clone()))
        .bench_local_values(|mask| {
            export_canonical_pipeline(&DType::Bool(NonNullable), array.len(), &expr, &mask).unwrap()
        });
}
