// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks comparing [`PVector`] take vs [`DictArray`] canonicalization.
//!
//! Both are tracked by number of indices/codes for fair comparison.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::unwrap_used)]

use divan::Bencher;
use rand::distr::Uniform;
use rand::prelude::*;
use rand_distr::Zipf;
use vortex_array::IntoArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_buffer::Buffer;
use vortex_compute::take::Take;
use vortex_mask::Mask;
use vortex_vector::primitive::PVector;

fn main() {
    divan::main();
}

/// Number of indices/codes to process.
const NUM_INDICES: &[usize] = &[1_000, 10_000, 100_000];

// --- PVector take benchmarks ---
// Source vector is 1/10th the indices size (same as dict values).

#[divan::bench(args = NUM_INDICES, sample_count = 100_000)]
fn pvector_take_uniform(bencher: Bencher, num_indices: usize) {
    let vector_size = num_indices / 10;
    let data: Buffer<u32> = (0..vector_size as u32).collect();
    let pvector = PVector::new(data, Mask::AllTrue(vector_size));

    let rng = StdRng::seed_from_u64(0);
    let range = Uniform::new(0u32, vector_size as u32).unwrap();
    let indices: Vec<u32> = rng.sample_iter(range).take(num_indices).collect();

    bencher
        .with_inputs(|| (&pvector, indices.as_slice()))
        .bench_refs(|(pv, idx)| pv.take(*idx));
}

#[divan::bench(args = NUM_INDICES, sample_count = 100_000)]
fn pvector_take_zipfian(bencher: Bencher, num_indices: usize) {
    let vector_size = num_indices / 10;
    let data: Buffer<u32> = (0..vector_size as u32).collect();
    let pvector = PVector::new(data, Mask::AllTrue(vector_size));

    let rng = StdRng::seed_from_u64(0);
    let zipf = Zipf::new(vector_size as f64, 1.0).unwrap();
    let indices: Vec<u32> = rng
        .sample_iter(&zipf)
        .take(num_indices)
        .map(|i: f64| (i as u32 - 1).min(vector_size as u32 - 1))
        .collect();

    bencher
        .with_inputs(|| (&pvector, indices.as_slice()))
        .bench_refs(|(pv, idx)| pv.take(*idx));
}

// --- DictArray canonicalization benchmarks ---
// Dictionary has num_indices/10 unique values, num_indices codes.

#[divan::bench(args = NUM_INDICES, sample_count = 100_000)]
fn dict_canonicalize_uniform(bencher: Bencher, num_indices: usize) {
    let num_values = num_indices / 10;
    let values = PrimitiveArray::from_iter(0..num_values as u32);

    let rng = StdRng::seed_from_u64(0);
    let range = Uniform::new(0u32, num_values as u32).unwrap();
    let codes = PrimitiveArray::from_iter(rng.sample_iter(range).take(num_indices));

    let dict = DictArray::try_new(codes.into_array(), values.into_array()).unwrap();

    bencher
        .with_inputs(|| &dict)
        .bench_refs(|dict| dict.to_canonical());
}

#[divan::bench(args = NUM_INDICES, sample_count = 100_000)]
fn dict_canonicalize_zipfian(bencher: Bencher, num_indices: usize) {
    let num_values = num_indices / 10;
    let values = PrimitiveArray::from_iter(0..num_values as u32);

    let rng = StdRng::seed_from_u64(0);
    let zipf = Zipf::new(num_values as f64, 1.0).unwrap();
    let codes = PrimitiveArray::from_iter(
        rng.sample_iter(&zipf)
            .take(num_indices)
            .map(|i: f64| (i as u32 - 1).min(num_values as u32 - 1)),
    );

    let dict = DictArray::try_new(codes.into_array(), values.into_array()).unwrap();

    bencher
        .with_inputs(|| &dict)
        .bench_refs(|dict| dict.to_canonical());
}
