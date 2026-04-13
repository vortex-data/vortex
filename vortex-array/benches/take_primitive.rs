// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks comparing [`PVector`] take vs [`DictArray`] canonicalization.
//!
//! Both are tracked by number of indices/codes for fair comparison.

#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::unwrap_used)]

use divan::Bencher;
use rand::distr::Uniform;
use rand::prelude::*;
use rand_distr::Zipf;
use vortex_array::IntoArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;

fn main() {
    divan::main();
}

/// Number of indices to take.
const NUM_INDICES: &[usize] = &[1_000, 10_000, 100_000];

/// Size of the source vector / dictionary values.
const VECTOR_SIZE: &[usize] = &[16, 256, 2048, 8192];

// --- DictArray canonicalization benchmarks ---

#[divan::bench(args = NUM_INDICES, consts = VECTOR_SIZE, sample_count = 100_000)]
fn dict_canonicalize_uniform<const NUM_VALUES: usize>(bencher: Bencher, num_indices: usize) {
    let values = PrimitiveArray::from_iter(0..NUM_VALUES as u32);

    let rng = StdRng::seed_from_u64(0);
    let range = Uniform::new(0u32, NUM_VALUES as u32).unwrap();
    let codes = PrimitiveArray::from_iter(rng.sample_iter(range).take(num_indices));

    let dict = DictArray::try_new(codes.into_array(), values.into_array()).unwrap();

    bencher
        .with_inputs(|| &dict)
        .bench_refs(|dict| dict.to_canonical());
}

#[divan::bench(args = NUM_INDICES, consts = VECTOR_SIZE, sample_count = 100_000)]
fn dict_canonicalize_zipfian<const NUM_VALUES: usize>(bencher: Bencher, num_indices: usize) {
    let values = PrimitiveArray::from_iter(0..NUM_VALUES as u32);

    let rng = StdRng::seed_from_u64(0);
    let zipf = Zipf::new(NUM_VALUES as f64, 1.0).unwrap();
    let codes = PrimitiveArray::from_iter(
        rng.sample_iter(&zipf)
            .take(num_indices)
            .map(|i: f64| (i as u32 - 1).min(NUM_VALUES as u32 - 1)),
    );

    let dict = DictArray::try_new(codes.into_array(), values.into_array()).unwrap();

    bencher
        .with_inputs(|| &dict)
        .bench_refs(|dict| dict.to_canonical());
}
