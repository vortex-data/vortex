// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use divan::Bencher;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::IntoArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;

fn main() {
    divan::main();
}

/// Benchmark with many codes (65K) relative to 1024 values.
/// This tests performance when the values dictionary is small but many codes reference it.
#[divan::bench(args = [
    1024,    // Small dictionary
    2048,    // Medium dictionary
    4096,    // Larger dictionary
])]
fn bench_many_codes_few_values(bencher: Bencher, num_values: i32) {
    let mut rng = StdRng::seed_from_u64(0);

    let num_codes = 65_536;

    // Create values array with the specified number of unique values
    let values = PrimitiveArray::from_iter(0..num_values).into_array();

    // Create codes that randomly reference the values
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let codes = PrimitiveArray::from_iter(
        (0..num_codes).map(|_| rng.random_range(0..num_values as usize) as u32),
    )
    .into_array();

    let array = DictArray::try_new(codes, values).unwrap();

    bencher
        .with_inputs(|| &array)
        .bench_refs(|array| array.compute_referenced_values_mask(false).unwrap());
}

/// Benchmark with many nulls in the codes array.
/// This tests performance when most codes are null and thus don't reference values.
#[divan::bench(args = [
    0.01,   // 1% valid codes
    0.1,    // 10% valid codes
    0.5,    // 50% valid codes
    0.9,    // 90% valid codes
])]
fn bench_many_nulls(bencher: Bencher, fraction_valid: f64) {
    let mut rng = StdRng::seed_from_u64(0);

    let num_codes = 65_536;
    let num_values = 1024i32;

    // Create values array
    let values = PrimitiveArray::from_iter(0..num_values).into_array();

    // Create codes with many nulls based on fraction_valid
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let codes = PrimitiveArray::from_option_iter((0..num_codes).map(|_| {
        rng.random_bool(fraction_valid)
            .then(|| rng.random_range(0..num_values as usize) as u32)
    }))
    .into_array();

    let array = DictArray::try_new(codes, values).unwrap();

    bencher
        .with_inputs(|| &array)
        .bench_refs(|array| array.compute_referenced_values_mask(false).unwrap());
}

/// Benchmark with sparse code coverage (many unreferenced values).
/// This tests when only a small subset of values are actually referenced.
#[divan::bench(args = [
    0.01,   // Only 1% of values are referenced
    0.1,    // 10% of values referenced
    0.5,    // 50% of values referenced
])]
fn bench_sparse_coverage(bencher: Bencher, fraction_coverage: f64) {
    let mut rng = StdRng::seed_from_u64(0);

    let num_codes = 65_536;
    let num_values = 1024i32;

    // Create values array
    let values = PrimitiveArray::from_iter(0..num_values).into_array();

    // Calculate how many unique values we'll actually reference
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let num_referenced = (num_values as f64 * fraction_coverage).max(1.0) as usize;

    // Create codes that only reference a subset of values
    #[allow(clippy::cast_possible_truncation)]
    let codes = PrimitiveArray::from_iter(
        (0..num_codes).map(|_| rng.random_range(0..num_referenced) as u32),
    )
    .into_array();

    let array = DictArray::try_new(codes, values).unwrap();

    bencher
        .with_inputs(|| &array)
        .bench_refs(|array| array.compute_referenced_values_mask(false).unwrap());
}
