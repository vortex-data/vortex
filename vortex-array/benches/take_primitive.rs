// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for primitive take kernels.
//!
//! Compares the auto-vectorized `PrimitiveArray::take` against a simple scalar gather
//! baseline, across different value types, array sizes, and access patterns.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::unwrap_used)]

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

// ---------------------------------------------------------------------------
// A simple scalar gather baseline for comparison.
// These are intentionally not auto-vectorized: `#[inline(never)]` prevents
// the compiler from applying SIMD optimizations, giving us a clean scalar
// baseline to measure the speedup from the multiversion kernel.
// ---------------------------------------------------------------------------

#[inline(never)]
fn scalar_gather_i32(values: &[i32], indices: &[u32]) -> Vec<i32> {
    let mut result = Vec::with_capacity(indices.len());
    for &idx in indices {
        result.push(values[idx as usize]);
    }
    result
}

#[inline(never)]
fn scalar_gather_f64(values: &[f64], indices: &[u32]) -> Vec<f64> {
    let mut result = Vec::with_capacity(indices.len());
    for &idx in indices {
        result.push(values[idx as usize]);
    }
    result
}

#[inline(never)]
fn scalar_gather_u64(values: &[u64], indices: &[u32]) -> Vec<u64> {
    let mut result = Vec::with_capacity(indices.len());
    for &idx in indices {
        result.push(values[idx as usize]);
    }
    result
}

// ---------------------------------------------------------------------------
// PrimitiveArray::take (auto-vectorized kernel) benchmarks
// ---------------------------------------------------------------------------

// --- i32 values, u32 indices ---

#[divan::bench(args = NUM_INDICES, consts = VECTOR_SIZE, sample_count = 10_000)]
fn take_i32_u32_uniform<const NUM_VALUES: usize>(bencher: Bencher, num_indices: usize) {
    let values = PrimitiveArray::from_iter(0..NUM_VALUES as i32);
    let rng = StdRng::seed_from_u64(42);
    let range = Uniform::new(0u32, NUM_VALUES as u32).unwrap();
    let indices = PrimitiveArray::from_iter(rng.sample_iter(range).take(num_indices));

    bencher.bench(|| {
        values
            .take(std::hint::black_box(indices.clone().into_array()))
            .unwrap()
            .into_canonical()
            .unwrap()
    });
}

#[divan::bench(args = NUM_INDICES, consts = VECTOR_SIZE, sample_count = 10_000)]
fn scalar_i32_u32_uniform<const NUM_VALUES: usize>(bencher: Bencher, num_indices: usize) {
    let values: Vec<i32> = (0..NUM_VALUES as i32).collect();
    let rng = StdRng::seed_from_u64(42);
    let range = Uniform::new(0u32, NUM_VALUES as u32).unwrap();
    let indices: Vec<u32> = rng.sample_iter(range).take(num_indices).collect();

    bencher
        .with_inputs(|| (&values, &indices))
        .bench_local_values(|(v, i)| scalar_gather_i32(v, i));
}

// --- i32 values, u16 indices ---

#[divan::bench(args = NUM_INDICES, consts = VECTOR_SIZE, sample_count = 10_000)]
fn take_i32_u16_uniform<const NUM_VALUES: usize>(bencher: Bencher, num_indices: usize) {
    let values = PrimitiveArray::from_iter(0..NUM_VALUES as i32);
    let rng = StdRng::seed_from_u64(42);
    let range = Uniform::new(0u16, NUM_VALUES as u16).unwrap();
    let indices = PrimitiveArray::from_iter(rng.sample_iter(range).take(num_indices));

    bencher.bench(|| {
        values
            .take(std::hint::black_box(indices.clone().into_array()))
            .unwrap()
            .into_canonical()
            .unwrap()
    });
}

// --- f64 values, u32 indices ---

#[divan::bench(args = NUM_INDICES, consts = VECTOR_SIZE, sample_count = 10_000)]
fn take_f64_u32_uniform<const NUM_VALUES: usize>(bencher: Bencher, num_indices: usize) {
    let values = PrimitiveArray::from_iter((0..NUM_VALUES).map(|i| i as f64 * 1.5));
    let rng = StdRng::seed_from_u64(42);
    let range = Uniform::new(0u32, NUM_VALUES as u32).unwrap();
    let indices = PrimitiveArray::from_iter(rng.sample_iter(range).take(num_indices));

    bencher.bench(|| {
        values
            .take(std::hint::black_box(indices.clone().into_array()))
            .unwrap()
            .into_canonical()
            .unwrap()
    });
}

#[divan::bench(args = NUM_INDICES, consts = VECTOR_SIZE, sample_count = 10_000)]
fn scalar_f64_u32_uniform<const NUM_VALUES: usize>(bencher: Bencher, num_indices: usize) {
    let values: Vec<f64> = (0..NUM_VALUES).map(|i| i as f64 * 1.5).collect();
    let rng = StdRng::seed_from_u64(42);
    let range = Uniform::new(0u32, NUM_VALUES as u32).unwrap();
    let indices: Vec<u32> = rng.sample_iter(range).take(num_indices).collect();

    bencher
        .with_inputs(|| (&values, &indices))
        .bench_local_values(|(v, i)| scalar_gather_f64(v, i));
}

// --- u64 values, u32 indices ---

#[divan::bench(args = NUM_INDICES, consts = VECTOR_SIZE, sample_count = 10_000)]
fn take_u64_u32_uniform<const NUM_VALUES: usize>(bencher: Bencher, num_indices: usize) {
    let values = PrimitiveArray::from_iter(0..NUM_VALUES as u64);
    let rng = StdRng::seed_from_u64(42);
    let range = Uniform::new(0u32, NUM_VALUES as u32).unwrap();
    let indices = PrimitiveArray::from_iter(rng.sample_iter(range).take(num_indices));

    bencher.bench(|| {
        values
            .take(std::hint::black_box(indices.clone().into_array()))
            .unwrap()
            .into_canonical()
            .unwrap()
    });
}

#[divan::bench(args = NUM_INDICES, consts = VECTOR_SIZE, sample_count = 10_000)]
fn scalar_u64_u32_uniform<const NUM_VALUES: usize>(bencher: Bencher, num_indices: usize) {
    let values: Vec<u64> = (0..NUM_VALUES as u64).collect();
    let rng = StdRng::seed_from_u64(42);
    let range = Uniform::new(0u32, NUM_VALUES as u32).unwrap();
    let indices: Vec<u32> = rng.sample_iter(range).take(num_indices).collect();

    bencher
        .with_inputs(|| (&values, &indices))
        .bench_local_values(|(v, i)| scalar_gather_u64(v, i));
}

// --- Zipfian distribution (skewed access, more cache-friendly) ---

#[divan::bench(args = NUM_INDICES, consts = VECTOR_SIZE, sample_count = 10_000)]
fn take_i32_u32_zipfian<const NUM_VALUES: usize>(bencher: Bencher, num_indices: usize) {
    let values = PrimitiveArray::from_iter(0..NUM_VALUES as i32);
    let rng = StdRng::seed_from_u64(42);
    let zipf = Zipf::new(NUM_VALUES as f64, 1.0).unwrap();
    let indices = PrimitiveArray::from_iter(
        rng.sample_iter(&zipf)
            .take(num_indices)
            .map(|i: f64| (i as u32 - 1).min(NUM_VALUES as u32 - 1)),
    );

    bencher.bench(|| {
        values
            .take(std::hint::black_box(indices.clone().into_array()))
            .unwrap()
            .into_canonical()
            .unwrap()
    });
}

#[divan::bench(args = NUM_INDICES, consts = VECTOR_SIZE, sample_count = 10_000)]
fn scalar_i32_u32_zipfian<const NUM_VALUES: usize>(bencher: Bencher, num_indices: usize) {
    let values: Vec<i32> = (0..NUM_VALUES as i32).collect();
    let rng = StdRng::seed_from_u64(42);
    let zipf = Zipf::new(NUM_VALUES as f64, 1.0).unwrap();
    let indices: Vec<u32> = rng
        .sample_iter(&zipf)
        .take(num_indices)
        .map(|i: f64| (i as u32 - 1).min(NUM_VALUES as u32 - 1))
        .collect();

    bencher
        .with_inputs(|| (&values, &indices))
        .bench_local_values(|(v, i)| scalar_gather_i32(v, i));
}

// --- Sequential (best-case for cache) ---

#[divan::bench(args = NUM_INDICES, consts = VECTOR_SIZE, sample_count = 10_000)]
fn take_i32_u32_sequential<const NUM_VALUES: usize>(bencher: Bencher, num_indices: usize) {
    let values = PrimitiveArray::from_iter(0..NUM_VALUES as i32);
    let indices =
        PrimitiveArray::from_iter((0..num_indices).map(|i| (i % NUM_VALUES) as u32));

    bencher.bench(|| {
        values
            .take(std::hint::black_box(indices.clone().into_array()))
            .unwrap()
            .into_canonical()
            .unwrap()
    });
}

#[divan::bench(args = NUM_INDICES, consts = VECTOR_SIZE, sample_count = 10_000)]
fn scalar_i32_u32_sequential<const NUM_VALUES: usize>(bencher: Bencher, num_indices: usize) {
    let values: Vec<i32> = (0..NUM_VALUES as i32).collect();
    let indices: Vec<u32> = (0..num_indices).map(|i| (i % NUM_VALUES) as u32).collect();

    bencher
        .with_inputs(|| (&values, &indices))
        .bench_local_values(|(v, i)| scalar_gather_i32(v, i));
}

// --- DictArray canonicalization benchmarks (existing) ---

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
