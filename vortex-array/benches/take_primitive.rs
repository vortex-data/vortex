// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for primitive take kernels.
//!
//! Compares three kernel implementations head-to-head on raw `&[V]` → `Buffer<V>`:
//! - **autovec**: auto-vectorized via `multiversion` (AVX-512 masked gather / AVX2 cmov)
//! - **avx2**: hand-written AVX2 intrinsics with explicit `vpgatherdd`
//! - **scalar**: simple loop with bounds-checked indexing
//!
//! All three operate on the same input slices so the only variable is the gather loop.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::unwrap_used)]

use divan::Bencher;
use rand::distr::Uniform;
use rand::prelude::*;
use rand_distr::Zipf;
use vortex_array::IntoArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::take_kernels;

fn main() {
    divan::main();
}

/// Number of indices to take.
const NUM_INDICES: &[usize] = &[1_000, 10_000, 100_000];

/// Size of the source vector / dictionary values.
const VECTOR_SIZE: &[usize] = &[256, 8192, 65536];

// ===========================================================================
// i32 values, u32 indices — uniform random
// ===========================================================================

#[divan::bench(args = NUM_INDICES, consts = VECTOR_SIZE, sample_count = 10_000)]
fn autovec_i32_u32_uniform<const N: usize>(bencher: Bencher, num_indices: usize) {
    let values: Vec<i32> = (0..N as i32).collect();
    let indices: Vec<u32> = StdRng::seed_from_u64(42)
        .sample_iter(Uniform::new(0u32, N as u32).unwrap())
        .take(num_indices)
        .collect();
    bencher
        .with_inputs(|| (&values, &indices))
        .bench_local_values(|(v, i)| take_kernels::autovec::<i32, u32>(v, i));
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[divan::bench(args = NUM_INDICES, consts = VECTOR_SIZE, sample_count = 10_000)]
fn avx2_i32_u32_uniform<const N: usize>(bencher: Bencher, num_indices: usize) {
    let values: Vec<i32> = (0..N as i32).collect();
    let indices: Vec<u32> = StdRng::seed_from_u64(42)
        .sample_iter(Uniform::new(0u32, N as u32).unwrap())
        .take(num_indices)
        .collect();
    bencher
        .with_inputs(|| (&values, &indices))
        .bench_local_values(|(v, i)| take_kernels::avx2::<i32, u32>(v, i));
}

#[divan::bench(args = NUM_INDICES, consts = VECTOR_SIZE, sample_count = 10_000)]
fn scalar_i32_u32_uniform<const N: usize>(bencher: Bencher, num_indices: usize) {
    let values: Vec<i32> = (0..N as i32).collect();
    let indices: Vec<u32> = StdRng::seed_from_u64(42)
        .sample_iter(Uniform::new(0u32, N as u32).unwrap())
        .take(num_indices)
        .collect();
    bencher
        .with_inputs(|| (&values, &indices))
        .bench_local_values(|(v, i)| take_kernels::scalar::<i32, u32>(v, i));
}

// ===========================================================================
// f64 values, u32 indices — uniform random
// ===========================================================================

#[divan::bench(args = NUM_INDICES, consts = VECTOR_SIZE, sample_count = 10_000)]
fn autovec_f64_u32_uniform<const N: usize>(bencher: Bencher, num_indices: usize) {
    let values: Vec<f64> = (0..N).map(|i| i as f64 * 1.5).collect();
    let indices: Vec<u32> = StdRng::seed_from_u64(42)
        .sample_iter(Uniform::new(0u32, N as u32).unwrap())
        .take(num_indices)
        .collect();
    bencher
        .with_inputs(|| (&values, &indices))
        .bench_local_values(|(v, i)| take_kernels::autovec::<f64, u32>(v, i));
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[divan::bench(args = NUM_INDICES, consts = VECTOR_SIZE, sample_count = 10_000)]
fn avx2_f64_u32_uniform<const N: usize>(bencher: Bencher, num_indices: usize) {
    let values: Vec<f64> = (0..N).map(|i| i as f64 * 1.5).collect();
    let indices: Vec<u32> = StdRng::seed_from_u64(42)
        .sample_iter(Uniform::new(0u32, N as u32).unwrap())
        .take(num_indices)
        .collect();
    bencher
        .with_inputs(|| (&values, &indices))
        .bench_local_values(|(v, i)| take_kernels::avx2::<f64, u32>(v, i));
}

#[divan::bench(args = NUM_INDICES, consts = VECTOR_SIZE, sample_count = 10_000)]
fn scalar_f64_u32_uniform<const N: usize>(bencher: Bencher, num_indices: usize) {
    let values: Vec<f64> = (0..N).map(|i| i as f64 * 1.5).collect();
    let indices: Vec<u32> = StdRng::seed_from_u64(42)
        .sample_iter(Uniform::new(0u32, N as u32).unwrap())
        .take(num_indices)
        .collect();
    bencher
        .with_inputs(|| (&values, &indices))
        .bench_local_values(|(v, i)| take_kernels::scalar::<f64, u32>(v, i));
}

// ===========================================================================
// i32 values, u32 indices — zipfian (skewed, cache-friendly)
// ===========================================================================

#[divan::bench(args = NUM_INDICES, consts = VECTOR_SIZE, sample_count = 10_000)]
fn autovec_i32_u32_zipfian<const N: usize>(bencher: Bencher, num_indices: usize) {
    let values: Vec<i32> = (0..N as i32).collect();
    let zipf = Zipf::new(N as f64, 1.0).unwrap();
    let indices: Vec<u32> = StdRng::seed_from_u64(42)
        .sample_iter(&zipf)
        .take(num_indices)
        .map(|i: f64| (i as u32 - 1).min(N as u32 - 1))
        .collect();
    bencher
        .with_inputs(|| (&values, &indices))
        .bench_local_values(|(v, i)| take_kernels::autovec::<i32, u32>(v, i));
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[divan::bench(args = NUM_INDICES, consts = VECTOR_SIZE, sample_count = 10_000)]
fn avx2_i32_u32_zipfian<const N: usize>(bencher: Bencher, num_indices: usize) {
    let values: Vec<i32> = (0..N as i32).collect();
    let zipf = Zipf::new(N as f64, 1.0).unwrap();
    let indices: Vec<u32> = StdRng::seed_from_u64(42)
        .sample_iter(&zipf)
        .take(num_indices)
        .map(|i: f64| (i as u32 - 1).min(N as u32 - 1))
        .collect();
    bencher
        .with_inputs(|| (&values, &indices))
        .bench_local_values(|(v, i)| take_kernels::avx2::<i32, u32>(v, i));
}

#[divan::bench(args = NUM_INDICES, consts = VECTOR_SIZE, sample_count = 10_000)]
fn scalar_i32_u32_zipfian<const N: usize>(bencher: Bencher, num_indices: usize) {
    let values: Vec<i32> = (0..N as i32).collect();
    let zipf = Zipf::new(N as f64, 1.0).unwrap();
    let indices: Vec<u32> = StdRng::seed_from_u64(42)
        .sample_iter(&zipf)
        .take(num_indices)
        .map(|i: f64| (i as u32 - 1).min(N as u32 - 1))
        .collect();
    bencher
        .with_inputs(|| (&values, &indices))
        .bench_local_values(|(v, i)| take_kernels::scalar::<i32, u32>(v, i));
}

// ===========================================================================
// DictArray canonicalization (exercises the full take pipeline end-to-end)
// ===========================================================================

const DICT_VECTOR_SIZE: &[usize] = &[16, 256, 2048, 8192];

#[divan::bench(args = NUM_INDICES, consts = DICT_VECTOR_SIZE, sample_count = 100_000)]
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

#[divan::bench(args = NUM_INDICES, consts = DICT_VECTOR_SIZE, sample_count = 100_000)]
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
