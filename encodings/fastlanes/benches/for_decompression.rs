// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for FoR (Frame-of-Reference) decompression throughput.
//!
//! These benchmarks measure pure CPU decompression performance for comparison
//! with GPU-accelerated implementations (vortex-metal, vortex-cuda).

#![allow(clippy::cast_possible_truncation)]

use std::mem::size_of;
use std::ops::Add;

use divan::Bencher;
use divan::counter::BytesCount;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_fastlanes::FoRArray;

fn main() {
    divan::main();
}

const REFERENCE_VALUE: u8 = 10;

/// Array sizes to benchmark - matching the Metal benchmark for comparison.
const BENCH_SIZES: &[(usize, &str)] = &[(100_000, "100K"), (1_000_000, "1M"), (10_000_000, "10M")];

/// Creates a FoR array with the specified type and length.
fn make_for_array<T>(len: usize) -> FoRArray
where
    T: NativePType + From<u8> + Add<Output = T>,
    Scalar: From<T>,
{
    let reference = <T as From<u8>>::from(REFERENCE_VALUE);
    let data: Vec<T> = (0..len)
        .map(|i| <T as From<u8>>::from((i % 256) as u8))
        .collect();

    let primitive_array =
        PrimitiveArray::new(Buffer::from(data), Validity::NonNullable).into_array();

    FoRArray::try_new(primitive_array, reference.into()).vortex_expect("failed to create FoR array")
}

// --- u32 benchmarks ---

#[divan::bench(args = BENCH_SIZES)]
fn for_decompress_u32(bencher: Bencher, (len, _name): (usize, &str)) {
    let for_array = make_for_array::<u32>(len);

    bencher
        .counter(BytesCount::new(len * size_of::<u32>()))
        .with_inputs(|| &for_array)
        .bench_refs(|arr| arr.to_canonical());
}

// --- u64 benchmarks ---

#[divan::bench(args = BENCH_SIZES)]
fn for_decompress_u64(bencher: Bencher, (len, _name): (usize, &str)) {
    let for_array = make_for_array::<u64>(len);

    bencher
        .counter(BytesCount::new(len * size_of::<u64>()))
        .with_inputs(|| &for_array)
        .bench_refs(|arr| arr.to_canonical());
}

// --- i32 benchmarks ---

#[divan::bench(args = BENCH_SIZES)]
fn for_decompress_i32(bencher: Bencher, (len, _name): (usize, &str)) {
    let for_array = make_for_array::<i32>(len);

    bencher
        .counter(BytesCount::new(len * size_of::<i32>()))
        .with_inputs(|| &for_array)
        .bench_refs(|arr| arr.to_canonical());
}

// --- i64 benchmarks ---

#[divan::bench(args = BENCH_SIZES)]
fn for_decompress_i64(bencher: Bencher, (len, _name): (usize, &str)) {
    let for_array = make_for_array::<i64>(len);

    bencher
        .counter(BytesCount::new(len * size_of::<i64>()))
        .with_inputs(|| &for_array)
        .bench_refs(|arr| arr.to_canonical());
}
