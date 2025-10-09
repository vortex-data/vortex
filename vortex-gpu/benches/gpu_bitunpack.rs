// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use std::sync::Arc;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use cudarc::driver::CudaContext;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use vortex_array::{IntoArray, ToCanonical};
use vortex_buffer::BufferMut;
use vortex_dtype::NativePType;
use vortex_error::VortexUnwrap;
use vortex_fastlanes::{BitPackedArray, FoRArray};
use vortex_gpu::{cuda_bit_unpack_timed, cuda_for_bp_unpack_timed, cuda_for_unpack_timed};

// Data sizes: 1GB, 2.5GB, 5GB, 10GB
// These are approximate sizes in bytes, accounting for bit-packing compression
const DATA_SIZES: &[(usize, &str)] = &[
    (268_435_456, "1GB"), // ~1GB when unpacked (268M * 4 bytes)
    // (671_088_640, "2.5GB"),  // ~2.5GB when unpacked
    // (1_342_177_280, "5GB"),  // ~5GB when unpacked
    (2_684_354_560, "10GB"), // ~10GB when unpacked
];

/// Creates a bitpackable dataset of the given size.
/// Values are chosen to fit in 6 bits (0-63) to ensure no patches are needed.
fn make_bitpackable_array<T: NativePType>(len: usize) -> BitPackedArray {
    let mut rng = StdRng::seed_from_u64(42);
    // Generate values that fit in 6 bits (0-63)
    let values = (0..len)
        .map(|_| T::from(rng.random_range(0..64)).unwrap())
        .collect::<BufferMut<T>>()
        .into_array()
        .to_primitive();

    // Encode with 6-bit width, which will not need patches
    BitPackedArray::encode(values.as_ref(), 6).unwrap()
}

/// Creates a FoR bitpackable dataset of the given size.
/// Values are chosen to fit in 6 bits after subtracting the reference value.
fn make_for_bitpackable_array(len: usize) -> FoRArray {
    let mut rng = StdRng::seed_from_u64(42);
    let reference = 100u32;

    // Generate values that fit in 6 bits (0-63)
    let values = (0..len)
        .map(|_| rng.random_range(0..64))
        .collect::<BufferMut<u32>>()
        .into_array()
        .to_primitive();

    // Create bitpacked array first
    let bitpacked = BitPackedArray::encode(values.as_ref(), 6).unwrap();

    // Wrap in FoR encoding with reference value
    FoRArray::try_new(bitpacked.into_array(), reference.into()).vortex_unwrap()
}

fn benchmark_gpu_decompress_kernel_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_decompress_kernel_only");

    group.sample_size(10);

    // Initialize CUDA context once

    for (len, label) in DATA_SIZES {
        let len = len.next_multiple_of(1024);
        let array = make_bitpackable_array::<u32>(len);

        let ctx = CudaContext::new(0).unwrap();
        ctx.set_blocking_synchronize().unwrap();
        let ctx = Arc::new(ctx);

        group.throughput(Throughput::Bytes((len * size_of::<u32>()) as u64));
        group.bench_with_input(BenchmarkId::new("u32", label), &array, |b, array| {
            b.iter_custom(|iters| {
                let mut total_time = Duration::ZERO;
                for _ in 0..iters {
                    // This only measures kernel execution time, not memory transfers
                    let kernel_time_ns = cuda_bit_unpack_timed(array, Arc::clone(&ctx)).unwrap();
                    total_time += kernel_time_ns;
                }
                total_time
            });
        });
    }

    group.finish();
}

fn benchmark_gpu_for_decompress_kernel_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_for_decompress_kernel_only");

    group.sample_size(10);

    for (len, label) in DATA_SIZES {
        let len = len.next_multiple_of(1024);
        let array = make_for_bitpackable_array(len);

        let ctx = CudaContext::new(0).unwrap();
        ctx.set_blocking_synchronize().unwrap();
        let ctx = Arc::new(ctx);

        group.throughput(Throughput::Bytes((len * size_of::<u32>()) as u64));
        group.bench_with_input(BenchmarkId::new("u32", label), &array, |b, array| {
            b.iter_custom(|iters| {
                let mut total_time = Duration::ZERO;
                for _ in 0..iters {
                    // This only measures kernel execution time, not memory transfers
                    let (_result, kernel_time) =
                        cuda_for_unpack_timed(array, Arc::clone(&ctx)).unwrap();
                    total_time += kernel_time;
                }
                total_time
            });
        });
    }

    group.finish();
}

fn benchmark_gpu_for_bp_fused_decompress_kernel_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_for_bp_fused_decompress_kernel_only");

    group.sample_size(10);

    for (len, label) in DATA_SIZES {
        let len = len.next_multiple_of(1024);
        let array = make_for_bitpackable_array(len);

        let ctx = CudaContext::new(0).unwrap();
        ctx.set_blocking_synchronize().unwrap();
        let ctx = Arc::new(ctx);

        group.throughput(Throughput::Bytes((len * size_of::<u32>()) as u64));
        group.bench_with_input(BenchmarkId::new("u32", label), &array, |b, array| {
            b.iter_custom(|iters| {
                let mut total_time = Duration::ZERO;
                for _ in 0..iters {
                    // This only measures kernel execution time, not memory transfers
                    let (_result, kernel_time) =
                        cuda_for_bp_unpack_timed(array, Arc::clone(&ctx)).unwrap();
                    total_time += kernel_time;
                }
                total_time
            });
        });
    }

    group.finish();
}

#[allow(dead_code)]
fn benchmark_cpu_canonicalize(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_canonicalize");

    for (len, label) in DATA_SIZES {
        let len = len.next_multiple_of(1024);
        let array = make_bitpackable_array::<u32>(len);

        group.throughput(Throughput::Bytes((len * size_of::<u32>()) as u64));
        group.bench_with_input(BenchmarkId::new("u32", label), &array, |b, array| {
            b.iter(|| array.clone().into_array().to_canonical());
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_gpu_decompress_kernel_only,
    benchmark_gpu_for_decompress_kernel_only,
    benchmark_gpu_for_bp_fused_decompress_kernel_only
);
criterion_main!(benches);
