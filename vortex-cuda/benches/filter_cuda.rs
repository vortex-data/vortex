// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA benchmarks for filter operations.

#![allow(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]

use std::mem::size_of;
use std::time::Duration;
use std::time::Instant;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use futures::executor::block_on;
use vortex_array::IntoArray;
use vortex_array::arrays::FilterArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_cuda::CudaSession;
use vortex_cuda::executor::CudaArrayExt;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;
use vortex_error::VortexExpect;
use vortex_mask::Mask;
use vortex_session::VortexSession;

const BENCH_ARGS: &[(usize, usize, &str)] = &[
    (10_000_000, 2, "10M_50pct"),
    (10_000_000, 10, "10M_10pct"),
    (10_000_000, 100, "10M_1pct"),
];

/// Creates a mask with the given selectivity pattern.
fn make_mask(len: usize, selectivity: usize) -> Mask {
    let mut mask = BitBufferMut::with_capacity(len);
    for idx in 0..len {
        if idx % selectivity == 0 {
            mask.append_true();
        } else {
            mask.append_false();
        }
    }
    Mask::from_buffer(mask.freeze())
}

/// Creates a FilterArray of u32 with a selectivity pattern.
fn make_filter_array_u32(len: usize, selectivity: usize) -> FilterArray {
    let buf: Buffer<u32> = (0..len as u32).collect();
    let array = PrimitiveArray::new(buf, Validity::NonNullable);
    FilterArray::new(array.into_array(), make_mask(len, selectivity))
}

/// Creates a FilterArray of u64 with a selectivity pattern.
fn make_filter_array_u64(len: usize, selectivity: usize) -> FilterArray {
    let buf: Buffer<u64> = (0..len as u64).collect();
    let array = PrimitiveArray::new(buf, Validity::NonNullable);
    FilterArray::new(array.into_array(), make_mask(len, selectivity))
}

/// Creates a FilterArray of i32 with a selectivity pattern.
fn make_filter_array_i32(len: usize, selectivity: usize) -> FilterArray {
    let buf: Buffer<i32> = (0..len as i32).collect();
    let array = PrimitiveArray::new(buf, Validity::NonNullable);
    FilterArray::new(array.into_array(), make_mask(len, selectivity))
}

/// Creates a FilterArray of f64 with a selectivity pattern.
fn make_filter_array_f64(len: usize, selectivity: usize) -> FilterArray {
    let buf: Buffer<f64> = (0..len).map(|i| i as f64).collect();
    let array = PrimitiveArray::new(buf, Validity::NonNullable);
    FilterArray::new(array.into_array(), make_mask(len, selectivity))
}

/// Benchmark u32 filter operations
fn benchmark_filter_u32(c: &mut Criterion) {
    let mut group = c.benchmark_group("Filter_cuda_u32");
    group.sample_size(10);

    for (len, selectivity, label) in BENCH_ARGS {
        let filter_array = make_filter_array_u32(*len, *selectivity);

        group.throughput(Throughput::Bytes((len * size_of::<u32>()) as u64));
        group.bench_with_input(
            BenchmarkId::new("u32_Filter", label),
            &filter_array,
            |b, filter_array| {
                b.iter_custom(|iters| {
                    let mut total_time = Duration::ZERO;

                    for _ in 0..iters {
                        let mut cuda_ctx =
                            CudaSession::create_execution_ctx(&VortexSession::empty())
                                .vortex_expect("failed to create execution context");

                        let start = Instant::now();
                        let _result = block_on(
                            filter_array
                                .clone()
                                .into_array()
                                .execute_cuda(&mut cuda_ctx),
                        )
                        .vortex_expect("GPU filter failed");
                        cuda_ctx
                            .synchronize_stream()
                            .vortex_expect("failed to synchronize");
                        total_time += start.elapsed();
                    }

                    total_time
                });
            },
        );
    }

    group.finish();
}

/// Benchmark u64 filter operations
fn benchmark_filter_u64(c: &mut Criterion) {
    let mut group = c.benchmark_group("Filter_cuda_u64");
    group.sample_size(10);

    for (len, selectivity, label) in BENCH_ARGS {
        let filter_array = make_filter_array_u64(*len, *selectivity);

        group.throughput(Throughput::Bytes((len * size_of::<u64>()) as u64));
        group.bench_with_input(
            BenchmarkId::new("u64_Filter", label),
            &filter_array,
            |b, filter_array| {
                b.iter_custom(|iters| {
                    let mut total_time = Duration::ZERO;

                    for _ in 0..iters {
                        let mut cuda_ctx =
                            CudaSession::create_execution_ctx(&VortexSession::empty())
                                .vortex_expect("failed to create execution context");

                        let start = Instant::now();
                        let _result = block_on(
                            filter_array
                                .clone()
                                .into_array()
                                .execute_cuda(&mut cuda_ctx),
                        )
                        .vortex_expect("GPU filter failed");
                        cuda_ctx
                            .synchronize_stream()
                            .vortex_expect("failed to synchronize");
                        total_time += start.elapsed();
                    }

                    total_time
                });
            },
        );
    }

    group.finish();
}

/// Benchmark i32 filter operations
fn benchmark_filter_i32(c: &mut Criterion) {
    let mut group = c.benchmark_group("Filter_cuda_i32");
    group.sample_size(10);

    for (len, selectivity, label) in BENCH_ARGS {
        let filter_array = make_filter_array_i32(*len, *selectivity);

        group.throughput(Throughput::Bytes((len * size_of::<i32>()) as u64));
        group.bench_with_input(
            BenchmarkId::new("i32_Filter", label),
            &filter_array,
            |b, filter_array| {
                b.iter_custom(|iters| {
                    let mut total_time = Duration::ZERO;

                    for _ in 0..iters {
                        let mut cuda_ctx =
                            CudaSession::create_execution_ctx(&VortexSession::empty())
                                .vortex_expect("failed to create execution context");

                        let start = Instant::now();
                        let _result = block_on(
                            filter_array
                                .clone()
                                .into_array()
                                .execute_cuda(&mut cuda_ctx),
                        )
                        .vortex_expect("GPU filter failed");
                        cuda_ctx
                            .synchronize_stream()
                            .vortex_expect("failed to synchronize");
                        total_time += start.elapsed();
                    }

                    total_time
                });
            },
        );
    }

    group.finish();
}

/// Benchmark f64 filter operations
fn benchmark_filter_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("Filter_cuda_f64");
    group.sample_size(10);

    for (len, selectivity, label) in BENCH_ARGS {
        let filter_array = make_filter_array_f64(*len, *selectivity);

        group.throughput(Throughput::Bytes((len * size_of::<f64>()) as u64));
        group.bench_with_input(
            BenchmarkId::new("f64_Filter", label),
            &filter_array,
            |b, filter_array| {
                b.iter_custom(|iters| {
                    let mut total_time = Duration::ZERO;

                    for _ in 0..iters {
                        let mut cuda_ctx =
                            CudaSession::create_execution_ctx(&VortexSession::empty())
                                .vortex_expect("failed to create execution context");

                        let start = Instant::now();
                        let _result = block_on(
                            filter_array
                                .clone()
                                .into_array()
                                .execute_cuda(&mut cuda_ctx),
                        )
                        .vortex_expect("GPU filter failed");
                        cuda_ctx
                            .synchronize_stream()
                            .vortex_expect("failed to synchronize");
                        total_time += start.elapsed();
                    }

                    total_time
                });
            },
        );
    }

    group.finish();
}

pub fn benchmark_filter(c: &mut Criterion) {
    benchmark_filter_u32(c);
    benchmark_filter_u64(c);
    benchmark_filter_i32(c);
    benchmark_filter_f64(c);
}

criterion::criterion_group!(benches, benchmark_filter);

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
