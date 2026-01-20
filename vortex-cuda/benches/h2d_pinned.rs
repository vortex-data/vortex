// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for H2D transfer throughput with pinned vs regular memory.
//!
//! Run with: cargo bench -p vortex-cuda --bench h2d_pinned
//!
//! This benchmark measures:
//! 1. Pure H2D transfer: pinned memory vs regular Vec<u8>
//! 2. Read-into-pinned: reading from RAM buffer into pinned memory
//! 3. Full pipeline: RAM -> pinned -> GPU

#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#![allow(clippy::redundant_clone)]

use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use cudarc::driver::CudaContext;
use vortex_cuda::PinnedByteBufferPool;
use vortex_cuda::has_nvcc;

// Buffer sizes to test: 1KB, 64KB, 1MB, 16MB, 64MB, 256MB
const SIZES: &[(usize, &str)] = &[
    (1 << 10, "1KB"),
    (1 << 16, "64KB"),
    (1 << 20, "1MB"),
    (16 << 20, "16MB"),
    (64 << 20, "64MB"),
    (256 << 20, "256MB"),
];

/// Benchmark H2D transfer from regular (pageable) memory.
/// CUDA internally stages through a pinned buffer, so this measures the slower path.
fn bench_h2d_regular(c: &mut Criterion) {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.new_stream().expect("Failed to create stream");

    let mut group = c.benchmark_group("h2d_regular");
    group.sample_size(10);

    for (size, label) in SIZES {
        // Skip very large sizes for regular memory test to save time
        if *size > 64 << 20 {
            continue;
        }

        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::new("regular", label), size, |b, &size| {
            // Allocate regular memory and touch it
            let data: Vec<u8> = vec![0x42u8; size];

            // Pre-allocate device buffer
            let mut device = unsafe { stream.alloc::<u8>(size) }.expect("Failed to alloc device");

            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let start = Instant::now();
                    stream.memcpy_htod(&data, &mut device).expect("H2D failed");
                    stream.synchronize().expect("Sync failed");
                    total += start.elapsed();
                }
                total
            });
        });
    }

    group.finish();
}

/// Benchmark H2D transfer from pinned memory.
/// This uses DMA and should be faster than regular memory.
fn bench_h2d_pinned(c: &mut Criterion) {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.new_stream().expect("Failed to create stream");
    let pool = Arc::new(PinnedByteBufferPool::new(ctx.clone()));

    let mut group = c.benchmark_group("h2d_pinned");
    group.sample_size(10);

    for (size, label) in SIZES {
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::new("pinned", label), size, |b, &size| {
            // Allocate pinned memory and touch it
            let mut pinned = pool.get(size).expect("Failed to get pinned buffer");
            pinned.as_mut_slice().expect("slice").fill(0x42);

            // Pre-allocate device buffer
            let mut device = unsafe { stream.alloc::<u8>(size) }.expect("Failed to alloc device");

            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let start = Instant::now();
                    stream
                        .memcpy_htod(&pinned, &mut device)
                        .expect("H2D failed");
                    stream.synchronize().expect("Sync failed");
                    total += start.elapsed();
                }
                total
            });

            // Return to pool
            pool.put(pinned).ok();
        });
    }

    group.finish();
}

/// Benchmark the PooledPinnedBuffer path (what the allocator uses).
fn bench_h2d_pooled_pinned(c: &mut Criterion) {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.new_stream().expect("Failed to create stream");
    let pool = Arc::new(PinnedByteBufferPool::new(ctx.clone()));

    let mut group = c.benchmark_group("h2d_pooled_pinned");
    group.sample_size(10);

    for (size, label) in SIZES {
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::new("pooled", label), size, |b, &size| {
            // Pre-allocate device buffer
            let mut device = unsafe { stream.alloc::<u8>(size) }.expect("Failed to alloc device");

            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    // Get from pool, fill, transfer, return to pool
                    let mut pooled = pool.get_pooled(size).expect("Failed to get pooled buffer");
                    pooled.as_mut_slice().fill(0x42);

                    let start = Instant::now();
                    stream
                        .memcpy_htod(&pooled, &mut device)
                        .expect("H2D failed");
                    stream.synchronize().expect("Sync failed");
                    total += start.elapsed();

                    // pooled is returned to pool on drop
                }
                total
            });
        });
    }

    group.finish();
}

/// Benchmark copying from RAM into pinned buffer, then H2D.
/// This simulates: read from file/network into RAM, copy to pinned, transfer to GPU.
fn bench_ram_to_pinned_to_gpu(c: &mut Criterion) {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.new_stream().expect("Failed to create stream");
    let pool = Arc::new(PinnedByteBufferPool::new(ctx.clone()));

    let mut group = c.benchmark_group("ram_pinned_gpu");
    group.sample_size(10);

    for (size, label) in SIZES {
        // Skip very large for this combined test
        if *size > 64 << 20 {
            continue;
        }

        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("ram_to_pinned_to_gpu", label),
            size,
            |b, &size| {
                // Source data in regular RAM
                let ram_data: Vec<u8> = vec![0x42u8; size];

                // Pre-allocate device buffer
                let mut device =
                    unsafe { stream.alloc::<u8>(size) }.expect("Failed to alloc device");

                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let start = Instant::now();

                        // Get pinned buffer
                        let mut pinned =
                            pool.get_pooled(size).expect("Failed to get pooled buffer");

                        // Copy RAM -> pinned
                        pinned.as_mut_slice().copy_from_slice(&ram_data);

                        // Transfer pinned -> GPU
                        stream
                            .memcpy_htod(&pinned, &mut device)
                            .expect("H2D failed");
                        stream.synchronize().expect("Sync failed");

                        total += start.elapsed();
                    }
                    total
                });
            },
        );
    }

    group.finish();
}

/// Benchmark direct RAM to GPU (baseline without pinned intermediate).
fn bench_ram_to_gpu_direct(c: &mut Criterion) {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.new_stream().expect("Failed to create stream");

    let mut group = c.benchmark_group("ram_gpu_direct");
    group.sample_size(10);

    for (size, label) in SIZES {
        if *size > 64 << 20 {
            continue;
        }

        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("ram_to_gpu_direct", label),
            size,
            |b, &size| {
                let ram_data: Vec<u8> = vec![0x42u8; size];
                let mut device =
                    unsafe { stream.alloc::<u8>(size) }.expect("Failed to alloc device");

                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let start = Instant::now();
                        stream
                            .memcpy_htod(&ram_data, &mut device)
                            .expect("H2D failed");
                        stream.synchronize().expect("Sync failed");
                        total += start.elapsed();
                    }
                    total
                });
            },
        );
    }

    group.finish();
}

/// Quick sanity check that prints bandwidth numbers.
fn print_bandwidth_summary() {
    println!("\n=== H2D Bandwidth Quick Test ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.new_stream().expect("Failed to create stream");
    let pool = Arc::new(PinnedByteBufferPool::new(ctx.clone()));

    let size = 256 << 20; // 256MB
    let iterations = 10;

    // Pinned test
    let mut pinned = pool.get(size).expect("Failed to get pinned buffer");
    pinned.as_mut_slice().expect("slice").fill(0x42);
    let mut device = unsafe { stream.alloc::<u8>(size) }.expect("Failed to alloc device");

    // Warmup
    for _ in 0..3 {
        stream
            .memcpy_htod(&pinned, &mut device)
            .expect("H2D failed");
        stream.synchronize().expect("Sync failed");
    }

    let start = Instant::now();
    for _ in 0..iterations {
        stream
            .memcpy_htod(&pinned, &mut device)
            .expect("H2D failed");
        stream.synchronize().expect("Sync failed");
    }
    let pinned_time = start.elapsed();
    let pinned_bw = (size * iterations) as f64 / pinned_time.as_secs_f64() / 1e9;

    pool.put(pinned).ok();

    // Regular test
    let regular: Vec<u8> = vec![0x42u8; size];

    // Warmup
    for _ in 0..3 {
        stream
            .memcpy_htod(&regular, &mut device)
            .expect("H2D failed");
        stream.synchronize().expect("Sync failed");
    }

    let start = Instant::now();
    for _ in 0..iterations {
        stream
            .memcpy_htod(&regular, &mut device)
            .expect("H2D failed");
        stream.synchronize().expect("Sync failed");
    }
    let regular_time = start.elapsed();
    let regular_bw = (size * iterations) as f64 / regular_time.as_secs_f64() / 1e9;

    println!("Buffer size: {} MB", size >> 20);
    println!("Iterations: {}", iterations);
    println!();
    println!(
        "Pinned memory:  {:.2} GB/s ({:.2} ms per transfer)",
        pinned_bw,
        pinned_time.as_secs_f64() * 1000.0 / iterations as f64
    );
    println!(
        "Regular memory: {:.2} GB/s ({:.2} ms per transfer)",
        regular_bw,
        regular_time.as_secs_f64() * 1000.0 / iterations as f64
    );
    println!("Speedup: {:.2}x", pinned_bw / regular_bw);
    println!();
}

fn all_benchmarks(c: &mut Criterion) {
    if !has_nvcc() {
        eprintln!("nvcc not found, skipping CUDA benchmarks");
        return;
    }

    // Print quick summary first
    print_bandwidth_summary();

    // Run detailed benchmarks
    bench_h2d_pinned(c);
    bench_h2d_regular(c);
    bench_h2d_pooled_pinned(c);
    bench_ram_to_pinned_to_gpu(c);
    bench_ram_to_gpu_direct(c);
}

criterion_group!(benches, all_benchmarks);
criterion_main!(benches);
