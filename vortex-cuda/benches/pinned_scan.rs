// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmark Vortex file scanning with pinned buffer allocator.
//!
//! Run with: cargo bench -p vortex-cuda --bench pinned_scan
//!
//! This benchmark:
//! 1. Creates a synthetic Vortex file in memory
//! 2. Scans it with default allocator vs pinned allocator
//! 3. Measures total I/O + decode time

#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#![allow(clippy::len_zero)]

use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use cudarc::driver::CudaContext;
use tokio::runtime::Runtime;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::expr::session::ExprSession;
use vortex_array::session::ArraySession;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_cuda::PinnedBufferAllocator;
use vortex_cuda::PinnedByteBufferPool;
use vortex_cuda::PinnedDeviceAllocator;
use vortex_cuda::has_nvcc;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_file::register_default_encodings;
use vortex_io::session::RuntimeSessionExt;
use vortex_io::session::RuntimeSession;
use vortex_layout::session::LayoutSession;
use vortex_metrics::VortexMetrics;
use vortex_session::VortexSession;

// Test sizes: 1M, 10M, 100M rows of i64 (8 bytes each)
const ROW_COUNTS: &[(usize, &str)] = &[
    (1_000_000, "1M_rows"),
    (10_000_000, "10M_rows"),
    (100_000_000, "100M_rows"),
];

fn create_session(rt: &Runtime) -> VortexSession {
    let mut session = VortexSession::empty()
        .with::<VortexMetrics>()
        .with::<ArraySession>()
        .with::<LayoutSession>()
        .with::<ExprSession>()
        .with::<RuntimeSession>();
    register_default_encodings(&mut session);
    rt.block_on(async { session.with_tokio() })
}

/// Create a synthetic Vortex file in memory with the given number of rows.
fn create_vortex_buffer(rt: &Runtime, session: &VortexSession, num_rows: usize) -> ByteBuffer {
    // Create a simple i64 array with predictable data
    let data: Vec<i64> = (0..num_rows as i64).collect();
    let array = PrimitiveArray::new(Buffer::from(data), Validity::NonNullable).into_array();

    let mut buf = ByteBufferMut::empty();
    rt.block_on(async {
        session
            .write_options()
            .write(&mut buf, array.to_array_stream())
            .await
            .expect("Failed to write Vortex file");
    });

    ByteBuffer::from(buf)
}

/// Scan with default allocator (regular memory).
fn scan_default(rt: &Runtime, session: &VortexSession, buffer: &ByteBuffer) -> Duration {
    rt.block_on(async {
        let file = session
            .open_options()
            .open_buffer(buffer.clone())
            .expect("Failed to open file");

        let start = Instant::now();

        let result = file
            .scan()
            .expect("Failed to create scan")
            .into_array_stream()
            .expect("Failed to create stream")
            .read_all()
            .await
            .expect("Scan failed");

        let elapsed = start.elapsed();
        assert!(result.len() > 0);
        elapsed
    })
}

/// Scan with pinned allocator (data stays on host in pinned memory).
fn scan_pinned(
    rt: &Runtime,
    session: &VortexSession,
    buffer: &ByteBuffer,
    pool: &Arc<PinnedByteBufferPool>,
) -> Duration {
    let allocator = Arc::new(PinnedBufferAllocator::new(pool.clone()));

    rt.block_on(async {
        let file = session
            .open_options()
            .with_allocator(allocator)
            .open_buffer(buffer.clone())
            .expect("Failed to open file");

        let start = Instant::now();

        let result = file
            .scan()
            .expect("Failed to create scan")
            .into_array_stream()
            .expect("Failed to create stream")
            .read_all()
            .await
            .expect("Scan failed");

        let elapsed = start.elapsed();
        assert!(result.len() > 0);
        elapsed
    })
}

/// Scan with pinned device allocator (data transferred to GPU).
fn scan_device(
    rt: &Runtime,
    session: &VortexSession,
    buffer: &ByteBuffer,
    pool: &Arc<PinnedByteBufferPool>,
    stream: &Arc<cudarc::driver::CudaStream>,
) -> Duration {
    let allocator = Arc::new(PinnedDeviceAllocator::new(pool.clone(), stream.clone()));

    rt.block_on(async {
        let file = session
            .open_options()
            .with_allocator(allocator.clone())
            .open_buffer(buffer.clone())
            .expect("Failed to open file");

        let start = Instant::now();

        let result = file
            .scan()
            .expect("Failed to create scan")
            .into_array_stream()
            .expect("Failed to create stream")
            .read_all()
            .await
            .expect("Scan failed");

        // Synchronize to ensure all H2D transfers complete
        allocator.synchronize().expect("Failed to synchronize");

        let elapsed = start.elapsed();
        assert!(result.len() > 0);
        elapsed
    })
}

fn bench_scan_default(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let session = create_session(&rt);

    let mut group = c.benchmark_group("scan_default");
    group.sample_size(10);

    for (num_rows, label) in ROW_COUNTS {
        // Skip very large for CI
        if *num_rows > 10_000_000 {
            continue;
        }

        let buffer = create_vortex_buffer(&rt, &session, *num_rows);
        let bytes = buffer.len();

        group.throughput(Throughput::Bytes(bytes as u64));
        group.bench_with_input(BenchmarkId::new("default", label), &buffer, |b, buffer| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    total += scan_default(&rt, &session, buffer);
                }
                total
            });
        });
    }

    group.finish();
}

fn bench_scan_pinned(c: &mut Criterion) {
    if !has_nvcc() {
        eprintln!("nvcc not found, skipping pinned scan benchmark");
        return;
    }

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let pool = Arc::new(PinnedByteBufferPool::new(ctx));
    let rt = Runtime::new().unwrap();
    let session = create_session(&rt);

    let mut group = c.benchmark_group("scan_pinned");
    group.sample_size(10);

    for (num_rows, label) in ROW_COUNTS {
        if *num_rows > 10_000_000 {
            continue;
        }

        let buffer = create_vortex_buffer(&rt, &session, *num_rows);
        let bytes = buffer.len();

        group.throughput(Throughput::Bytes(bytes as u64));
        group.bench_with_input(BenchmarkId::new("pinned", label), &buffer, |b, buffer| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    total += scan_pinned(&rt, &session, buffer, &pool);
                }
                total
            });
        });
    }

    group.finish();
}

fn bench_scan_device(c: &mut Criterion) {
    if !has_nvcc() {
        eprintln!("nvcc not found, skipping device scan benchmark");
        return;
    }

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = Arc::new(ctx.new_stream().expect("Failed to create stream"));
    let pool = Arc::new(PinnedByteBufferPool::new(ctx));
    let rt = Runtime::new().unwrap();
    let session = create_session(&rt);

    let mut group = c.benchmark_group("scan_device");
    group.sample_size(10);

    for (num_rows, label) in ROW_COUNTS {
        if *num_rows > 10_000_000 {
            continue;
        }

        let buffer = create_vortex_buffer(&rt, &session, *num_rows);
        let bytes = buffer.len();

        group.throughput(Throughput::Bytes(bytes as u64));
        group.bench_with_input(BenchmarkId::new("device", label), &buffer, |b, buffer| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    total += scan_device(&rt, &session, buffer, &pool, &stream);
                }
                total
            });
        });
    }

    group.finish();
}

/// Quick comparison that prints results directly.
fn print_scan_comparison() {
    if !has_nvcc() {
        eprintln!("nvcc not found, skipping scan comparison");
        return;
    }

    println!("\n=== Vortex Scan: Default vs Pinned vs Device Allocator ===\n");

    let rt = Runtime::new().unwrap();
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = Arc::new(ctx.new_stream().expect("Failed to create stream"));
    let pool = Arc::new(PinnedByteBufferPool::new(ctx));
    let session = create_session(&rt);

    let num_rows = 10_000_000; // 10M rows
    println!("Creating Vortex file with {} rows...", num_rows);
    let buffer = create_vortex_buffer(&rt, &session, num_rows);
    println!(
        "File size: {:.2} MB ({} bytes)\n",
        buffer.len() as f64 / 1e6,
        buffer.len()
    );

    let iterations = 5;

    // Warmup
    println!("Warming up...");
    scan_default(&rt, &session, &buffer);
    scan_pinned(&rt, &session, &buffer, &pool);
    scan_device(&rt, &session, &buffer, &pool, &stream);

    // Default allocator
    println!(
        "Running {} iterations with default allocator...",
        iterations
    );
    let start = Instant::now();
    for _ in 0..iterations {
        scan_default(&rt, &session, &buffer);
    }
    let default_time = start.elapsed();
    let default_throughput = (buffer.len() * iterations) as f64 / default_time.as_secs_f64() / 1e9;

    // Pinned allocator (host)
    println!(
        "Running {} iterations with pinned allocator (host)...",
        iterations
    );
    let start = Instant::now();
    for _ in 0..iterations {
        scan_pinned(&rt, &session, &buffer, &pool);
    }
    let pinned_time = start.elapsed();
    let pinned_throughput = (buffer.len() * iterations) as f64 / pinned_time.as_secs_f64() / 1e9;

    // Device allocator (pinned + H2D)
    println!(
        "Running {} iterations with device allocator (pinned + H2D)...",
        iterations
    );
    let start = Instant::now();
    for _ in 0..iterations {
        scan_device(&rt, &session, &buffer, &pool, &stream);
    }
    let device_time = start.elapsed();
    let device_throughput = (buffer.len() * iterations) as f64 / device_time.as_secs_f64() / 1e9;

    println!();
    println!("Results:");
    println!(
        "  Default allocator:        {:.2} GB/s ({:.2} ms avg)",
        default_throughput,
        default_time.as_secs_f64() * 1000.0 / iterations as f64
    );
    println!(
        "  Pinned allocator (host):  {:.2} GB/s ({:.2} ms avg)",
        pinned_throughput,
        pinned_time.as_secs_f64() * 1000.0 / iterations as f64
    );
    println!(
        "  Device allocator (H2D):   {:.2} GB/s ({:.2} ms avg)",
        device_throughput,
        device_time.as_secs_f64() * 1000.0 / iterations as f64
    );
    println!();
    println!("Ratios vs default:");
    println!(
        "  Pinned (host): {:.2}x",
        pinned_throughput / default_throughput.max(0.001)
    );
    println!(
        "  Device (H2D):  {:.2}x",
        device_throughput / default_throughput.max(0.001)
    );
    println!();
}

fn all_benchmarks(c: &mut Criterion) {
    // Print quick summary first
    print_scan_comparison();

    // Run detailed benchmarks
    bench_scan_default(c);
    bench_scan_pinned(c);
    bench_scan_device(c);
}

criterion_group!(benches, all_benchmarks);
criterion_main!(benches);
