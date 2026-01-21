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
use std::{env, fmt};

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use cudarc::driver::CudaContext;
use rand::rng;
use rand::seq::SliceRandom;
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
use vortex_io::DefaultAllocator;
use vortex_layout::session::LayoutSession;
use vortex_metrics::VortexMetrics;
use vortex_session::VortexSession;

// Test sizes: 1M, 10M, 100M rows of i64 (8 bytes each)
const ROW_COUNTS: &[(usize, &str)] = &[
    (1_000_000, "1M_rows"),
    (10_000_000, "10M_rows"),
    (100_000_000, "100M_rows"),
];

fn format_row_label(num_rows: usize) -> String {
    if num_rows % 1_000_000 == 0 {
        format!("{}M_rows", num_rows / 1_000_000)
    } else {
        format!("{num_rows}_rows")
    }
}

fn row_counts() -> Vec<(usize, String)> {
    if let Some(list) = split_env_list("VORTEX_PINNED_SCAN_ROWS") {
        let mut rows = Vec::new();
        for item in list {
            if let Ok(value) = item.parse::<usize>()
                && value > 0
            {
                rows.push((value, format_row_label(value)));
            }
        }
        if !rows.is_empty() {
            return rows;
        }
    }

    ROW_COUNTS
        .iter()
        .map(|(rows, label)| (*rows, (*label).to_string()))
        .collect()
}

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
    let mut data: Vec<i64> = (0..num_rows as i64).collect();
    data.shuffle(&mut rng());
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

/// Scan with default allocator but force a copy into an owned buffer.
fn scan_default_copy(rt: &Runtime, session: &VortexSession, buffer: &ByteBuffer) -> Duration {
    let allocator = Arc::new(DefaultAllocator);

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

fn run_scan_iters<F>(iterations: usize, mut f: F) -> Duration
where
    F: FnMut() -> Duration,
{
    let mut total = Duration::ZERO;
    for _ in 0..iterations {
        total += f();
    }
    total
}

fn split_env_list(var: &str) -> Option<Vec<String>> {
    let raw = env::var(var).ok()?;
    let items: Vec<String> = raw
        .split(',')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect();
    if items.is_empty() {
        None
    } else {
        Some(items)
    }
}

fn is_selected(name: &str) -> bool {
    let name = name.to_lowercase();
    if let Some(only) = split_env_list("VORTEX_PINNED_SCAN_ONLY") {
        return only.iter().any(|v| v == &name);
    }
    if let Some(skip) = split_env_list("VORTEX_PINNED_SCAN_SKIP") {
        return !skip.iter().any(|v| v == &name);
    }
    true
}

fn format_throughput(bytes: usize, total: Duration, iterations: usize) -> ThroughputRow {
    let gb_per_s = (bytes * iterations) as f64 / total.as_secs_f64() / 1e9;
    let ms_avg = total.as_secs_f64() * 1000.0 / iterations as f64;
    ThroughputRow { gb_per_s, ms_avg }
}

struct ThroughputRow {
    gb_per_s: f64,
    ms_avg: f64,
}

impl fmt::Display for ThroughputRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2} GB/s ({:.2} ms avg)", self.gb_per_s, self.ms_avg)
    }
}

fn bench_scan_default(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let session = create_session(&rt);

    let run_zero_copy = is_selected("default_zero_copy");
    let run_copy = is_selected("default_copy");
    if !run_zero_copy && !run_copy {
        return;
    }

    let mut group = c.benchmark_group("scan_default");
    group.sample_size(10);

    for (num_rows, label) in row_counts() {
        // Skip very large for CI unless overridden.
        if env::var("VORTEX_PINNED_SCAN_ROWS").is_err() && num_rows > 10_000_000 {
            continue;
        }

        let buffer = create_vortex_buffer(&rt, &session, num_rows);
        let bytes = buffer.len();

        group.throughput(Throughput::Bytes(bytes as u64));
        if run_zero_copy {
            group.bench_with_input(
                BenchmarkId::new("default_zero_copy", label.as_str()),
                &buffer,
                |b, buffer| {
                    b.iter_custom(|iters| {
                        let mut total = Duration::ZERO;
                        for _ in 0..iters {
                            total += scan_default(&rt, &session, buffer);
                        }
                        total
                    });
                },
            );
        }
        if run_copy {
            group.bench_with_input(
                BenchmarkId::new("default_copy", label.as_str()),
                &buffer,
                |b, buffer| {
                    b.iter_custom(|iters| {
                        let mut total = Duration::ZERO;
                        for _ in 0..iters {
                            total += scan_default_copy(&rt, &session, buffer);
                        }
                        total
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_scan_pinned(c: &mut Criterion) {
    if !is_selected("pinned") {
        return;
    }

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

    for (num_rows, label) in row_counts() {
        if env::var("VORTEX_PINNED_SCAN_ROWS").is_err() && num_rows > 10_000_000 {
            continue;
        }

        let buffer = create_vortex_buffer(&rt, &session, num_rows);
        let bytes = buffer.len();

        group.throughput(Throughput::Bytes(bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("pinned", label.as_str()),
            &buffer,
            |b, buffer| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        total += scan_pinned(&rt, &session, buffer, &pool);
                    }
                    total
                });
            },
        );
    }

    group.finish();
}

fn bench_scan_device(c: &mut Criterion) {
    if !is_selected("device") {
        return;
    }

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

    for (num_rows, label) in row_counts() {
        if env::var("VORTEX_PINNED_SCAN_ROWS").is_err() && num_rows > 10_000_000 {
            continue;
        }

        let buffer = create_vortex_buffer(&rt, &session, num_rows);
        let bytes = buffer.len();

        group.throughput(Throughput::Bytes(bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("device", label.as_str()),
            &buffer,
            |b, buffer| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        total += scan_device(&rt, &session, buffer, &pool, &stream);
                    }
                    total
                });
            },
        );
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

    let rows = row_counts();
    let (num_rows, label) = rows
        .first()
        .map(|(rows, label)| (*rows, label.as_str()))
        .unwrap_or((10_000_000, "10M_rows"));
    println!("Creating Vortex file with {} rows ({})...", num_rows, label);
    let buffer = create_vortex_buffer(&rt, &session, num_rows);
    println!(
        "File size: {:.2} MB ({} bytes)\n",
        buffer.len() as f64 / 1e6,
        buffer.len()
    );

    let iterations = 5;

    // Warmup
    println!("Warming up...");
    if is_selected("default_zero_copy") {
        scan_default(&rt, &session, &buffer);
    }
    if is_selected("default_copy") {
        scan_default_copy(&rt, &session, &buffer);
    }
    if is_selected("pinned") {
        scan_pinned(&rt, &session, &buffer, &pool);
    }
    if is_selected("device") {
        scan_device(&rt, &session, &buffer, &pool, &stream);
    }

    // Default allocator (zero-copy)
    let mut default_zero_copy_time = None;
    let mut default_zero_copy_throughput = None;
    if is_selected("default_zero_copy") {
        println!(
            "Running {} iterations with default allocator (zero-copy)...",
            iterations
        );
        let start = Instant::now();
        for _ in 0..iterations {
            scan_default(&rt, &session, &buffer);
        }
        let elapsed = start.elapsed();
        default_zero_copy_time = Some(elapsed);
        default_zero_copy_throughput =
            Some((buffer.len() * iterations) as f64 / elapsed.as_secs_f64() / 1e9);
    }

    // Default allocator (forced copy)
    let mut default_copy_time = None;
    let mut default_copy_throughput = None;
    if is_selected("default_copy") {
        println!(
            "Running {} iterations with default allocator (copy)...",
            iterations
        );
        let start = Instant::now();
        for _ in 0..iterations {
            scan_default_copy(&rt, &session, &buffer);
        }
        let elapsed = start.elapsed();
        default_copy_time = Some(elapsed);
        default_copy_throughput =
            Some((buffer.len() * iterations) as f64 / elapsed.as_secs_f64() / 1e9);
    }

    // Pinned allocator (host)
    let mut pinned_time = None;
    let mut pinned_throughput = None;
    if is_selected("pinned") {
        println!(
            "Running {} iterations with pinned allocator (host)...",
            iterations
        );
        let start = Instant::now();
        for _ in 0..iterations {
            scan_pinned(&rt, &session, &buffer, &pool);
        }
        let elapsed = start.elapsed();
        pinned_time = Some(elapsed);
        pinned_throughput =
            Some((buffer.len() * iterations) as f64 / elapsed.as_secs_f64() / 1e9);
    }

    // Device allocator (pinned + H2D)
    let mut device_time = None;
    let mut device_throughput = None;
    if is_selected("device") {
        println!(
            "Running {} iterations with device allocator (pinned + H2D)...",
            iterations
        );
        let start = Instant::now();
        for _ in 0..iterations {
            scan_device(&rt, &session, &buffer, &pool, &stream);
        }
        let elapsed = start.elapsed();
        device_time = Some(elapsed);
        device_throughput =
            Some((buffer.len() * iterations) as f64 / elapsed.as_secs_f64() / 1e9);
    }

    println!();
    println!("Results:");
    if let (Some(tp), Some(time)) = (default_zero_copy_throughput, default_zero_copy_time) {
        println!(
            "  Default allocator (zero-copy): {:.2} GB/s ({:.2} ms avg)",
            tp,
            time.as_secs_f64() * 1000.0 / iterations as f64
        );
    }
    if let (Some(tp), Some(time)) = (default_copy_throughput, default_copy_time) {
        println!(
            "  Default allocator (copy):      {:.2} GB/s ({:.2} ms avg)",
            tp,
            time.as_secs_f64() * 1000.0 / iterations as f64
        );
    }
    if let (Some(tp), Some(time)) = (pinned_throughput, pinned_time) {
        println!(
            "  Pinned allocator (host):  {:.2} GB/s ({:.2} ms avg)",
            tp,
            time.as_secs_f64() * 1000.0 / iterations as f64
        );
    }
    if let (Some(tp), Some(time)) = (device_throughput, device_time) {
        println!(
            "  Device allocator (H2D):   {:.2} GB/s ({:.2} ms avg)",
            tp,
            time.as_secs_f64() * 1000.0 / iterations as f64
        );
    }
    println!();
    if let (Some(pinned), Some(device), Some(default_copy)) = (
        pinned_throughput,
        device_throughput,
        default_copy_throughput,
    ) {
        println!("Ratios vs default (copy):");
        println!("  Pinned (host): {:.2}x", pinned / default_copy.max(0.001));
        println!("  Device (H2D):  {:.2}x", device / default_copy.max(0.001));
    }
    if let (Some(pinned), Some(device), Some(default_zero)) = (
        pinned_throughput,
        device_throughput,
        default_zero_copy_throughput,
    ) {
        println!("Ratios vs default (zero-copy):");
        println!("  Pinned (host): {:.2}x", pinned / default_zero.max(0.001));
        println!("  Device (H2D):  {:.2}x", device / default_zero.max(0.001));
    }
    println!();
}

/// Quick comparison across multiple sizes, skipping Criterion output.
fn print_scan_comparison_quick() {
    if !has_nvcc() {
        eprintln!("nvcc not found, skipping scan comparison");
        return;
    }

    let iterations = env::var("VORTEX_PINNED_SCAN_ITERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(10);

    println!("\n=== Vortex Scan Quick Throughput ===\n");
    println!(
        "Iterations: {} (set VORTEX_PINNED_SCAN_ITERS to override)\n",
        iterations
    );

    let rt = Runtime::new().unwrap();
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = Arc::new(ctx.new_stream().expect("Failed to create stream"));
    let pool = Arc::new(PinnedByteBufferPool::new(ctx));
    let session = create_session(&rt);

    for (num_rows, label) in row_counts() {
        if env::var("VORTEX_PINNED_SCAN_ROWS").is_err() && num_rows > 10_000_000 {
            continue;
        }

        println!("Rows: {} ({})", num_rows, label);
        let buffer = create_vortex_buffer(&rt, &session, num_rows);
        println!(
            "  File size: {:.2} MB ({} bytes)",
            buffer.len() as f64 / 1e6,
            buffer.len()
        );

        // Warmup
        if is_selected("default_zero_copy") {
            scan_default(&rt, &session, &buffer);
        }
        if is_selected("default_copy") {
            scan_default_copy(&rt, &session, &buffer);
        }
        if is_selected("pinned") {
            scan_pinned(&rt, &session, &buffer, &pool);
        }
        if is_selected("device") {
            scan_device(&rt, &session, &buffer, &pool, &stream);
        }

        if is_selected("default_zero_copy") {
            let total = run_scan_iters(iterations, || scan_default(&rt, &session, &buffer));
            let row = format_throughput(buffer.len(), total, iterations);
            println!("  Default (zero-copy): {}", row);
        }
        if is_selected("default_copy") {
            let total = run_scan_iters(iterations, || scan_default_copy(&rt, &session, &buffer));
            let row = format_throughput(buffer.len(), total, iterations);
            println!("  Default (copy):      {}", row);
        }
        if is_selected("pinned") {
            let total = run_scan_iters(iterations, || {
                scan_pinned(&rt, &session, &buffer, &pool)
            });
            let row = format_throughput(buffer.len(), total, iterations);
            println!("  Pinned (host):       {}", row);
        }
        if is_selected("device") {
            let total =
                run_scan_iters(iterations, || scan_device(&rt, &session, &buffer, &pool, &stream));
            let row = format_throughput(buffer.len(), total, iterations);
            println!("  Device (H2D):        {}", row);
        }
        println!();
    }
}

fn all_benchmarks(c: &mut Criterion) {
    if env::var("VORTEX_PINNED_SCAN_QUICK").is_ok() {
        print_scan_comparison_quick();
        return;
    }

    // Print quick summary first
    print_scan_comparison();

    // Run detailed benchmarks
    bench_scan_default(c);
    bench_scan_pinned(c);
    bench_scan_device(c);
}

criterion_group!(benches, all_benchmarks);
criterion_main!(benches);
