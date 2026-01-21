// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmark Vortex file scanning with pinned/device allocators.
//!
//! Example:
//!   cargo run -p vortex-cuda --bin pinned_scan -- --rows 10000000,50000000 --iters 10 --scan device

#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#![allow(clippy::len_zero)]

use std::env;
use std::fmt;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

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
use vortex_io::DefaultAllocator;
use vortex_io::session::RuntimeSessionExt;
use vortex_io::session::RuntimeSession;
use vortex_layout::session::LayoutSession;
use vortex_metrics::VortexMetrics;
use vortex_session::VortexSession;

const DEFAULT_ROWS: &[usize] = &[10_000_000];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ScanType {
    DefaultZeroCopy,
    DefaultCopy,
    Pinned,
    Device,
}

struct Config {
    rows: Vec<usize>,
    iterations: usize,
    scans: Vec<ScanType>,
}

fn usage() -> &'static str {
    "Usage: pinned_scan [--rows N1,N2] [--iters N] [--scan NAME]\n\
\n\
Flags:\n\
  --rows    Comma-separated row counts (default: 10000000)\n\
  --iters   Iterations per scan (default: 10)\n\
  --scan    One of: default_zero_copy, default_copy, pinned, device, default, all\n"
}

fn parse_csv_usize(value: &str) -> Vec<usize> {
    value
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .collect()
}

fn parse_scan_list(value: &str) -> Vec<ScanType> {
    let mut scans = Vec::new();
    for item in value.split(',').map(|s| s.trim().to_lowercase()) {
        match item.as_str() {
            "default_zero_copy" | "zero_copy" => scans.push(ScanType::DefaultZeroCopy),
            "default_copy" | "copy" => scans.push(ScanType::DefaultCopy),
            "default" => {
                scans.push(ScanType::DefaultZeroCopy);
                scans.push(ScanType::DefaultCopy);
            }
            "pinned" => scans.push(ScanType::Pinned),
            "device" => scans.push(ScanType::Device),
            "all" => {
                scans = vec![
                    ScanType::DefaultZeroCopy,
                    ScanType::DefaultCopy,
                    ScanType::Pinned,
                    ScanType::Device,
                ];
                break;
            }
            _ => {}
        }
    }
    scans.sort();
    scans.dedup();
    scans
}

impl fmt::Display for ScanType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            ScanType::DefaultZeroCopy => "default_zero_copy",
            ScanType::DefaultCopy => "default_copy",
            ScanType::Pinned => "pinned",
            ScanType::Device => "device",
        };
        f.write_str(name)
    }
}

fn parse_args() -> Result<Config, String> {
    let mut rows: Option<Vec<usize>> = None;
    let mut iterations: usize = 10;
    let mut scans: Option<Vec<ScanType>> = None;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--rows" => {
                let value = args.next().ok_or_else(|| "Missing value for --rows".to_string())?;
                rows = Some(parse_csv_usize(&value));
            }
            "--iters" => {
                let value = args.next().ok_or_else(|| "Missing value for --iters".to_string())?;
                iterations = value
                    .parse::<usize>()
                    .map_err(|_| "Invalid value for --iters".to_string())?;
            }
            "--scan" => {
                let value = args.next().ok_or_else(|| "Missing value for --scan".to_string())?;
                scans = Some(parse_scan_list(&value));
            }
            "--help" | "-h" => return Err(usage().to_string()),
            _ if arg.starts_with("--rows=") => {
                let value = arg.trim_start_matches("--rows=");
                rows = Some(parse_csv_usize(value));
            }
            _ if arg.starts_with("--iters=") => {
                let value = arg.trim_start_matches("--iters=");
                iterations = value
                    .parse::<usize>()
                    .map_err(|_| "Invalid value for --iters".to_string())?;
            }
            _ if arg.starts_with("--scan=") => {
                let value = arg.trim_start_matches("--scan=");
                scans = Some(parse_scan_list(value));
            }
            _ => return Err(format!("Unknown argument: {arg}\n{}", usage())),
        }
    }

    let rows = rows.filter(|v| !v.is_empty()).unwrap_or_else(|| DEFAULT_ROWS.to_vec());
    let scans = scans.filter(|v| !v.is_empty()).unwrap_or_else(|| {
        vec![
            ScanType::DefaultZeroCopy,
            ScanType::DefaultCopy,
            ScanType::Pinned,
            ScanType::Device,
        ]
    });
    if iterations == 0 {
        return Err("Iterations must be > 0".to_string());
    }

    Ok(Config {
        rows,
        iterations,
        scans,
    })
}

fn format_row_label(num_rows: usize) -> String {
    if num_rows % 1_000_000 == 0 {
        format!("{}M_rows", num_rows / 1_000_000)
    } else {
        format!("{num_rows}_rows")
    }
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

fn create_vortex_buffer(rt: &Runtime, session: &VortexSession, num_rows: usize) -> ByteBuffer {
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
        assert!(!result.is_empty());
        elapsed
    })
}

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
        assert!(!result.is_empty());
        elapsed
    })
}

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
        assert!(!result.is_empty());
        elapsed
    })
}

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

        allocator.synchronize().expect("Failed to synchronize");

        let elapsed = start.elapsed();
        assert!(!result.is_empty());
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

struct ThroughputRow {
    gb_per_s: f64,
    ms_avg: f64,
}

impl fmt::Display for ThroughputRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2} GB/s ({:.2} ms avg)", self.gb_per_s, self.ms_avg)
    }
}

fn format_throughput(bytes: usize, total: Duration, iterations: usize) -> ThroughputRow {
    let gb_per_s = (bytes * iterations) as f64 / total.as_secs_f64() / 1e9;
    let ms_avg = total.as_secs_f64() * 1000.0 / iterations as f64;
    ThroughputRow { gb_per_s, ms_avg }
}

fn main() -> ExitCode {
    let config = match parse_args() {
        Ok(config) => config,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::FAILURE;
        }
    };

    println!("\n=== Vortex Scan Throughput (binary) ===\n");
    println!("Iterations: {}", config.iterations);
    let scan_list = config
        .scans
        .iter()
        .map(|scan| scan.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    println!("Scans: {}\n", scan_list);

    let rt = Runtime::new().unwrap();
    let session = create_session(&rt);

    let needs_cuda = config
        .scans
        .iter()
        .any(|s| matches!(s, ScanType::Pinned | ScanType::Device));
    if needs_cuda && !has_nvcc() {
        eprintln!("nvcc not found, cannot run pinned/device scans");
        return ExitCode::FAILURE;
    }
    let ctx = needs_cuda.then(|| CudaContext::new(0).expect("Failed to create CUDA context"));

    let stream = ctx
        .as_ref()
        .map(|ctx| Arc::new(ctx.new_stream().expect("Failed to create stream")));
    let pool = ctx.as_ref().map(|ctx| Arc::new(PinnedByteBufferPool::new(ctx.clone())));

    for num_rows in &config.rows {
        let label = format_row_label(*num_rows);
        println!("Rows: {} ({})", num_rows, label);
        let buffer = create_vortex_buffer(&rt, &session, *num_rows);
        println!(
            "  File size: {:.2} MB ({} bytes)",
            buffer.len() as f64 / 1e6,
            buffer.len()
        );

        if config.scans.contains(&ScanType::DefaultZeroCopy) {
            let total = run_scan_iters(config.iterations, || {
                scan_default(&rt, &session, &buffer)
            });
            println!(
                "  Default (zero-copy): {}",
                format_throughput(buffer.len(), total, config.iterations)
            );
        }
        if config.scans.contains(&ScanType::DefaultCopy) {
            let total = run_scan_iters(config.iterations, || {
                scan_default_copy(&rt, &session, &buffer)
            });
            println!(
                "  Default (copy):      {}",
                format_throughput(buffer.len(), total, config.iterations)
            );
        }
        if config.scans.contains(&ScanType::Pinned) {
            let pool = pool.as_ref().expect("Pinned pool required");
            let total = run_scan_iters(config.iterations, || {
                scan_pinned(&rt, &session, &buffer, pool)
            });
            println!(
                "  Pinned (host):       {}",
                format_throughput(buffer.len(), total, config.iterations)
            );
        }
        if config.scans.contains(&ScanType::Device) {
            let pool = pool.as_ref().expect("Pinned pool required");
            let stream = stream.as_ref().expect("CUDA stream required");
            let total = run_scan_iters(config.iterations, || {
                scan_device(&rt, &session, &buffer, pool, stream)
            });
            println!(
                "  Device (H2D):        {}",
                format_throughput(buffer.len(), total, config.iterations)
            );
        }
        println!();
    }
    ExitCode::SUCCESS
}
