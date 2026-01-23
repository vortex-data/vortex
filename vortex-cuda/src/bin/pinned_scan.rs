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
use futures::stream;
use futures::StreamExt;
use rand::RngCore;
use rand::SeedableRng;
use rand::rngs::StdRng;
use tokio::runtime::Runtime;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::expr::session::ExprSession;
use vortex_array::session::ArraySession;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::alignment_copy_stats;
use vortex_buffer::reset_alignment_copy_stats;
use vortex_buffer::Alignment;
use vortex_cuda::PinnedBufferAllocator;
use vortex_cuda::PinnedByteBufferPool;
use vortex_cuda::PinnedDeviceAllocator;
use vortex_cuda::has_nvcc;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_file::WriteStrategyBuilder;
use vortex_file::register_default_encodings;
use vortex_file::segments::io_request_stats;
use vortex_file::segments::reset_io_request_stats;
use vortex_io::DefaultAllocator;
use vortex_io::HostByteBufferPool;
use vortex_io::PooledHostAllocator;
use vortex_io::copy_stats;
use vortex_io::default_alloc_stats;
use vortex_io::reset_copy_stats;
use vortex_io::reset_default_alloc_stats;
use vortex_io::session::RuntimeSessionExt;
use vortex_io::session::RuntimeSession;
use vortex_layout::session::LayoutSession;
use vortex_metrics::VortexMetrics;
use vortex_scan::reset_scan_task_stats;
use vortex_scan::scan_task_stats;
use vortex_session::VortexSession;

const DEFAULT_ROWS: &[usize] = &[10_000_000];
const DEFAULT_ROW_BLOCK_SIZE: usize = 8192;
const DEFAULT_BUFFERED_BYTES: u64 = 2 * 1024 * 1024;
const DEFAULT_COALESCE_MIN_BYTES: u64 = 1024 * 1024;
const DEFAULT_CHUNK_ROWS: usize = 1_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ScanType {
    DefaultZeroCopy,
    DefaultCopy,
    DefaultCopyPooled,
    Pinned,
    Device,
}

struct Config {
    rows: Vec<usize>,
    iterations: usize,
    scans: Vec<ScanType>,
    row_block_size: usize,
    buffered_bytes: u64,
    coalesce_min_bytes: u64,
    chunk_rows: usize,
    sweep_buffered_bytes: Vec<u64>,
    sweep_coalesce_min_bytes: Vec<u64>,
    sweep_row_block_rows: Vec<usize>,
    sweep_chunk_rows: Vec<usize>,
}

fn usage() -> &'static str {
    "Usage: pinned_scan [--rows N1,N2] [--iters N] [--scan NAME]\n\
\n\
Flags:\n\
  --rows    Comma-separated row counts (default: 10000000)\n\
  --iters   Iterations per scan (default: 10)\n\
  --scan    One of: default_zero_copy, default_copy, default_copy_pooled, device, default, all\n\
  --row-block-rows     Rows per row block (default: 8192)\n\
  --buffered-bytes     Buffered segment target in bytes (default: 2097152)\n\
  --coalesce-min-bytes Minimum coalesced segment size in bytes (default: 1048576)\n\
  --chunk-rows         Rows per generated chunk (default: 1000000)\n\
  --sweep-buffered-bytes    Comma-separated buffered bytes values to sweep\n\
  --sweep-coalesce-min-bytes Comma-separated coalesce min bytes values to sweep\n\
  --sweep-row-block-rows    Comma-separated row block sizes to sweep\n\
  --sweep-chunk-rows        Comma-separated chunk row sizes to sweep\n"
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

fn parse_csv_u64(value: &str) -> Vec<u64> {
    value
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .collect()
}

fn parse_scan_list(value: &str) -> Vec<ScanType> {
    let mut scans = Vec::new();
    for item in value.split(',').map(|s| s.trim().to_lowercase()) {
        match item.as_str() {
            "default_zero_copy" | "zero_copy" => scans.push(ScanType::DefaultZeroCopy),
            "default_copy" | "copy" => scans.push(ScanType::DefaultCopy),
            "default_copy_pooled" | "copy_pooled" => scans.push(ScanType::DefaultCopyPooled),
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
                    ScanType::DefaultCopyPooled,
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
            ScanType::DefaultCopyPooled => "default_copy_pooled",
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
    let mut row_block_size = DEFAULT_ROW_BLOCK_SIZE;
    let mut buffered_bytes = DEFAULT_BUFFERED_BYTES;
    let mut coalesce_min_bytes = DEFAULT_COALESCE_MIN_BYTES;
    let mut chunk_rows = DEFAULT_CHUNK_ROWS;
    let mut sweep_buffered_bytes = Vec::new();
    let mut sweep_coalesce_min_bytes = Vec::new();
    let mut sweep_row_block_rows = Vec::new();
    let mut sweep_chunk_rows = Vec::new();

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
            "--row-block-rows" => {
                let value = args
                    .next()
                    .ok_or_else(|| "Missing value for --row-block-rows".to_string())?;
                row_block_size = value
                    .parse::<usize>()
                    .map_err(|_| "Invalid value for --row-block-rows".to_string())?;
            }
            "--buffered-bytes" => {
                let value = args
                    .next()
                    .ok_or_else(|| "Missing value for --buffered-bytes".to_string())?;
                buffered_bytes = value
                    .parse::<u64>()
                    .map_err(|_| "Invalid value for --buffered-bytes".to_string())?;
            }
            "--coalesce-min-bytes" => {
                let value = args
                    .next()
                    .ok_or_else(|| "Missing value for --coalesce-min-bytes".to_string())?;
                coalesce_min_bytes = value
                    .parse::<u64>()
                    .map_err(|_| "Invalid value for --coalesce-min-bytes".to_string())?;
            }
            "--chunk-rows" => {
                let value = args
                    .next()
                    .ok_or_else(|| "Missing value for --chunk-rows".to_string())?;
                chunk_rows = value
                    .parse::<usize>()
                    .map_err(|_| "Invalid value for --chunk-rows".to_string())?;
            }
            "--sweep-buffered-bytes" => {
                let value = args
                    .next()
                    .ok_or_else(|| "Missing value for --sweep-buffered-bytes".to_string())?;
                sweep_buffered_bytes = parse_csv_u64(&value);
            }
            "--sweep-coalesce-min-bytes" => {
                let value = args
                    .next()
                    .ok_or_else(|| "Missing value for --sweep-coalesce-min-bytes".to_string())?;
                sweep_coalesce_min_bytes = parse_csv_u64(&value);
            }
            "--sweep-row-block-rows" => {
                let value = args
                    .next()
                    .ok_or_else(|| "Missing value for --sweep-row-block-rows".to_string())?;
                sweep_row_block_rows = parse_csv_usize(&value);
            }
            "--sweep-chunk-rows" => {
                let value = args
                    .next()
                    .ok_or_else(|| "Missing value for --sweep-chunk-rows".to_string())?;
                sweep_chunk_rows = parse_csv_usize(&value);
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
            _ if arg.starts_with("--row-block-rows=") => {
                let value = arg.trim_start_matches("--row-block-rows=");
                row_block_size = value
                    .parse::<usize>()
                    .map_err(|_| "Invalid value for --row-block-rows".to_string())?;
            }
            _ if arg.starts_with("--buffered-bytes=") => {
                let value = arg.trim_start_matches("--buffered-bytes=");
                buffered_bytes = value
                    .parse::<u64>()
                    .map_err(|_| "Invalid value for --buffered-bytes".to_string())?;
            }
            _ if arg.starts_with("--coalesce-min-bytes=") => {
                let value = arg.trim_start_matches("--coalesce-min-bytes=");
                coalesce_min_bytes = value
                    .parse::<u64>()
                    .map_err(|_| "Invalid value for --coalesce-min-bytes".to_string())?;
            }
            _ if arg.starts_with("--chunk-rows=") => {
                let value = arg.trim_start_matches("--chunk-rows=");
                chunk_rows = value
                    .parse::<usize>()
                    .map_err(|_| "Invalid value for --chunk-rows".to_string())?;
            }
            _ if arg.starts_with("--sweep-buffered-bytes=") => {
                let value = arg.trim_start_matches("--sweep-buffered-bytes=");
                sweep_buffered_bytes = parse_csv_u64(value);
            }
            _ if arg.starts_with("--sweep-coalesce-min-bytes=") => {
                let value = arg.trim_start_matches("--sweep-coalesce-min-bytes=");
                sweep_coalesce_min_bytes = parse_csv_u64(value);
            }
            _ if arg.starts_with("--sweep-row-block-rows=") => {
                let value = arg.trim_start_matches("--sweep-row-block-rows=");
                sweep_row_block_rows = parse_csv_usize(value);
            }
            _ if arg.starts_with("--sweep-chunk-rows=") => {
                let value = arg.trim_start_matches("--sweep-chunk-rows=");
                sweep_chunk_rows = parse_csv_usize(value);
            }
            _ => return Err(format!("Unknown argument: {arg}\n{}", usage())),
        }
    }

    let rows = rows.filter(|v| !v.is_empty()).unwrap_or_else(|| DEFAULT_ROWS.to_vec());
    let scans = scans.filter(|v| !v.is_empty()).unwrap_or_else(|| {
        vec![
            ScanType::DefaultZeroCopy,
            ScanType::DefaultCopy,
            ScanType::DefaultCopyPooled,
            ScanType::Device,
        ]
    });
    if iterations == 0 {
        return Err("Iterations must be > 0".to_string());
    }
    if row_block_size == 0 || chunk_rows == 0 {
        return Err("Row sizes must be > 0".to_string());
    }

    Ok(Config {
        rows,
        iterations,
        scans,
        row_block_size,
        buffered_bytes,
        coalesce_min_bytes,
        chunk_rows,
        sweep_buffered_bytes,
        sweep_coalesce_min_bytes,
        sweep_row_block_rows,
        sweep_chunk_rows,
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

fn count_splits(rt: &Runtime, session: &VortexSession, buffer: &ByteBuffer) -> usize {
    rt.block_on(async {
        let file = session
            .open_options()
            .open_buffer(buffer.clone())
            .expect("Failed to open file");
        file.splits().map(|splits| splits.len()).unwrap_or(0)
    })
}

#[derive(Clone, Copy)]
struct WriteParams {
    row_block_size: usize,
    buffered_bytes: u64,
    coalesce_min_bytes: u64,
    chunk_rows: usize,
}

fn create_vortex_buffer(
    rt: &Runtime,
    session: &VortexSession,
    num_rows: usize,
    params: WriteParams,
) -> ByteBuffer {
    let dtype = PrimitiveArray::new(Buffer::from(Vec::<i64>::new()), Validity::NonNullable)
        .into_array()
        .dtype()
        .clone();
    let chunk_rows = params.chunk_rows.max(1);
    let stream = stream::unfold(
        (num_rows, StdRng::seed_from_u64(0xC0FFEE)),
        move |(remaining, mut rng)| async move {
            if remaining == 0 {
                None
            } else {
                let len = remaining.min(chunk_rows);
                let mut data = Vec::with_capacity(len);
                for _ in 0..len {
                    data.push(rng.next_u64() as i64);
                }
                let array =
                    PrimitiveArray::new(Buffer::from(data), Validity::NonNullable).into_array();
                Some((Ok(array), (remaining - len, rng)))
            }
        },
    );
    let array_stream = ArrayStreamAdapter::new(dtype, stream);
    let strategy = WriteStrategyBuilder::new()
        .with_row_block_size(params.row_block_size)
        .with_buffered_bytes(params.buffered_bytes)
        .with_coalesce_min_bytes(params.coalesce_min_bytes)
        .build();

    let mut buf = ByteBufferMut::empty();
    rt.block_on(async {
        session
            .write_options()
            .with_strategy(strategy)
            .write(&mut buf, array_stream)
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
        let mut stream = file
            .scan()
            .expect("Failed to create scan")
            .into_array_stream()
            .expect("Failed to create stream");

        let mut saw_any = false;
        while let Some(chunk) = stream.next().await {
            let _chunk = chunk.expect("Scan failed");
            saw_any = true;
        }
        assert!(saw_any);
        start.elapsed()
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
        let mut stream = file
            .scan()
            .expect("Failed to create scan")
            .into_array_stream()
            .expect("Failed to create stream");

        let mut saw_any = false;
        while let Some(chunk) = stream.next().await {
            let _chunk = chunk.expect("Scan failed");
            saw_any = true;
        }
        assert!(saw_any);
        start.elapsed()
    })
}

fn scan_default_copy_pooled(
    rt: &Runtime,
    session: &VortexSession,
    buffer: &ByteBuffer,
    pool: &Arc<HostByteBufferPool>,
) -> Duration {
    let allocator = Arc::new(PooledHostAllocator::new(pool.clone()));

    rt.block_on(async {
        let file = session
            .open_options()
            .with_allocator(allocator)
            .open_buffer(buffer.clone())
            .expect("Failed to open file");

        let start = Instant::now();
        let mut stream = file
            .scan()
            .expect("Failed to create scan")
            .into_array_stream()
            .expect("Failed to create stream");

        let mut saw_any = false;
        while let Some(chunk) = stream.next().await {
            let _chunk = chunk.expect("Scan failed");
            saw_any = true;
        }
        assert!(saw_any);
        start.elapsed()
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
        let mut stream = file
            .scan()
            .expect("Failed to create scan")
            .into_array_stream()
            .expect("Failed to create stream");

        let mut saw_any = false;
        while let Some(chunk) = stream.next().await {
            let _chunk = chunk.expect("Scan failed");
            saw_any = true;
        }
        assert!(saw_any);
        start.elapsed()
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
        let mut stream = file
            .scan()
            .expect("Failed to create scan")
            .into_array_stream()
            .expect("Failed to create stream");

        let mut saw_any = false;
        while let Some(chunk) = stream.next().await {
            let _chunk = chunk.expect("Scan failed");
            saw_any = true;
        }

        allocator.synchronize().expect("Failed to synchronize");

        let elapsed = start.elapsed();
        assert!(saw_any);
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

fn print_stats(pinned_pool: Option<&PinnedByteBufferPool>, host_pool: Option<&HostByteBufferPool>) {
    let tasks = scan_task_stats();
    if tasks.started > 0 || tasks.completed > 0 {
        println!(
            "    Scan tasks: started={} completed={} max_active={}",
            tasks.started, tasks.completed, tasks.max_active
        );
    }
    let io = io_request_stats();
    if io.registered > 0 || io.dispatched > 0 || io.completed > 0 {
        println!(
            "    IO requests: registered={} polled={} dispatched={} completed={} max_in_flight={}",
            io.registered, io.polled, io.dispatched, io.completed, io.max_in_flight
        );
    }
    let align = alignment_copy_stats();
    println!(
        "    Alignment copies: {} ({} bytes)",
        align.count, align.bytes
    );
    let copy = copy_stats();
    if copy.bytes > 0 && copy.nanos > 0 {
        let gb_per_s = copy.bytes as f64 / (copy.nanos as f64 / 1e9) / 1e9;
        let ms = copy.nanos as f64 / 1e6;
        println!(
            "    In-memory memcpy: {:.2} GB/s ({:.2} ms total, {} copies)",
            gb_per_s, ms, copy.count
        );
    }
    let alloc = default_alloc_stats();
    if alloc.count > 0 {
        println!(
            "    Default allocs: {} ({} bytes)",
            alloc.count, alloc.bytes
        );
    }
    if let Some(pool) = host_pool {
        let stats = pool.stats();
        println!(
            "    Host pool: hits={}, misses={}, allocs={}, puts={}",
            stats.hits, stats.misses, stats.allocs, stats.puts
        );
    }
    if let Some(pool) = pinned_pool {
        let stats = pool.stats();
        println!(
            "    Pool reuse: hits={}, misses={}, allocs={}, puts={}",
            stats.hits, stats.misses, stats.allocs, stats.puts
        );
    }
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
    let host_pool = config
        .scans
        .iter()
        .any(|s| matches!(s, ScanType::DefaultCopyPooled))
        .then(|| Arc::new(HostByteBufferPool::with_fixed_alignment_pow2(Alignment::new(4096), 4)));

    let stream = ctx
        .as_ref()
        .map(|ctx| Arc::new(ctx.new_stream().expect("Failed to create stream")));
    let pool = ctx.as_ref().map(|ctx| Arc::new(PinnedByteBufferPool::new(ctx.clone())));

    let buffered_values = if config.sweep_buffered_bytes.is_empty() {
        vec![config.buffered_bytes]
    } else {
        config.sweep_buffered_bytes.clone()
    };
    let coalesce_values = if config.sweep_coalesce_min_bytes.is_empty() {
        vec![config.coalesce_min_bytes]
    } else {
        config.sweep_coalesce_min_bytes.clone()
    };
    let row_block_values = if config.sweep_row_block_rows.is_empty() {
        vec![config.row_block_size]
    } else {
        config.sweep_row_block_rows.clone()
    };
    let chunk_values = if config.sweep_chunk_rows.is_empty() {
        vec![config.chunk_rows]
    } else {
        config.sweep_chunk_rows.clone()
    };

    for num_rows in &config.rows {
        let label = format_row_label(*num_rows);
        println!("Rows: {} ({})", num_rows, label);
        for buffered_bytes in &buffered_values {
            for coalesce_min_bytes in &coalesce_values {
                for row_block_size in &row_block_values {
                    for chunk_rows in &chunk_values {
                        let params = WriteParams {
                            row_block_size: *row_block_size,
                            buffered_bytes: *buffered_bytes,
                            coalesce_min_bytes: *coalesce_min_bytes,
                            chunk_rows: *chunk_rows,
                        };
                        println!(
                            "  Params: buffered={} coalesce={} row_block={} chunk_rows={}",
                            buffered_bytes, coalesce_min_bytes, row_block_size, chunk_rows
                        );
                        let buffer = create_vortex_buffer(&rt, &session, *num_rows, params);
                        println!(
                            "  File size: {:.2} MB ({} bytes)",
                            buffer.len() as f64 / 1e6,
                            buffer.len()
                        );
                        let splits = count_splits(&rt, &session, &buffer);
                        println!("  Splits: {}", splits);

                        if config.scans.contains(&ScanType::DefaultZeroCopy) {
                            reset_default_alloc_stats();
                            reset_copy_stats();
                            reset_alignment_copy_stats();
                            reset_io_request_stats();
                            reset_scan_task_stats();
                            let total = run_scan_iters(config.iterations, || {
                                scan_default(&rt, &session, &buffer)
                            });
                            println!(
                                "  Default (zero-copy): {}",
                                format_throughput(buffer.len(), total, config.iterations)
                            );
                            print_stats(None, None);
                        }
                        if config.scans.contains(&ScanType::DefaultCopy) {
                            reset_default_alloc_stats();
                            reset_copy_stats();
                            reset_alignment_copy_stats();
                            reset_io_request_stats();
                            reset_scan_task_stats();
                            let total = run_scan_iters(config.iterations, || {
                                scan_default_copy(&rt, &session, &buffer)
                            });
                            println!(
                                "  Default (copy):      {}",
                                format_throughput(buffer.len(), total, config.iterations)
                            );
                            print_stats(None, None);
                        }
                        if config.scans.contains(&ScanType::DefaultCopyPooled) {
                            let pool = host_pool.as_ref().expect("Host pool required");
                            pool.reset_stats();
                            reset_default_alloc_stats();
                            reset_copy_stats();
                            reset_alignment_copy_stats();
                            reset_io_request_stats();
                            reset_scan_task_stats();
                            let total = run_scan_iters(config.iterations, || {
                                scan_default_copy_pooled(&rt, &session, &buffer, pool)
                            });
                            println!(
                                "  Default (copy pooled): {}",
                                format_throughput(buffer.len(), total, config.iterations)
                            );
                            print_stats(None, Some(pool));
                        }
                        if config.scans.contains(&ScanType::Pinned) {
                            let pool = pool.as_ref().expect("Pinned pool required");
                            pool.reset_stats();
                            reset_default_alloc_stats();
                            reset_copy_stats();
                            reset_alignment_copy_stats();
                            reset_io_request_stats();
                            reset_scan_task_stats();
                            let total = run_scan_iters(config.iterations, || {
                                scan_pinned(&rt, &session, &buffer, pool)
                            });
                            println!(
                                "  Pinned (host):       {}",
                                format_throughput(buffer.len(), total, config.iterations)
                            );
                            print_stats(Some(pool), None);
                        }
                        if config.scans.contains(&ScanType::Device) {
                            let pool = pool.as_ref().expect("Pinned pool required");
                            let stream = stream.as_ref().expect("CUDA stream required");
                            pool.reset_stats();
                            reset_default_alloc_stats();
                            reset_copy_stats();
                            reset_alignment_copy_stats();
                            reset_io_request_stats();
                            reset_scan_task_stats();
                            let total = run_scan_iters(config.iterations, || {
                                scan_device(&rt, &session, &buffer, pool, stream)
                            });
                            println!(
                                "  Device (H2D):        {}",
                                format_throughput(buffer.len(), total, config.iterations)
                            );
                            print_stats(Some(pool), None);
                        }
                        println!();
                    }
                }
            }
        }
    }
    ExitCode::SUCCESS
}
