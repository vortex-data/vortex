// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmark H2D throughput with pinned vs regular memory.
//!
//! Example:
//!   cargo run -p vortex-cuda --bin h2d_pinned -- --sizes 1MB,16MB,64MB --iters 10 --mode pinned

#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#![allow(clippy::redundant_clone)]

use std::env;
use std::fmt;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use cudarc::driver::CudaContext;
use vortex_cuda::PinnedByteBufferPool;
use vortex_cuda::has_nvcc;

const DEFAULT_SIZES: &[usize] = &[
    1 << 10,  // 1KB
    1 << 16,  // 64KB
    1 << 20,  // 1MB
    16 << 20, // 16MB
    64 << 20, // 64MB
    256 << 20, // 256MB
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Mode {
    Pinned,
    Regular,
    PooledPinned,
    RamPinnedGpu,
    RamGpuDirect,
}

struct Config {
    sizes: Vec<usize>,
    iterations: usize,
    modes: Vec<Mode>,
}

fn usage() -> &'static str {
    "Usage: h2d_pinned [--sizes S1,S2] [--iters N] [--mode NAME]\n\
\n\
Flags:\n\
  --sizes   Comma-separated sizes (e.g., 1MB,16MB,67108864)\n\
  --iters   Iterations per size (default: 10)\n\
  --mode    One of: pinned, regular, pooled, ram_pinned_gpu, ram_gpu_direct, all\n"
}

fn parse_sizes(value: &str) -> Vec<usize> {
    value
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .filter_map(parse_size)
        .filter(|v| *v > 0)
        .collect()
}

fn parse_size(value: &str) -> Option<usize> {
    let value = value.trim();
    let value = value.to_lowercase();
    let (num, suffix) = value
        .chars()
        .position(|c| !c.is_ascii_digit())
        .map(|idx| value.split_at(idx))
        .unwrap_or((value.as_str(), ""));

    let base = num.parse::<usize>().ok()?;
    let bytes = match suffix {
        "" => base,
        "k" | "kb" | "kib" => base.saturating_mul(1 << 10),
        "m" | "mb" | "mib" => base.saturating_mul(1 << 20),
        "g" | "gb" | "gib" => base.saturating_mul(1 << 30),
        _ => return None,
    };
    Some(bytes)
}

fn parse_modes(value: &str) -> Vec<Mode> {
    let mut modes = Vec::new();
    for item in value.split(',').map(|s| s.trim().to_lowercase()) {
        match item.as_str() {
            "pinned" => modes.push(Mode::Pinned),
            "regular" => modes.push(Mode::Regular),
            "pooled" | "pooled_pinned" => modes.push(Mode::PooledPinned),
            "ram_pinned_gpu" | "ram_to_pinned_to_gpu" => modes.push(Mode::RamPinnedGpu),
            "ram_gpu_direct" | "ram_to_gpu_direct" => modes.push(Mode::RamGpuDirect),
            "all" => {
                modes = vec![
                    Mode::Pinned,
                    Mode::Regular,
                    Mode::PooledPinned,
                    Mode::RamPinnedGpu,
                    Mode::RamGpuDirect,
                ];
                break;
            }
            _ => {}
        }
    }
    modes.sort();
    modes.dedup();
    modes
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Mode::Pinned => "pinned",
            Mode::Regular => "regular",
            Mode::PooledPinned => "pooled",
            Mode::RamPinnedGpu => "ram_pinned_gpu",
            Mode::RamGpuDirect => "ram_gpu_direct",
        };
        f.write_str(name)
    }
}

fn parse_args() -> Result<Config, String> {
    let mut sizes: Option<Vec<usize>> = None;
    let mut iterations: usize = 10;
    let mut modes: Option<Vec<Mode>> = None;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--sizes" => {
                let value = args.next().ok_or_else(|| "Missing value for --sizes".to_string())?;
                sizes = Some(parse_sizes(&value));
            }
            "--iters" => {
                let value = args.next().ok_or_else(|| "Missing value for --iters".to_string())?;
                iterations = value
                    .parse::<usize>()
                    .map_err(|_| "Invalid value for --iters".to_string())?;
            }
            "--mode" => {
                let value = args.next().ok_or_else(|| "Missing value for --mode".to_string())?;
                modes = Some(parse_modes(&value));
            }
            "--help" | "-h" => return Err(usage().to_string()),
            _ if arg.starts_with("--sizes=") => {
                let value = arg.trim_start_matches("--sizes=");
                sizes = Some(parse_sizes(value));
            }
            _ if arg.starts_with("--iters=") => {
                let value = arg.trim_start_matches("--iters=");
                iterations = value
                    .parse::<usize>()
                    .map_err(|_| "Invalid value for --iters".to_string())?;
            }
            _ if arg.starts_with("--mode=") => {
                let value = arg.trim_start_matches("--mode=");
                modes = Some(parse_modes(value));
            }
            _ => return Err(format!("Unknown argument: {arg}\n{}", usage())),
        }
    }

    let sizes = sizes.filter(|v| !v.is_empty()).unwrap_or_else(|| DEFAULT_SIZES.to_vec());
    let modes = modes.filter(|v| !v.is_empty()).unwrap_or_else(|| {
        vec![
            Mode::Pinned,
            Mode::Regular,
            Mode::PooledPinned,
            Mode::RamPinnedGpu,
            Mode::RamGpuDirect,
        ]
    });
    if iterations == 0 {
        return Err("Iterations must be > 0".to_string());
    }

    Ok(Config {
        sizes,
        iterations,
        modes,
    })
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

fn run_iters<F>(iterations: usize, mut f: F) -> Duration
where
    F: FnMut(),
{
    let mut total = Duration::ZERO;
    for _ in 0..iterations {
        let start = Instant::now();
        f();
        total += start.elapsed();
    }
    total
}

fn main() -> ExitCode {
    let config = match parse_args() {
        Ok(config) => config,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::FAILURE;
        }
    };

    if !has_nvcc() {
        eprintln!("nvcc not found, skipping CUDA benchmarks");
        return ExitCode::FAILURE;
    }

    println!("\n=== H2D Throughput (binary) ===\n");
    println!("Iterations: {}", config.iterations);
    let mode_list = config
        .modes
        .iter()
        .map(|mode| mode.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    println!("Modes: {}\n", mode_list);

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.new_stream().expect("Failed to create stream");
    let pool = Arc::new(PinnedByteBufferPool::new(ctx.clone()));

    for size in &config.sizes {
        println!("Size: {} bytes", size);

        if config.modes.contains(&Mode::Pinned) {
            let mut pinned = pool.get(*size).expect("Failed to get pinned buffer");
            pinned.as_mut_slice().expect("slice").fill(0x42);
            let mut device = unsafe { stream.alloc::<u8>(*size) }.expect("Failed to alloc device");
            let total = run_iters(config.iterations, || {
                stream
                    .memcpy_htod(&pinned, &mut device)
                    .expect("H2D failed");
                stream.synchronize().expect("Sync failed");
            });
            println!(
                "  Pinned:           {}",
                format_throughput(*size, total, config.iterations)
            );
            pool.put(pinned).ok();
        }

        if config.modes.contains(&Mode::Regular) {
            let data: Vec<u8> = vec![0x42u8; *size];
            let mut device = unsafe { stream.alloc::<u8>(*size) }.expect("Failed to alloc device");
            let total = run_iters(config.iterations, || {
                stream.memcpy_htod(&data, &mut device).expect("H2D failed");
                stream.synchronize().expect("Sync failed");
            });
            println!(
                "  Regular:          {}",
                format_throughput(*size, total, config.iterations)
            );
        }

        if config.modes.contains(&Mode::PooledPinned) {
            let mut device = unsafe { stream.alloc::<u8>(*size) }.expect("Failed to alloc device");
            let total = run_iters(config.iterations, || {
                let mut pooled = pool.get_pooled(*size).expect("Failed to get pooled buffer");
                pooled.as_mut_slice().fill(0x42);
                stream
                    .memcpy_htod(&pooled, &mut device)
                    .expect("H2D failed");
                stream.synchronize().expect("Sync failed");
            });
            println!(
                "  Pooled pinned:    {}",
                format_throughput(*size, total, config.iterations)
            );
        }

        if config.modes.contains(&Mode::RamPinnedGpu) {
            let ram_data: Vec<u8> = vec![0x42u8; *size];
            let mut device = unsafe { stream.alloc::<u8>(*size) }.expect("Failed to alloc device");
            let total = run_iters(config.iterations, || {
                let mut pinned = pool.get_pooled(*size).expect("Failed to get pooled buffer");
                pinned.as_mut_slice().copy_from_slice(&ram_data);
                stream
                    .memcpy_htod(&pinned, &mut device)
                    .expect("H2D failed");
                stream.synchronize().expect("Sync failed");
            });
            println!(
                "  RAM->Pinned->GPU: {}",
                format_throughput(*size, total, config.iterations)
            );
        }

        if config.modes.contains(&Mode::RamGpuDirect) {
            let ram_data: Vec<u8> = vec![0x42u8; *size];
            let mut device = unsafe { stream.alloc::<u8>(*size) }.expect("Failed to alloc device");
            let total = run_iters(config.iterations, || {
                stream
                    .memcpy_htod(&ram_data, &mut device)
                    .expect("H2D failed");
                stream.synchronize().expect("Sync failed");
            });
            println!(
                "  RAM->GPU direct:  {}",
                format_throughput(*size, total, config.iterations)
            );
        }

        println!();
    }
    ExitCode::SUCCESS
}
