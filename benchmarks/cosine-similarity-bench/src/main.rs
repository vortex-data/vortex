//! Cosine similarity scan benchmark.
//!
//! One binary with three subcommands:
//!  * `generate`   - write a synthetic corpus file of unit-normalized f32 vectors.
//!  * `scan-local` - stream a local corpus file through the dot-product kernel.
//!  * `scan-s3`    - stream an S3 object through the dot-product kernel.
//!
//! See the crate README for the design of the IO path and kernel.
#![allow(clippy::too_many_arguments)]

use std::path::PathBuf;

use cosine_similarity_bench::generate;
use cosine_similarity_bench::kernel;
use cosine_similarity_bench::metrics;
use cosine_similarity_bench::scan_local;
use cosine_similarity_bench::scan_s3;

use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use clap::Subcommand;

use crate::kernel::DotKernel;
use crate::metrics::IterationResult;
use crate::metrics::RunSummary;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Generate a synthetic corpus file.
    Generate {
        /// Output path.
        #[arg(short, long)]
        out: PathBuf,
        /// Vector dimension.
        #[arg(short, long, default_value_t = 1024)]
        dim: usize,
        /// Number of vectors.
        #[arg(short = 'n', long)]
        n_vectors: Option<u64>,
        /// Target size in bytes (alternative to --n-vectors).
        #[arg(long, value_parser = parse_size)]
        size_bytes: Option<u64>,
        /// RNG seed.
        #[arg(long, default_value_t = 0xc0ffee)]
        seed: u64,
    },
    /// Scan a local corpus file.
    ScanLocal {
        /// Corpus file path.
        #[arg(short, long)]
        path: PathBuf,
        /// Vector dimension.
        #[arg(short, long, default_value_t = 1024)]
        dim: usize,
        /// Number of worker threads. If absent, sweep 1..=2x physical cores.
        #[arg(short, long)]
        threads: Option<usize>,
        /// Per-chunk read size in bytes. Must be a multiple of 4096 and of
        /// dim*4.
        #[arg(long, value_parser = parse_size, default_value = "4MiB")]
        chunk_bytes: u64,
        /// Disable O_DIRECT/F_NOCACHE (use the OS page cache).
        #[arg(long)]
        no_direct: bool,
        /// Warmup iterations (not reported).
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        /// Measured iterations.
        #[arg(long, default_value_t = 5)]
        iters: usize,
        /// Query seed.
        #[arg(long, default_value_t = 42)]
        query_seed: u64,
    },
    /// Scan an S3 object.
    ScanS3 {
        /// S3 bucket name.
        #[arg(long)]
        bucket: String,
        /// Object key.
        #[arg(long)]
        key: String,
        /// Vector dimension.
        #[arg(short, long, default_value_t = 1024)]
        dim: usize,
        /// Sweep of concurrency values, comma separated. If absent, sweep
        /// powers of 2 from 1 to 256.
        #[arg(long, value_delimiter = ',')]
        concurrency: Option<Vec<usize>>,
        /// Range request size in bytes. Must be a multiple of dim*4.
        #[arg(long, value_parser = parse_size, default_value = "8MiB")]
        range_bytes: u64,
        /// Warmup iterations (not reported).
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        /// Measured iterations.
        #[arg(long, default_value_t = 5)]
        iters: usize,
        /// Query seed.
        #[arg(long, default_value_t = 42)]
        query_seed: u64,
    },
}

fn parse_size(s: &str) -> Result<u64, String> {
    let s = s.trim();
    let (num, mult) = if let Some(v) = s.strip_suffix("GiB") {
        (v, 1u64 << 30)
    } else if let Some(v) = s.strip_suffix("MiB") {
        (v, 1u64 << 20)
    } else if let Some(v) = s.strip_suffix("KiB") {
        (v, 1u64 << 10)
    } else if let Some(v) = s.strip_suffix("GB") {
        (v, 1_000_000_000)
    } else if let Some(v) = s.strip_suffix("MB") {
        (v, 1_000_000)
    } else if let Some(v) = s.strip_suffix("KB") {
        (v, 1_000)
    } else if let Some(v) = s.strip_suffix('B') {
        (v, 1)
    } else {
        (s, 1)
    };
    let n: f64 = num
        .trim()
        .parse()
        .map_err(|e| format!("invalid size: {e}"))?;
    Ok((n * mult as f64) as u64)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Generate {
            out,
            dim,
            n_vectors,
            size_bytes,
            seed,
        } => cmd_generate(&out, dim, n_vectors, size_bytes, seed),
        Cmd::ScanLocal {
            path,
            dim,
            threads,
            chunk_bytes,
            no_direct,
            warmup,
            iters,
            query_seed,
        } => cmd_scan_local(
            &path,
            dim,
            threads,
            chunk_bytes as usize,
            !no_direct,
            warmup,
            iters,
            query_seed,
        ),
        Cmd::ScanS3 {
            bucket,
            key,
            dim,
            concurrency,
            range_bytes,
            warmup,
            iters,
            query_seed,
        } => cmd_scan_s3(
            &bucket,
            &key,
            dim,
            concurrency,
            range_bytes,
            warmup,
            iters,
            query_seed,
        ),
    }
}

fn cmd_generate(
    out: &std::path::Path,
    dim: usize,
    n_vectors: Option<u64>,
    size_bytes: Option<u64>,
    seed: u64,
) -> Result<()> {
    let n = match (n_vectors, size_bytes) {
        (Some(n), None) => n,
        (None, Some(bytes)) => {
            let per_vec = dim as u64 * 4;
            anyhow::ensure!(per_vec > 0, "dim must be positive");
            bytes / per_vec
        }
        (Some(_), Some(_)) => anyhow::bail!("pass either --n-vectors or --size-bytes, not both"),
        (None, None) => anyhow::bail!("pass either --n-vectors or --size-bytes"),
    };
    println!(
        "generating {} vectors x {} dims = {:.2} GiB to {}",
        n,
        dim,
        (n as f64 * dim as f64 * 4.0) / (1024.0 * 1024.0 * 1024.0),
        out.display()
    );
    let t0 = std::time::Instant::now();
    let bytes = generate::generate(out, n, dim, seed)?;
    let elapsed = t0.elapsed();
    println!(
        "wrote {} bytes in {:.2}s ({:.2} GB/s write)",
        bytes,
        elapsed.as_secs_f64(),
        (bytes as f64) / elapsed.as_secs_f64() / 1e9,
    );
    Ok(())
}

fn physical_cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
    // treat the logical count as an upper bound; on SMT systems this is
    // ~2x the physical cores but in practice for a memory-hungry scan we
    // want to sweep to 2x logical since contention is limited.
}

fn cmd_scan_local(
    path: &std::path::Path,
    dim: usize,
    threads: Option<usize>,
    chunk_bytes: usize,
    direct: bool,
    warmup: usize,
    iters: usize,
    query_seed: u64,
) -> Result<()> {
    let kernel = DotKernel::detect();
    println!("kernel: {}", kernel.name());
    let query = generate::random_unit_vector(dim, query_seed);

    let thread_sweep: Vec<usize> = match threads {
        Some(t) => vec![t],
        None => {
            let max = (physical_cores() * 2).max(2);
            let mut v = vec![1usize];
            let mut t = 2;
            while t < max {
                v.push(t);
                t *= 2;
            }
            v.push(max);
            v.sort_unstable();
            v.dedup();
            v
        }
    };

    for t in thread_sweep {
        let cfg = scan_local::LocalScanConfig {
            path: path.to_path_buf(),
            dim,
            query: query.clone(),
            threads: t,
            chunk_bytes,
            kernel,
            direct,
        };
        for _ in 0..warmup {
            let _ = scan_local::run_once(&cfg).context("warmup")?;
        }
        let mut iter_results: Vec<IterationResult> = Vec::with_capacity(iters);
        for _ in 0..iters {
            iter_results.push(scan_local::run_once(&cfg).context("measurement")?);
        }
        let summary = RunSummary {
            iters: iter_results,
        };
        summary.report(&format!(
            "scan-local threads={t} chunk={chunk_bytes} direct={direct}"
        ));
    }
    Ok(())
}

fn cmd_scan_s3(
    bucket: &str,
    key: &str,
    dim: usize,
    concurrency: Option<Vec<usize>>,
    range_bytes: u64,
    warmup: usize,
    iters: usize,
    query_seed: u64,
) -> Result<()> {
    let kernel = DotKernel::detect();
    println!("kernel: {}", kernel.name());
    let query = generate::random_unit_vector(dim, query_seed);

    let sweep = concurrency.unwrap_or_else(|| vec![1, 2, 4, 8, 16, 32, 64, 128, 256]);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("tokio runtime")?;

    rt.block_on(async move {
        let sdk_cfg = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
        let client = aws_sdk_s3::Client::new(&sdk_cfg);

        for c in sweep {
            let cfg = scan_s3::S3ScanConfig {
                bucket: bucket.to_string(),
                key: key.to_string(),
                dim,
                query: query.clone(),
                concurrency: c,
                range_bytes,
                kernel,
            };
            for _ in 0..warmup {
                let _ = scan_s3::run_once(&client, &cfg).await.context("warmup")?;
            }
            let mut iter_results: Vec<IterationResult> = Vec::with_capacity(iters);
            for _ in 0..iters {
                iter_results.push(
                    scan_s3::run_once(&client, &cfg)
                        .await
                        .context("measurement")?,
                );
            }
            RunSummary {
                iters: iter_results,
            }
            .report(&format!("scan-s3 concurrency={c} range={range_bytes}"));
        }
        Ok::<_, anyhow::Error>(())
    })?;

    Ok(())
}
