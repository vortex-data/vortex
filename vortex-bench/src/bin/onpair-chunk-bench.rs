// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! OnPair chunked-array compression benchmark CLI.
//!
//! Two subcommands:
//! * `gen-tpch` — generate every TPC-H table parquet for a scale factor.
//! * `run` — sample a column, OnPair-compress it across a `bits × chunk ×
//!   threshold` matrix into Vortex files, verify the string round-trip, and
//!   print the per-cell results as JSON to stdout.
//!
//! The Python orchestrator (`benchmarks/onpair-bench/run.py`) drives this
//! binary across a registry of datasets/columns. The binary itself is
//! dataset-agnostic: point `--parquet` at any parquet file and `--column` at
//! any string column.

#![expect(clippy::print_stdout)]

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use clap::Subcommand;
use vortex_bench::onpair_bench::GpuBenchmarkConfig;
use vortex_bench::onpair_bench::ensure_tpch_all_parquet;
use vortex_bench::onpair_bench::run_column;
use vortex_bench::onpair_bench::run_vortex_gpu_decode;
use vortex_bench::setup_logging_and_tracing;

const MB: u64 = 1 << 20;

#[derive(Parser)]
#[command(name = "onpair-chunk-bench")]
#[command(about = "OnPair chunked-array compression benchmark")]
struct Args {
    #[command(subcommand)]
    command: Command,

    /// Enable verbose logging.
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Generate every TPC-H table parquet for a scale factor (idempotent).
    GenTpch {
        /// Scale factor (e.g. 10).
        #[arg(long, default_value_t = 10.0)]
        sf: f64,
        /// Output directory; tables land in `<out-dir>/parquet/<table>_0.parquet`.
        #[arg(long)]
        out_dir: PathBuf,
    },
    /// Generate every TPC-DS table parquet for a scale factor via DuckDB
    /// `dsdgen` (idempotent). Requires the `duckdb` CLI on PATH.
    GenTpcds {
        /// Scale factor (e.g. 10).
        #[arg(long, default_value_t = 10.0)]
        sf: f64,
        /// Output directory; tables land in `<out-dir>/parquet/<table>.parquet`.
        #[arg(long)]
        out_dir: PathBuf,
    },
    /// Compress one column across the matrix and emit JSON results.
    Run {
        /// Source parquet file.
        #[arg(long)]
        parquet: PathBuf,
        /// String column to compress.
        #[arg(long)]
        column: String,
        /// Stable dataset id used in the output path and results.
        #[arg(long)]
        dataset_id: String,
        /// OnPair dictionary bit widths.
        #[arg(long, value_delimiter = ',', default_value = "12,16")]
        bits: Vec<u32>,
        /// Per-chunk uncompressed byte budgets (default 1MB,10MB,100MB,1000MB).
        #[arg(long, value_delimiter = ',', default_values_t = [MB, 10 * MB, 100 * MB, 1000 * MB])]
        chunk_bytes: Vec<u64>,
        /// OnPair training thresholds.
        #[arg(long, value_delimiter = ',', default_value = "0.2")]
        threshold: Vec<f64>,
        /// Raw-payload sample cap (default ~1GB).
        #[arg(long, default_value_t = 1_000_000_000)]
        sample_bytes: u64,
        /// Approximate per-file on-disk target (default ~200MB).
        #[arg(long, default_value_t = 200 * MB)]
        file_target_bytes: u64,
        /// Output root for the `.vortex` files.
        #[arg(long)]
        out_dir: PathBuf,
        /// Also benchmark CUDA kernel-only OnPair decompression.
        #[arg(long)]
        gpu_decode: bool,
        /// Timed CUDA iterations for each applicable kernel.
        #[arg(long, default_value_t = 10)]
        gpu_iters: u64,
        /// Copy GPU output bytes back and compare every applicable kernel against CPU decode.
        #[arg(long)]
        gpu_validate: bool,
    },
    /// Run CUDA OnPair decode directly from existing `.vortex` files.
    GpuDecodeVortex {
        /// Existing `.vortex` file or directory containing `*.vortex` parts.
        #[arg(long, value_name = "FILE_OR_DIR", required = true)]
        vortex: Vec<PathBuf>,
        /// OnPair string column to extract from each file.
        #[arg(long)]
        column: String,
        /// Timed CUDA iterations for each applicable kernel.
        #[arg(long, default_value_t = 10)]
        gpu_iters: u64,
        /// Copy GPU output bytes back and compare every applicable kernel against CPU decode.
        #[arg(long)]
        gpu_validate: bool,
    },
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();
    setup_logging_and_tracing(args.verbose, false)?;

    match args.command {
        Command::GenTpch { sf, out_dir } => {
            ensure_tpch_all_parquet(sf, &out_dir).await?;
            eprintln!("TPC-H (sf={sf}) ready under {}/parquet", out_dir.display());
        }
        Command::GenTpcds { sf, out_dir } => {
            vortex_bench::tpcds::duckdb::generate_tpcds(out_dir.clone(), format!("{sf}"))?;
            eprintln!("TPC-DS (sf={sf}) ready under {}/parquet", out_dir.display());
        }
        Command::Run {
            parquet,
            column,
            dataset_id,
            bits,
            chunk_bytes,
            threshold,
            sample_bytes,
            file_target_bytes,
            out_dir,
            gpu_decode,
            gpu_iters,
            gpu_validate,
        } => {
            let results = run_column(
                &dataset_id,
                &parquet,
                &column,
                &bits,
                &chunk_bytes,
                &threshold,
                sample_bytes,
                file_target_bytes,
                &out_dir,
                gpu_decode.then_some(GpuBenchmarkConfig {
                    iterations: gpu_iters,
                    validate: gpu_validate,
                }),
            )
            .await?;
            println!("{}", serde_json::to_string_pretty(&results)?);
        }
        Command::GpuDecodeVortex {
            vortex,
            column,
            gpu_iters,
            gpu_validate,
        } => {
            let files = collect_vortex_files(&vortex)?;
            let result = run_vortex_gpu_decode(
                &files,
                &column,
                GpuBenchmarkConfig {
                    iterations: gpu_iters,
                    validate: gpu_validate,
                },
            )
            .await?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
    }
    Ok(())
}

fn collect_vortex_files(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for path in paths {
        if path.is_dir() {
            let mut entries = std::fs::read_dir(path)?
                .map(|entry| entry.map(|e| e.path()))
                .collect::<std::io::Result<Vec<_>>>()?;
            entries.sort();
            files.extend(entries.into_iter().filter(|p| {
                p.extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| ext == "vortex")
            }));
        } else {
            files.push(path.clone());
        }
    }

    if files.is_empty() {
        anyhow::bail!("no .vortex files found");
    }
    Ok(files)
}
