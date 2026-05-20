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
use vortex_bench::onpair_bench::ensure_tpch_all_parquet;
use vortex_bench::onpair_bench::run_column;
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
        /// Per-chunk uncompressed byte budgets (default 1MB,10MB,100MB).
        #[arg(long, value_delimiter = ',', default_values_t = [MB, 10 * MB, 100 * MB])]
        chunk_bytes: Vec<u64>,
        /// OnPair training thresholds.
        #[arg(long, value_delimiter = ',', default_value = "0.2")]
        threshold: Vec<f64>,
        /// Whether to consider delta-encoding the offset children (keeping the
        /// smaller of compressor-only vs delta+compressor per child, so it
        /// never regresses). Pass `true,false` to benchmark both.
        #[arg(long, value_delimiter = ',', default_value = "true,false")]
        delta_offsets: Vec<bool>,
        /// Raw-payload sample cap (default ~1GB).
        #[arg(long, default_value_t = 1_000_000_000)]
        sample_bytes: u64,
        /// Approximate per-file on-disk target (default ~200MB).
        #[arg(long, default_value_t = 200 * MB)]
        file_target_bytes: u64,
        /// Output root for the `.vortex` files.
        #[arg(long)]
        out_dir: PathBuf,
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
        Command::Run {
            parquet,
            column,
            dataset_id,
            bits,
            chunk_bytes,
            threshold,
            delta_offsets,
            sample_bytes,
            file_target_bytes,
            out_dir,
        } => {
            let results = run_column(
                &dataset_id,
                &parquet,
                &column,
                &bits,
                &chunk_bytes,
                &threshold,
                &delta_offsets,
                sample_bytes,
                file_target_bytes,
                &out_dir,
            )
            .await?;
            println!("{}", serde_json::to_string_pretty(&results)?);
        }
    }
    Ok(())
}
