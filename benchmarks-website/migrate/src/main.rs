// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `vortex-bench-migrate` CLI: a one-shot historical migrator from
//! v2's S3 dataset into a v3 DuckDB file, plus a structural diff
//! against the live v2 `/api/metadata` endpoint for spotting
//! classifier regressions.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Context as _;
use anyhow::Result;
use clap::Parser;
use clap::Subcommand;
use clap::ValueEnum;
use tracing_subscriber::EnvFilter;
use vortex_bench_migrate::migrate;
use vortex_bench_migrate::source::Source;
use vortex_bench_migrate::verify;

/// One-shot historical migrator from v2's S3 dataset to v3 DuckDB.
#[derive(Debug, Parser)]
#[command(name = "vortex-bench-migrate", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Read v2's data.json.gz / commits.json / file-sizes-*.json.gz
    /// and write a fully populated v3 DuckDB at `--output`.
    Run {
        /// Path to write the v3 DuckDB to. Created if absent.
        #[arg(long)]
        output: PathBuf,
        /// Where to fetch v2 dumps from.
        #[arg(long, value_enum, default_value_t = SourceKind::PublicS3)]
        source: SourceKind,
        /// For `--source=local`, the directory containing
        /// `data.json.gz`, `commits.json`, and `file-sizes-*.json.gz`.
        #[arg(long, required_if_eq("source", "local"))]
        source_dir: Option<PathBuf>,
        /// Continue past per-`file-sizes-*.json.gz` failures rather than
        /// failing the migration. By default a single failed
        /// `file-sizes-*` source is an error, because a "successful"
        /// migrated DB with missing compression-size history is a worse
        /// outcome than a loud failure that the operator can retry. Pass
        /// this flag when you genuinely want partial coverage (e.g. one
        /// known-bad source file you want to skip).
        #[arg(long, default_value_t = false)]
        allow_missing_file_sizes: bool,
    },
    /// Diff a migrated DuckDB against the live v2 `/api/metadata`
    /// endpoint. Exits 0 if every v2 group is present in v3, 1
    /// otherwise so this can gate a CI step.
    Verify {
        /// HTTPS root of a running v2 server (e.g. `https://bench.vortex.dev`).
        #[arg(long)]
        against: String,
        /// Path to the migrated v3 DuckDB.
        #[arg(long)]
        duckdb: PathBuf,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SourceKind {
    PublicS3,
    Local,
}

fn main() -> ExitCode {
    if let Err(err) = run() {
        eprintln!("error: {err:#}");
        return ExitCode::from(2);
    }
    ExitCode::SUCCESS
}

fn run() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_env("VORTEX_BENCH_LOG").unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Run {
            output,
            source,
            source_dir,
            allow_missing_file_sizes,
        } => {
            let source = match source {
                SourceKind::PublicS3 => Source::PublicS3,
                SourceKind::Local => {
                    Source::Local(source_dir.context("--source=local requires --source-dir")?)
                }
            };
            let summary = migrate::run(&source, &output)?;
            print!("{summary}");
            if summary.uncategorized_fraction() > 0.05 {
                anyhow::bail!(
                    "uncategorized records ({:.2}%) exceed the 5% gate; \
                     stop and report unmatched prefixes (see summary above) \
                     before proceeding",
                    100.0 * summary.uncategorized_fraction()
                );
            }
            if summary.file_sizes_failed > 0 && !allow_missing_file_sizes {
                anyhow::bail!(
                    "{} file-sizes-*.json.gz source file(s) failed (see warnings above); \
                     re-run with --allow-missing-file-sizes if partial coverage is intended",
                    summary.file_sizes_failed
                );
            }
            Ok(())
        }
        Command::Verify { against, duckdb } => {
            let report = verify::run(&against, &duckdb)?;
            print!("{report}");
            if !report.v2_groups_covered() {
                std::process::exit(1);
            }
            Ok(())
        }
    }
}
