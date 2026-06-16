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
use vortex_bench_migrate::postgres;
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
    /// Verify a migrated v3 DuckDB. Two modes: `--against` diffs it against the
    /// live v2 `/api/metadata` endpoint (group / chart structure); or
    /// `--postgres-target` value-verifies it against a loaded Postgres target per
    /// `measurement_id` (the PR-3.2 primary v4-correctness gate). Exactly one mode
    /// is required; exits non-zero on a diff so this can gate a CI step.
    Verify {
        /// Structural-diff mode: HTTPS root of a running v2 server (e.g.
        /// `https://bench.vortex.dev`). Mutually exclusive with `--postgres-target`.
        #[arg(long)]
        against: Option<String>,
        /// Path to the v3 DuckDB to verify (the migrated DB for `--against`, or the
        /// loaded source snapshot for `--postgres-target`).
        #[arg(long)]
        duckdb: PathBuf,
        /// Value-verify mode: Postgres DSN whose loaded rows are compared against
        /// `--duckdb` per `measurement_id`. Mutually exclusive with `--against`.
        #[arg(long)]
        postgres_target: Option<String>,
        /// PEM CA bundle for a host-verifying TLS connection to `--postgres-target`
        /// (the RDS CA for the prod check). Omit for a plaintext local connection.
        #[arg(long, requires = "postgres_target")]
        ca_cert: Option<PathBuf>,
    },
    /// Bulk-load an existing v3 DuckDB snapshot into Postgres in one
    /// atomic transaction (the v3 -> v4 historical-data load). Reads each
    /// table and `COPY`s it; `measurement_id` is preserved verbatim.
    Load {
        /// Path to the source v3 DuckDB snapshot (read-only input).
        #[arg(long)]
        duckdb: PathBuf,
        /// Postgres connection string to load into. For the prod RDS load this
        /// is the operator-local master-password DSN (`sslmode=require` + a
        /// `--ca-cert`); for the local rehearsal a plain `postgresql://.../db`.
        #[arg(long)]
        postgres_target: String,
        /// PEM CA bundle to trust for a host-verifying TLS connection (the RDS
        /// CA for the prod load). Omit for a plaintext local connection.
        #[arg(long)]
        ca_cert: Option<PathBuf>,
        /// Empty every target table inside the load transaction before the COPYs,
        /// making the load an atomic full replace instead of an append. Required
        /// when re-loading into an already-populated target (the data-refresh /
        /// re-migration path); without it a re-load aborts on the first duplicate
        /// `measurement_id`. `TRUNCATE` needs table ownership, so `--replace` must
        /// connect as the table owner (the RDS master), not `migrator`.
        #[arg(long, default_value_t = false)]
        replace: bool,
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
        Command::Verify {
            against,
            duckdb,
            postgres_target,
            ca_cert,
        } => match (against, postgres_target) {
            (Some(server), None) => {
                let report = verify::run(&server, &duckdb)?;
                print!("{report}");
                if !report.v2_groups_covered() {
                    std::process::exit(1);
                }
                Ok(())
            }
            (None, Some(dsn)) => {
                let report = verify::run_postgres_value_verify(&duckdb, &dsn, ca_cert.as_deref())?;
                print!("{report}");
                if !report.is_clean() {
                    std::process::exit(1);
                }
                Ok(())
            }
            (Some(_), Some(_)) => {
                anyhow::bail!("--against and --postgres-target are mutually exclusive")
            }
            (None, None) => anyhow::bail!(
                "verify requires exactly one of --against (v2 structural diff) or \
                 --postgres-target (DuckDB -> Postgres value verify)"
            ),
        },
        Command::Load {
            duckdb,
            postgres_target,
            ca_cert,
            replace,
        } => {
            let summary = postgres::load(&duckdb, &postgres_target, ca_cert.as_deref(), replace)?;
            print!("{summary}");
            Ok(())
        }
    }
}
