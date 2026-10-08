// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;

use clap::Parser;
use clap::Subcommand;
use vortex_compat::adapter;
use vortex_compat::check;
use vortex_compat::describe;
use vortex_compat::generate;
use vortex_compat::reader_check;
use vortex_compat::sweep;
use vortex_error::VortexResult;

#[derive(Parser)]
#[command(
    name = "vortex-compat",
    about = "Generate and check Vortex backward-compatibility fixtures",
    long_about = "\
Thin Rust binary for backward-compatibility testing.\n\
\n\
This tool generates .vortex fixture files from in-memory test data and \
checks that existing .vortex files can still be read and match expectations. \
It is designed to be called by the compat.py orchestrator, which handles \
versioning, S3 storage, and manifest management.\n\
\n\
Output protocol:\n\
  - Progress / diagnostics go to stderr\n\
  - Structured JSON results go to stdout (check command only)",
    after_help = "\
EXAMPLES:\n\
  Generate fixtures into a directory:\n\
    vortex-compat generate --output /tmp/fixtures\n\
\n\
  Check fixtures (default, old dir may be missing new fixtures):\n\
    vortex-compat check --dir /tmp/v0.62.0\n\
\n\
  Check fixtures (strict, must match exactly):\n\
    vortex-compat check --dir /tmp/v0.63.0 --mode exact\n\
\n\
  Build and run:\n\
    cargo run -p vortex-compat --release -- generate --output ./out"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate all fixture files into a directory.
    ///
    /// Writes one .vortex file per fixture plus a fixtures.json manifest
    /// listing all generated files. The output directory is created if needed.
    ///
    /// Progress is printed to stderr. On success, the output directory
    /// contains everything needed for `check` to validate.
    Generate {
        /// Output directory for .vortex files and fixtures.json.
        #[arg(long, value_name = "DIR")]
        output: PathBuf,

        /// Fixture name substrings to exclude (comma-separated, e.g. "clickbench,tpch").
        #[arg(long, value_delimiter = ',', value_name = "PATTERNS")]
        exclude: Vec<String>,

        /// Edition to write the fixtures with, e.g. `core2025.10.0`. Defaults to the session's
        /// default edition. Fixtures the edition forbids are skipped.
        #[arg(long, value_name = "EDITION")]
        edition: Option<String>,
    },

    /// Check .vortex files in a directory against in-memory fixtures.
    ///
    /// For each .vortex file, rebuilds the expected array from current code
    /// and compares it to the file contents. Results are printed as JSON to
    /// stdout (for machine consumption) and as human-readable summaries to
    /// stderr.
    ///
    /// The --mode flag controls how mismatches between directory contents
    /// and the current fixture set are handled.
    Check {
        /// Directory containing .vortex files to check.
        #[arg(long, value_name = "DIR")]
        dir: PathBuf,

        /// How to handle mismatches between directory contents and known fixtures.
        ///
        /// superset — directory may be missing files (skipped), no unknown files allowed.
        ///            Best for checking old versions that predate newly-added fixtures.
        /// exact    — directory must match current fixtures 1:1. No extras, no missing.
        /// subset   — directory may have extra files (skipped), all known must be present.
        #[arg(long, default_value = "superset", value_name = "MODE")]
        mode: check::Mode,

        /// Fixture name substrings to exclude from checking (comma-separated).
        #[arg(long, value_delimiter = ',', value_name = "PATTERNS")]
        exclude: Vec<String>,
    },

    /// Write a seeded random sweep of files: random canonical arrays through the flat layout and
    /// both compressor pipelines, plus randomly built dict, constant and run-end arrays.
    ///
    /// Honours --edition the same way `generate` does. Files the edition rejects are skipped.
    Sweep {
        /// Output directory for the .vortex files and sweep.json.
        #[arg(long, value_name = "DIR")]
        output: PathBuf,

        /// First seed. Each seed yields up to six files.
        #[arg(long, default_value_t = 0)]
        first_seed: u64,

        /// Number of seeds.
        #[arg(long, default_value_t = 100)]
        seeds: u64,

        /// Maximum row count of the random canonical array.
        #[arg(long, default_value_t = 4096)]
        max_len: usize,

        /// Edition to write with, e.g. `core2025.10.0`.
        #[arg(long, value_name = "EDITION")]
        edition: Option<String>,
    },

    /// Print the dtype, layout IDs and array encoding IDs of each .vortex file in a directory.
    Describe {
        /// Directory containing .vortex files.
        #[arg(long, value_name = "DIR")]
        dir: PathBuf,

        /// Only describe files whose name contains this substring.
        #[arg(long)]
        filter: Option<String>,
    },

    /// Check that an old reader decoded the fixtures in a directory to the same values as the
    /// current reader.
    ///
    /// The old reader is a separate binary built against a released vortex crate. It dumps each
    /// fixture as `<name>.arrow` (Arrow IPC), or `<name>.error` when it could not read the file.
    CheckReader {
        /// Directory containing the .vortex fixtures written by the current writer.
        #[arg(long, value_name = "DIR")]
        dir: PathBuf,

        /// Directory containing the old reader's `<name>.arrow` / `<name>.error` dumps.
        #[arg(long, value_name = "DIR")]
        arrow_dir: PathBuf,
    },
}

fn main() -> VortexResult<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            output,
            exclude,
            edition,
        } => {
            if let Some(edition) = edition {
                adapter::set_target_edition(&edition)?;
            }
            generate::generate(&output, &exclude)
        }
        Commands::Check { dir, mode, exclude } => check::check(&dir, mode, &exclude),
        Commands::Sweep {
            output,
            first_seed,
            seeds,
            max_len,
            edition,
        } => {
            if let Some(edition) = edition {
                adapter::set_target_edition(&edition)?;
            }
            sweep::sweep(&output, first_seed, seeds, max_len)
        }
        Commands::Describe { dir, filter } => describe::describe(&dir, filter.as_deref()),
        Commands::CheckReader { dir, arrow_dir } => reader_check::check_reader(&dir, &arrow_dir),
    }
}
