// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;

use clap::Parser;
use clap::Subcommand;
use vortex_compat::check;
use vortex_compat::generate;
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
  Check fixtures (allow extra files from older versions):\n\
    vortex-compat check --dir /tmp/v0.62.0 --mode subset\n\
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
        /// subset  — directory may have extra files (skipped), all known must be present.
        ///           Best for checking old versions that may have since-removed fixtures.
        /// exact   — directory must match current fixtures 1:1. No extras, no missing.
        /// superset — directory may be missing files (skipped), no unknown files allowed.
        #[arg(long, default_value = "subset", value_name = "MODE")]
        mode: check::Mode,

        /// Fixture name substrings to exclude from checking (comma-separated).
        #[arg(long, value_delimiter = ',', value_name = "PATTERNS")]
        exclude: Vec<String>,
    },
}

fn main() -> VortexResult<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate { output, exclude } => generate::generate(&output, &exclude),
        Commands::Check { dir, mode, exclude } => check::check(&dir, mode, &exclude),
    }
}
