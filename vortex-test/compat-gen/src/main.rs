// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;

use clap::Parser;
use clap::Subcommand;
use clap::ValueEnum;
use vortex_compat::check;
use vortex_compat::generate;
use vortex_error::VortexResult;

#[derive(Parser)]
#[command(
    name = "vortex-compat",
    about = "Generate and check Vortex backward-compatibility fixtures"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate all fixture files into a directory.
    Generate {
        /// Output directory for .vortex files and fixtures.json.
        #[arg(long)]
        output: PathBuf,
    },

    /// Check .vortex files in a directory against in-memory fixtures.
    Check {
        /// Directory containing .vortex files to check.
        #[arg(long)]
        dir: PathBuf,

        /// How to handle mismatches between directory contents and known fixtures.
        #[arg(long, default_value = "subset")]
        mode: CheckMode,
    },
}

#[derive(Clone, ValueEnum)]
enum CheckMode {
    /// Directory must contain exactly the fixtures we know about.
    Exact,
    /// Directory may have extra files (skip unknown), but all known must be present.
    Subset,
    /// Directory may be missing files (skip missing), but no unknown files allowed.
    Superset,
}

fn main() -> VortexResult<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate { output } => generate::generate(&output),
        Commands::Check { dir, mode } => {
            let mode = match mode {
                CheckMode::Exact => check::Mode::Exact,
                CheckMode::Subset => check::Mode::Subset,
                CheckMode::Superset => check::Mode::Superset,
            };
            check::check(&dir, mode)
        }
    }
}
