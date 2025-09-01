// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::expect_used)]
mod browse;
mod convert;
mod segments;
mod tree;

use std::path::PathBuf;
use std::sync::LazyLock;

use browse::exec_tui;
use clap::{CommandFactory, Parser};
use tokio::runtime::Runtime;
use tree::exec_tree;
use vortex::error::VortexExpect;

static TOKIO_RUNTIME: LazyLock<Runtime> =
    LazyLock::new(|| Runtime::new().expect("Tokio Runtime::new()"));

#[derive(clap::Parser)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Debug, clap::Subcommand)]
enum Commands {
    /// Print the encoding tree of a Vortex file.
    Tree {
        file: PathBuf,
    },
    /// Convert a Parquet file to a Vortex file. Chunking occurs on Parquet RowGroup boundaries.
    Convert(#[command(flatten)] convert::Flags),
    /// Interactively browse the Vortex file.
    Browse {
        file: PathBuf,
    },
    Segments {
        file: PathBuf,
    },
}

impl Commands {
    fn file_path(&self) -> &PathBuf {
        match self {
            Commands::Tree { file } | Commands::Browse { file } | Commands::Segments { file } => {
                file
            }
            Commands::Convert(flags) => &flags.file,
        }
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let cli = Cli::parse();

    let path = cli.command.file_path();
    if !std::fs::exists(path)? {
        Cli::command()
            .error(
                clap::error::ErrorKind::Io,
                format!(
                    "File '{}' does not exist.",
                    path.to_str().vortex_expect("file path")
                ),
            )
            .exit()
    }

    match cli.command {
        Commands::Tree { file } => exec_tree(file)?,
        Commands::Convert(flags) => TOKIO_RUNTIME.block_on(convert::exec_convert(flags))?,
        Commands::Browse { file } => exec_tui(file)?,
        Commands::Segments { file } => TOKIO_RUNTIME.block_on(segments::segments(file))?,
    };

    Ok(())
}
