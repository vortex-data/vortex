// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::expect_used)]
mod browse;
mod convert;
mod inspect;
mod tree;

use std::path::PathBuf;

use browse::exec_tui;
use clap::{CommandFactory, Parser};
use tree::exec_tree;
use vortex::error::VortexExpect;

use crate::inspect::InspectArgs;

#[derive(clap::Parser)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Debug, clap::Subcommand)]
enum Commands {
    /// Print the encoding tree of a Vortex file.
    Tree { file: PathBuf },
    /// Convert a Parquet file to a Vortex file. Chunking occurs on Parquet RowGroup boundaries.
    Convert(#[command(flatten)] convert::Flags),
    /// Interactively browse the Vortex file.
    Browse { file: PathBuf },
    /// Inspect Vortex file footer and metadata
    Inspect(InspectArgs),
}

impl Commands {
    fn file_path(&self) -> &PathBuf {
        match self {
            Commands::Tree { file } | Commands::Browse { file } => file,
            Commands::Convert(flags) => &flags.file,
            Commands::Inspect(args) => &args.file,
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
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
        Commands::Tree { file } => exec_tree(file).await?,
        Commands::Convert(flags) => convert::exec_convert(flags).await?,
        Commands::Browse { file } => exec_tui(file).await?,
        Commands::Inspect(args) => inspect::exec_inspect(args).await?,
    };

    Ok(())
}
