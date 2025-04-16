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

use crate::convert::{Flags, exec_convert};

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
    Convert {
        /// Path to the Parquet file on disk to convert to Vortex
        file: PathBuf,

        /// Execute quietly. No output will be printed.
        #[arg(short, long)]
        quiet: bool,
    },
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
            Commands::Tree { file }
            | Commands::Convert { file, .. }
            | Commands::Browse { file }
            | Commands::Segments { file } => file,
        }
    }
}

fn main() -> anyhow::Result<()> {
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
        Commands::Tree { file } => TOKIO_RUNTIME.block_on(exec_tree(file))?,
        Commands::Convert { file, quiet } => {
            TOKIO_RUNTIME.block_on(exec_convert(file, Flags { quiet }))?
        }
        Commands::Browse { file } => exec_tui(file)?,
        Commands::Segments { file } => TOKIO_RUNTIME.block_on(segments::segments(file))?,
    };

    Ok(())
}
