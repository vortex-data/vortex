#![allow(clippy::expect_used)]
mod browse;
mod tree;

use std::path::PathBuf;
use std::sync::LazyLock;

use browse::exec_tui;
use clap::Parser;
use tokio::runtime::Runtime;
use tree::exec_tree;

static TOKIO_RUNTIME: LazyLock<Runtime> =
    LazyLock::new(|| Runtime::new().expect("Tokio Runtime::new()"));

#[derive(clap::Parser)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Debug, clap::Subcommand)]
enum Commands {
    Tree { file: PathBuf },
    Browse { file: PathBuf },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Tree { file } => exec_tree(file).expect("exec_tre"),
        Commands::Browse { file } => exec_tui(file).expect("exec_tui"),
    }
}
