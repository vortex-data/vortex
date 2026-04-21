// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
#![expect(clippy::expect_used)]

mod data_prep;
mod inspect;
mod query_miner;
mod query_runner;

use clap::Parser;
use clap::Subcommand;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "string-filter-bench")]
#[command(about = "Benchmark toolkit: raw strings vs FSST-compressed string filtering")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Download datasets and write Vortex files with raw + FSST columns
    Prep(data_prep::PrepArgs),
    /// Mine queries from a prepared dataset
    Mine(query_miner::MineArgs),
    /// Run timed benchmarks comparing raw vs FSST filtering
    Run(query_runner::RunArgs),
    /// Inspect FSST symbol table and test memmem on compressed codes
    Inspect(inspect::InspectArgs),
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Prep(args) => data_prep::run(args).await,
        Command::Mine(args) => query_miner::run(args),
        Command::Run(args) => query_runner::run(args),
        Command::Inspect(args) => inspect::run(args),
    }
}
