// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod check_editions;
mod generate_editions;

use clap::Parser;

use crate::check_editions::check_editions;
use crate::generate_editions::generate_editions;

#[derive(clap::Parser)]
struct Xtask {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Subcommand to check that frozen edition records never change.
    #[command(name = "check-editions")]
    CheckEditions {
        /// The revision to compare against.
        #[arg(long, default_value = "origin/develop")]
        base: String,
    },
    /// Subcommand to regenerate the edition records under `vortex/editions`.
    #[command(name = "generate-editions")]
    Editions,
}

fn main() -> anyhow::Result<()> {
    let cli = Xtask::parse();
    match cli.command {
        Commands::CheckEditions { base } => check_editions(&base)?,
        Commands::Editions => generate_editions()?,
    }
    Ok(())
}
