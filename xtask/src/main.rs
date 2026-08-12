// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod check_editions;
mod generate_editions;
mod generate_fbs;
mod generate_proto;

use clap::Parser;

use crate::check_editions::check_editions;
use crate::generate_editions::generate_editions;
use crate::generate_fbs::generate_fbs;
use crate::generate_proto::generate_proto;

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
    /// Subcommand to regenerate flatbuffers language bindings for the Rust project.
    #[command(name = "generate-fbs")]
    Flatbuffers,
    /// Subcommand to regenerate protobuf language bindings for the Rust project.
    #[command(name = "generate-proto")]
    Proto,
}

fn main() -> anyhow::Result<()> {
    let cli = Xtask::parse();
    match cli.command {
        Commands::CheckEditions { base } => check_editions(&base)?,
        Commands::Editions => generate_editions()?,
        Commands::Flatbuffers => generate_fbs()?,
        Commands::Proto => generate_proto()?,
    }
    Ok(())
}
