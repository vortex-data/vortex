// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod check_editions;
#[cfg(feature = "editions")]
mod generate_editions;
mod generate_flatbuffers;
mod generate_proto;
mod workspace;

use clap::Parser;

use crate::check_editions::check_editions;
#[cfg(feature = "editions")]
use crate::generate_editions::generate_editions;
use crate::generate_flatbuffers::generate_flatbuffers;
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
    #[cfg(feature = "editions")]
    #[command(name = "generate-editions")]
    Editions,
    /// Subcommand to regenerate the checked-in FlatBuffers bindings with the pinned `flatc`.
    #[command(name = "generate-flatbuffers")]
    FlatBuffers,
    /// Subcommand to regenerate the checked-in Protocol Buffers bindings.
    #[command(name = "generate-proto")]
    Proto,
}

fn main() -> anyhow::Result<()> {
    let cli = Xtask::parse();
    match cli.command {
        Commands::CheckEditions { base } => check_editions(&base)?,
        #[cfg(feature = "editions")]
        Commands::Editions => generate_editions()?,
        Commands::FlatBuffers => generate_flatbuffers()?,
        Commands::Proto => generate_proto()?,
    }
    Ok(())
}
