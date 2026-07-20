// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod generate_editions;
mod generate_fbs;
mod generate_proto;

use clap::Parser;

use crate::generate_editions::generate_editions;
use crate::generate_fbs::generate_fbs;
use crate::generate_proto::generate_proto;

#[derive(clap::Parser)]
struct Xtask {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
#[expect(
    clippy::enum_variant_names,
    reason = "subcommands are all generators, named after their CLI commands"
)]
enum Commands {
    /// Subcommand to regenerate flatbuffers language bindings for the Rust project.
    #[command(name = "generate-fbs")]
    GenerateFlatbuffers,
    /// Subcommand to regenerate protobuf language bindings for the Rust project.
    #[command(name = "generate-proto")]
    GenerateProto,
    /// Subcommand to regenerate edition JSON definitions and documentation pages.
    #[command(name = "generate-editions")]
    GenerateEditions,
}

fn main() -> anyhow::Result<()> {
    let cli = Xtask::parse();
    match cli.command {
        Commands::GenerateFlatbuffers => generate_fbs()?,
        Commands::GenerateProto => generate_proto()?,
        Commands::GenerateEditions => generate_editions()?,
    }
    Ok(())
}
