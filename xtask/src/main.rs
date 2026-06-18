// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod generate_fbs;
mod generate_proto;

use clap::Parser;

use crate::generate_fbs::generate_fbs;
use crate::generate_proto::generate_proto;

#[derive(clap::Parser)]
struct Xtask {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Subcommand to regenerate flatbuffers language bindings for the Rust project.
    #[command(name = "generate-fbs")]
    GenerateFlatbuffers,
    /// Subcommand to regenerate protobuf language bindings for the Rust project.
    #[command(name = "generate-proto")]
    GenerateProto,
}

fn main() -> anyhow::Result<()> {
    let cli = Xtask::parse();
    match cli.command {
        Commands::GenerateFlatbuffers => generate_fbs()?,
        Commands::GenerateProto => generate_proto()?,
    }
    Ok(())
}
