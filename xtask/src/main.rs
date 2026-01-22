// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod check_deps;
mod generate_fbs;
mod generate_proto;
mod java_test_files;
mod sort_workspace;

use clap::Parser;

use crate::check_deps::check_deps;
use crate::generate_fbs::generate_fbs;
use crate::generate_proto::generate_proto;
use crate::java_test_files::java_test_files;
use crate::sort_workspace::sort_workspace;

#[derive(clap::Parser)]
struct Xtask {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Check for cyclic dependencies in the workspace.
    /// Dev-dependencies are allowed but logged as warnings.
    #[command(name = "check-deps")]
    CheckDeps,
    /// Subcommand to regenerate flatbuffers language bindings for the Rust project.
    #[command(name = "generate-fbs")]
    GenerateFlatbuffers,
    /// Subcommand to regenerate protobuf language bindings for the Rust project.
    #[command(name = "generate-proto")]
    GenerateProto,
    /// Subcommand to generate files for Java integration tests.
    #[command(name = "java-test-files")]
    JavaTestFiles,
    /// Sort workspace members in optimal order for cargo-hack.
    /// Packages are ordered by dependency level, with packages that have
    /// more dependents processed first to maximize Cargo cache reuse.
    #[command(name = "sort-workspace")]
    SortWorkspace {
        /// Check if the workspace is sorted without modifying it.
        #[arg(long)]
        check: bool,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Xtask::parse();
    match cli.command {
        Commands::CheckDeps => check_deps()?,
        Commands::GenerateFlatbuffers => generate_fbs()?,
        Commands::GenerateProto => generate_proto()?,
        Commands::JavaTestFiles => java_test_files()?,
        Commands::SortWorkspace { check } => sort_workspace(check)?,
    }
    Ok(())
}
