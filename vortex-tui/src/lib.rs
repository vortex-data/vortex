// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex TUI library for interactively browsing and inspecting Vortex files.
//!
//! This crate provides both a CLI tool (`vx`) and a library API for working with Vortex files.
//! Users can bring their own `VortexSession` to enable custom encodings and extensions.
//!
//! # Example
//!
//! ```ignore
//! use vortex::session::VortexSession;
//! use vortex::io::session::RuntimeSessionExt;
//! use vortex_tui::browse;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let session = VortexSession::default().with_tokio();
//!     browse::exec_tui(&session, "my_file.vortex").await?;
//!     Ok(())
//! }
//! ```

#![deny(clippy::missing_errors_doc)]
#![deny(clippy::missing_panics_doc)]
#![deny(clippy::missing_safety_doc)]
#![deny(missing_docs)]
#![allow(clippy::needless_pass_by_value)]

#[cfg_attr(
    all(not(feature = "native"), not(target_arch = "wasm32")),
    allow(dead_code, unused_imports)
)]
pub mod browse;
pub mod segment_tree;

#[cfg(feature = "native")]
pub mod convert;
#[cfg(feature = "native")]
pub mod datafusion_helper;
#[cfg(feature = "native")]
pub mod inspect;
#[cfg(feature = "native")]
pub mod query;
#[cfg(feature = "native")]
pub mod segments;
#[cfg(feature = "native")]
pub mod tree;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

#[cfg(feature = "native")]
mod native_cli {
    use std::ffi::OsString;
    use std::path::PathBuf;

    use clap::CommandFactory;
    use clap::Parser;
    use vortex::error::VortexExpect;
    use vortex::session::VortexSession;

    #[derive(clap::Parser)]
    #[command(version)]
    struct Cli {
        #[clap(subcommand)]
        command: Commands,
    }

    #[derive(Debug, clap::Subcommand)]
    enum Commands {
        /// Print tree views of a Vortex file (layout tree or array tree)
        Tree(super::tree::TreeArgs),
        /// Convert a Parquet file to a Vortex file. Chunking occurs on Parquet RowGroup boundaries.
        Convert(#[command(flatten)] super::convert::ConvertArgs),
        /// Interactively browse the Vortex file.
        Browse { file: PathBuf },
        /// Inspect Vortex file footer and metadata
        Inspect(super::inspect::InspectArgs),
        /// Execute a SQL query against a Vortex file using DataFusion
        Query(super::query::QueryArgs),
        /// Display segment information for a Vortex file
        Segments(super::segments::SegmentsArgs),
    }

    impl Commands {
        fn file_path(&self) -> &PathBuf {
            match self {
                Commands::Tree(args) => match &args.mode {
                    super::tree::TreeMode::Array { file, .. } => file,
                    super::tree::TreeMode::Layout { file, .. } => file,
                },
                Commands::Browse { file } => file,
                Commands::Convert(flags) => &flags.file,
                Commands::Inspect(args) => &args.file,
                Commands::Query(args) => &args.file,
                Commands::Segments(args) => &args.file,
            }
        }
    }

    /// Main entrypoint for `vx` that launches a [`VortexSession`].
    ///
    /// Parses arguments from [`std::env::args_os`]. See [`launch_from`] to supply explicit arguments.
    ///
    /// # Errors
    ///
    /// Raises any errors from subcommands.
    pub async fn launch(session: &VortexSession) -> anyhow::Result<()> {
        launch_from(session, std::env::args_os()).await
    }

    /// Launch `vx` with explicit command-line arguments.
    ///
    /// This is useful when embedding the TUI inside another process (e.g. Python) where
    /// [`std::env::args`] may not reflect the intended arguments.
    ///
    /// # Errors
    ///
    /// Raises any errors from subcommands.
    pub async fn launch_from(
        session: &VortexSession,
        args: impl IntoIterator<Item = impl Into<OsString> + Clone>,
    ) -> anyhow::Result<()> {
        let _ = env_logger::try_init();

        let cli = Cli::parse_from(args);

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
            Commands::Tree(args) => super::tree::exec_tree(session, args).await?,
            Commands::Convert(flags) => super::convert::exec_convert(session, flags).await?,
            Commands::Browse { file } => super::browse::exec_tui(session, file).await?,
            Commands::Inspect(args) => super::inspect::exec_inspect(session, args).await?,
            Commands::Query(args) => super::query::exec_query(session, args).await?,
            Commands::Segments(args) => super::segments::exec_segments(session, args).await?,
        };

        Ok(())
    }
}

#[cfg(feature = "native")]
pub use native_cli::launch;
#[cfg(feature = "native")]
pub use native_cli::launch_from;
