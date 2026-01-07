// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Print tree views of Vortex files.

use std::path::Path;
use std::path::PathBuf;

use vortex::array::stream::ArrayStreamExt;
use vortex::error::VortexResult;
use vortex::file::OpenOptionsSessionExt;
use vortex::session::VortexSession;

/// Command-line arguments for the tree command.
#[derive(Debug, clap::Parser)]
pub struct TreeArgs {
    /// Which kind of tree to display.
    #[clap(subcommand)]
    pub mode: TreeMode,
}

/// What kind of tree to display.
#[derive(Debug, clap::Subcommand)]
pub enum TreeMode {
    /// Display the array encoding tree (loads and materializes arrays)
    Array {
        /// Path to the Vortex file
        file: PathBuf,
    },
    /// Display the layout tree structure (metadata only, no array loading)
    Layout {
        /// Path to the Vortex file
        file: PathBuf,
        /// Show additional metadata information including buffer sizes (requires fetching segments)
        #[arg(short, long)]
        verbose: bool,
    },
}

/// Print tree views of a Vortex file (layout tree or array tree).
///
/// # Errors
///
/// Returns an error if the file cannot be opened or read.
pub async fn exec_tree(session: &VortexSession, args: TreeArgs) -> VortexResult<()> {
    match args.mode {
        TreeMode::Array { file } => exec_array_tree(session, &file).await?,
        TreeMode::Layout { file, verbose } => exec_layout_tree(session, &file, verbose).await?,
    }

    Ok(())
}

async fn exec_array_tree(session: &VortexSession, file: &Path) -> VortexResult<()> {
    let full = session
        .open_options()
        .open(file)
        .await?
        .scan()?
        .into_array_stream()?
        .read_all()
        .await?;

    println!("{}", full.display_tree());

    Ok(())
}

async fn exec_layout_tree(session: &VortexSession, file: &Path, verbose: bool) -> VortexResult<()> {
    let vxf = session.open_options().open(file).await?;

    if verbose {
        // In verbose mode, fetch segments to display buffer sizes.
        let output = vxf
            .footer()
            .layout()
            .display_tree_with_segments(vxf.segment_source())
            .await?;
        println!("{}", output);
    } else {
        // In non-verbose mode, just display layout tree without fetching segments.
        println!("{}", vxf.footer().layout().display_tree());
    }

    Ok(())
}
