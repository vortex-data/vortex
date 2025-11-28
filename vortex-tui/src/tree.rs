// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;
use std::path::PathBuf;

use vortex::array::stream::ArrayStreamExt;
use vortex::error::VortexResult;
use vortex::file::OpenOptionsSessionExt;

use crate::SESSION;

#[derive(Debug, clap::Parser)]
pub struct TreeArgs {
    /// What tree to display
    #[clap(subcommand)]
    pub mode: TreeMode,
}

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
        /// Show additional metadata information
        #[arg(short, long)]
        verbose: bool,
    },
}

pub async fn exec_tree(args: TreeArgs) -> VortexResult<()> {
    match args.mode {
        TreeMode::Array { file } => exec_array_tree(&file).await?,
        TreeMode::Layout { file, verbose } => exec_layout_tree(&file, verbose).await?,
    }

    Ok(())
}

async fn exec_array_tree(file: &Path) -> VortexResult<()> {
    let full = SESSION
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

async fn exec_layout_tree(file: &Path, verbose: bool) -> VortexResult<()> {
    let vxf = SESSION.open_options().open(file).await?;
    let footer = vxf.footer();

    println!("{}", footer.layout().display_tree_verbose(verbose));

    Ok(())
}
