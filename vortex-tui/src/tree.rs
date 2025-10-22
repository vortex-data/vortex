// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::{Path, PathBuf};

use vortex::error::VortexResult;
use vortex::file::VortexOpenOptions;
use vortex::stream::ArrayStreamExt;

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
    },
}

pub async fn exec_tree(args: TreeArgs) -> VortexResult<()> {
    match args.mode {
        TreeMode::Array { file } => exec_array_tree(&file).await?,
        TreeMode::Layout { file } => exec_layout_tree(&file).await?,
    }

    Ok(())
}

async fn exec_array_tree(file: &Path) -> VortexResult<()> {
    let full = VortexOpenOptions::new()
        .open(file)
        .await?
        .scan()?
        .into_array_stream()?
        .read_all()
        .await?;

    println!("{}", full.display_tree());

    Ok(())
}

async fn exec_layout_tree(file: &Path) -> VortexResult<()> {
    let vxf = VortexOpenOptions::new().open(file).await?;
    let footer = vxf.footer();

    println!("{}", footer.layout().display_tree());

    Ok(())
}
