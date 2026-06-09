// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Print tree views of Vortex files.

use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use serde::Serialize;
use vortex::array::stream::ArrayStreamExt;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::file::OpenOptionsSessionExt;
use vortex::layout::LayoutRef;
use vortex::session::VortexSession;

/// Command-line arguments for the tree command.
#[derive(Debug, clap::Parser)]
pub struct TreeArgs {
    /// Which kind of tree to display.
    #[clap(subcommand)]
    pub mode: TreeMode,

    /// Object store options for remote files (see `--store-option`).
    #[command(flatten)]
    pub store: crate::store_options::StoreOptions,
}

/// What kind of tree to display.
#[derive(Debug, clap::Subcommand)]
pub enum TreeMode {
    /// Display the array encoding tree (loads and materializes arrays)
    Array {
        /// Path to the Vortex file
        file: PathBuf,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
    /// Display the layout tree structure (metadata only, no array loading)
    Layout {
        /// Path to the Vortex file
        file: PathBuf,
        /// Show additional metadata information including buffer sizes (requires fetching segments)
        #[arg(short, long)]
        verbose: bool,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
}

/// Layout tree node for JSON output.
#[derive(Serialize)]
pub struct LayoutTreeNode {
    /// Encoding name.
    pub encoding: String,
    /// Data type.
    pub dtype: String,
    /// Number of rows.
    pub row_count: u64,
    /// Metadata size in bytes.
    pub metadata_bytes: usize,
    /// Segment IDs referenced by this layout.
    pub segment_ids: Vec<u32>,
    /// Child layouts.
    pub children: Vec<LayoutTreeNodeWithName>,
}

/// Layout tree node with name for JSON output.
#[derive(Serialize)]
pub struct LayoutTreeNodeWithName {
    /// Child name.
    pub name: String,
    /// Child node data.
    #[serde(flatten)]
    pub node: LayoutTreeNode,
}

/// Print tree views of a Vortex file (layout tree or array tree).
///
/// # Errors
///
/// Returns an error if the file cannot be opened or read.
pub async fn exec_tree(session: &VortexSession, args: TreeArgs) -> VortexResult<()> {
    let store = args.store.props();
    match args.mode {
        TreeMode::Array { file, json } => exec_array_tree(session, &file, json, store).await?,
        TreeMode::Layout {
            file,
            verbose,
            json,
        } => exec_layout_tree(session, &file, verbose, json, store).await?,
    }

    Ok(())
}

async fn exec_array_tree(
    session: &VortexSession,
    file: &Path,
    _json: bool,
    store: Vec<(String, String)>,
) -> VortexResult<()> {
    let url = file
        .to_str()
        .ok_or_else(|| vortex_err!("path is not valid UTF-8: {}", file.display()))?;
    let full = session
        .open_options()
        .open_url_with_props(url, store)
        .await?
        .scan()?
        .into_array_stream()?
        .read_all()
        .await?;

    println!("{}", full.display_tree());

    Ok(())
}

async fn exec_layout_tree(
    session: &VortexSession,
    file: &Path,
    verbose: bool,
    json: bool,
    store: Vec<(String, String)>,
) -> VortexResult<()> {
    let url = file
        .to_str()
        .ok_or_else(|| vortex_err!("path is not valid UTF-8: {}", file.display()))?;
    let vxf = session
        .open_options()
        .open_url_with_props(url, store)
        .await?;
    let footer = vxf.footer();

    if json {
        let tree = layout_to_json(Arc::clone(footer.layout()))?;
        let json_output = serde_json::to_string_pretty(&tree)
            .map_err(|e| vortex::error::vortex_err!("Failed to serialize JSON: {e}"))?;
        println!("{json_output}");
    } else if verbose {
        // In verbose mode, fetch segments to display buffer sizes.
        let output = footer
            .layout()
            .display_tree_with_segments(vxf.segment_source())
            .await?;
        println!("{output}");
    } else {
        // In non-verbose mode, just display layout tree without fetching segments.
        println!("{}", footer.layout().display_tree());
    }

    Ok(())
}

fn layout_to_json(layout: LayoutRef) -> VortexResult<LayoutTreeNode> {
    let children = layout.children()?;
    let child_names: Vec<_> = layout.child_names().collect();

    let children_json: Vec<LayoutTreeNodeWithName> = children
        .into_iter()
        .zip(child_names.into_iter())
        .map(|(child, name)| {
            let node = layout_to_json(child)?;
            Ok(LayoutTreeNodeWithName {
                name: name.to_string(),
                node,
            })
        })
        .collect::<VortexResult<Vec<_>>>()?;

    Ok(LayoutTreeNode {
        encoding: layout.encoding().to_string(),
        dtype: layout.dtype().to_string(),
        row_count: layout.row_count(),
        metadata_bytes: layout.metadata().len(),
        segment_ids: layout.segment_ids().iter().map(|s| **s).collect(),
        children: children_json,
    })
}
