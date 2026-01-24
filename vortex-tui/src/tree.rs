// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Print tree views of Vortex files.

use std::path::Path;
use std::path::PathBuf;

use serde::Deserialize;
use serde::Serialize;
use serde_json_path::JsonPath;
use termtree::Tree;
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
        /// Select specific struct fields (comma-separated). Example: -f "a,b,c"
        #[arg(short, long, value_delimiter = ',')]
        fields: Vec<String>,
        /// Show only encoding names (no dtype, rows, metadata, etc.)
        #[arg(short, long)]
        encoding_only: bool,
        /// Match encoding paths using JSONPath syntax. Examples:
        /// - "$..flat" (all flat nodes)
        /// - "$.struct..dict" (dict under struct)
        /// - "$.*.chunked" (chunked under any root child)
        #[arg(short = 'm', long = "match")]
        match_pattern: Option<String>,
        /// Number of parent levels to include in match results (default: 0)
        #[arg(short = 'c', long, default_value = "0")]
        context: usize,
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
    match args.mode {
        TreeMode::Array { file, json } => exec_array_tree(session, &file, json).await?,
        TreeMode::Layout {
            file,
            verbose,
            json,
            fields,
            encoding_only,
            match_pattern,
            context,
        } => {
            exec_layout_tree(
                session,
                &file,
                LayoutTreeOptions {
                    verbose,
                    json,
                    fields,
                    encoding_only,
                    match_pattern,
                    context,
                },
            )
            .await?
        }
    }

    Ok(())
}

/// Options for layout tree display.
struct LayoutTreeOptions {
    verbose: bool,
    json: bool,
    fields: Vec<String>,
    encoding_only: bool,
    match_pattern: Option<String>,
    context: usize,
}

async fn exec_array_tree(session: &VortexSession, file: &Path, _json: bool) -> VortexResult<()> {
    let full = session
        .open_options()
        .open_path(file)
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
    opts: LayoutTreeOptions,
) -> VortexResult<()> {
    let vxf = session.open_options().open_path(file).await?;
    let footer = vxf.footer();
    let layout = footer.layout().clone();

    // Build encoding tree for matching
    let encoding_tree = build_encoding_tree(&layout, None)?;

    // Apply field filtering if specified
    let encoding_tree = if opts.fields.is_empty() {
        encoding_tree
    } else {
        filter_by_fields(encoding_tree, &opts.fields)
    };

    // Apply JSONPath matching if specified
    let trees_to_display = if let Some(pattern) = &opts.match_pattern {
        match_encoding_pattern(&encoding_tree, pattern, opts.context)?
    } else {
        vec![("$".to_string(), encoding_tree)]
    };

    for (path, tree) in trees_to_display {
        if opts.match_pattern.is_some() {
            println!("# {path}");
        }

        if opts.json {
            let json_output = serde_json::to_string_pretty(&tree)
                .map_err(|e| vortex::error::vortex_err!("Failed to serialize JSON: {e}"))?;
            println!("{json_output}");
        } else if opts.encoding_only {
            println!("{}", display_encoding_only(&tree));
        } else if opts.verbose {
            println!("{}", display_tree_verbose(&tree));
        } else {
            println!("{}", display_tree_normal(&tree));
        }
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

/// Encoding tree node optimized for JSONPath matching.
/// The encoding name is used as the key for child nodes to enable path matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodingTreeNode {
    /// Short encoding name (e.g., "flat", "struct", "chunked")
    #[serde(rename = "_encoding")]
    pub encoding: String,
    /// Full encoding name (e.g., "vortex.flat")
    #[serde(rename = "_encoding_full")]
    pub encoding_full: String,
    /// Data type
    #[serde(rename = "_dtype")]
    pub dtype: String,
    /// Number of rows
    #[serde(rename = "_rows")]
    pub row_count: u64,
    /// Child name in parent (for display)
    #[serde(rename = "_name", skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Children indexed by encoding name for JSONPath matching
    #[serde(flatten)]
    pub children: std::collections::BTreeMap<String, Vec<EncodingTreeNode>>,
}

/// Build an encoding tree from a layout, suitable for JSONPath matching.
fn build_encoding_tree(layout: &LayoutRef, name: Option<String>) -> VortexResult<EncodingTreeNode> {
    let encoding_full = layout.encoding().to_string();
    let encoding = encoding_full
        .strip_prefix("vortex.")
        .unwrap_or(&encoding_full)
        .to_string();

    let children = layout.children()?;
    let child_names: Vec<_> = layout.child_names().collect();

    // Group children by their encoding name
    let mut children_by_encoding: std::collections::BTreeMap<String, Vec<EncodingTreeNode>> =
        std::collections::BTreeMap::new();

    for (child, child_name) in children.into_iter().zip(child_names.into_iter()) {
        let child_node = build_encoding_tree(&child, Some(child_name.to_string()))?;
        let child_encoding = child_node.encoding.clone();
        children_by_encoding
            .entry(child_encoding)
            .or_default()
            .push(child_node);
    }

    Ok(EncodingTreeNode {
        encoding,
        encoding_full,
        dtype: layout.dtype().to_string(),
        row_count: layout.row_count(),
        name,
        children: children_by_encoding,
    })
}

/// Filter encoding tree to only include specified fields.
fn filter_by_fields(mut tree: EncodingTreeNode, fields: &[String]) -> EncodingTreeNode {
    // Only filter if this is a struct
    if tree.encoding != "struct" {
        return tree;
    }

    // Filter children - keep only those whose name matches one of the fields
    for children_vec in tree.children.values_mut() {
        children_vec.retain(|child| {
            child
                .name
                .as_ref()
                .is_some_and(|n| fields.iter().any(|f| f == n))
        });
    }

    // Remove empty encoding groups
    tree.children.retain(|_, v| !v.is_empty());

    tree
}

/// Match encoding patterns using JSONPath syntax.
fn match_encoding_pattern(
    tree: &EncodingTreeNode,
    pattern: &str,
    context: usize,
) -> VortexResult<Vec<(String, EncodingTreeNode)>> {
    // Convert tree to JSON Value for JSONPath querying
    let json_value = serde_json::to_value(tree)
        .map_err(|e| vortex_err!("Failed to serialize encoding tree: {e}"))?;

    // Parse the JSONPath pattern
    let path = JsonPath::parse(pattern)
        .map_err(|e| vortex_err!("Invalid JSONPath pattern '{}': {}", pattern, e))?;

    // Query the tree
    let matches = path.query(&json_value);

    if matches.is_empty() {
        return Ok(vec![]);
    }

    // Convert matches back to EncodingTreeNode
    let mut results = Vec::new();
    for (idx, matched) in matches.iter().enumerate() {
        let matched_tree: EncodingTreeNode = serde_json::from_value(matched.clone())
            .map_err(|e| vortex_err!("Failed to deserialize matched node: {e}"))?;

        // Build path string from the match
        let path_str = format!("match[{}]", idx);

        // If context > 0, we need to find ancestors (this is a simplified version)
        // Full ancestor tracking would require more complex tree walking
        let result_tree = if context > 0 {
            find_with_ancestors(tree, &matched_tree, context)
        } else {
            matched_tree
        };

        results.push((path_str, result_tree));
    }

    Ok(results)
}

/// Find a node in the tree and return it with N ancestors.
fn find_with_ancestors(
    root: &EncodingTreeNode,
    target: &EncodingTreeNode,
    context: usize,
) -> EncodingTreeNode {
    // Build a path to the target and return the subtree starting from N levels up
    fn find_path(
        node: &EncodingTreeNode,
        target: &EncodingTreeNode,
        path: &mut Vec<EncodingTreeNode>,
    ) -> bool {
        // Check if this is the target (by encoding and dtype match)
        if node.encoding == target.encoding
            && node.dtype == target.dtype
            && node.row_count == target.row_count
        {
            return true;
        }

        path.push(node.clone());
        for children in node.children.values() {
            for child in children {
                if find_path(child, target, path) {
                    return true;
                }
            }
        }
        path.pop();
        false
    }

    let mut path = Vec::new();
    if find_path(root, target, &mut path) && !path.is_empty() {
        // Return the node at (path.len() - context) or root
        let start_idx = path.len().saturating_sub(context);
        path.get(start_idx).cloned().unwrap_or_else(|| target.clone())
    } else {
        target.clone()
    }
}

/// Display encoding tree with only encoding names.
fn display_encoding_only(node: &EncodingTreeNode) -> String {
    fn make_tree(node: &EncodingTreeNode) -> Tree<String> {
        let label = node.encoding.clone();

        let mut child_trees = Vec::new();
        for (_, children) in &node.children {
            for child in children {
                let child_tree = make_tree(child);
                let child_label = if let Some(name) = &child.name {
                    format!("{}: {}", name, child_tree.root)
                } else {
                    child_tree.root.clone()
                };
                child_trees.push(Tree::new(child_label).with_leaves(child_tree.leaves));
            }
        }

        Tree::new(label).with_leaves(child_trees)
    }

    make_tree(node).to_string()
}

/// Display encoding tree with normal info (encoding, dtype).
fn display_tree_normal(node: &EncodingTreeNode) -> String {
    fn make_tree(node: &EncodingTreeNode) -> Tree<String> {
        let label = format!("{}, dtype: {}", node.encoding_full, node.dtype);

        let mut child_trees = Vec::new();
        for (_, children) in &node.children {
            for child in children {
                let child_tree = make_tree(child);
                let child_label = if let Some(name) = &child.name {
                    format!("{}: {}", name, child_tree.root)
                } else {
                    child_tree.root.clone()
                };
                child_trees.push(Tree::new(child_label).with_leaves(child_tree.leaves));
            }
        }

        Tree::new(label).with_leaves(child_trees)
    }

    make_tree(node).to_string()
}

/// Display encoding tree with verbose info (encoding, dtype, rows).
fn display_tree_verbose(node: &EncodingTreeNode) -> String {
    fn make_tree(node: &EncodingTreeNode) -> Tree<String> {
        let label = format!(
            "{}, dtype: {}, rows: {}",
            node.encoding_full, node.dtype, node.row_count
        );

        let mut child_trees = Vec::new();
        for (_, children) in &node.children {
            for child in children {
                let child_tree = make_tree(child);
                let child_label = if let Some(name) = &child.name {
                    format!("{}: {}", name, child_tree.root)
                } else {
                    child_tree.root.clone()
                };
                child_trees.push(Tree::new(child_label).with_leaves(child_tree.leaves));
            }
        }

        Tree::new(label).with_leaves(child_trees)
    }

    make_tree(node).to_string()
}
