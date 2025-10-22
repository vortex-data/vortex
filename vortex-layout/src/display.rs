// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use termtree::Tree;
use vortex_error::VortexResult;

use crate::LayoutRef;

/// Display wrapper for layout tree visualization
pub struct DisplayLayoutTree(pub LayoutRef);

impl std::fmt::Display for DisplayLayoutTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn make_tree(layout: LayoutRef) -> VortexResult<Tree<String>> {
            // Build the node label with encoding, dtype, and metadata
            let mut node_parts = vec![
                format!("{}", layout.encoding()),
                format!("dtype: {}", layout.dtype()),
            ];

            // Add child count if there are children
            let nchildren = layout.nchildren();
            if nchildren > 0 {
                node_parts.push(format!("children: {}", nchildren));
            }

            let node_name = node_parts.join(", ");

            // Get children and child names directly from the layout (not loading arrays)
            let children = layout.children()?;
            let child_names: Vec<_> = layout.child_names().collect();

            // Build child trees
            let child_trees: VortexResult<Vec<Tree<String>>> =
                if !children.is_empty() && child_names.len() == children.len() {
                    // If we have names for all children, use them
                    children
                        .into_iter()
                        .zip(child_names.iter())
                        .map(|(child, name)| {
                            let child_tree = make_tree(child)?;
                            Ok(Tree::new(format!("{}: {}", name, child_tree.root))
                                .with_leaves(child_tree.leaves))
                        })
                        .collect()
                } else if !children.is_empty() {
                    // No names available, just show children
                    children.into_iter().map(make_tree).collect()
                } else {
                    // Leaf node - no children
                    Ok(Vec::new())
                };

            Ok(Tree::new(node_name).with_leaves(child_trees?))
        }

        match make_tree(self.0.clone()) {
            Ok(tree) => write!(f, "{}", tree),
            Err(e) => write!(f, "Error building layout tree: {}", e),
        }
    }
}
