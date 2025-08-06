// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::pipeline::nodes::pipeline::Pipeline;
use crate::pipeline::nodes::plan::PlanNode;
use std::hash::BuildHasher;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::{HashMap, RandomState};

/// A node in our execution DAG
#[derive(Clone)]
pub(super) struct DagNode<'a> {
    /// Index of this node in the DAG
    pub(super) index: usize,
    /// The original plan node
    pub(super) plan_node: &'a dyn PlanNode,
    /// Indices of children in the DAG
    pub(super) children: Vec<usize>,
    /// Indices of parents in the DAG (for dependency tracking)
    pub(super) parents: Vec<usize>,
    /// Hash of this subtree (for deduplication)
    pub(super) subtree_hash: u64,
    /// Output buffer assignment (if not writing to final output)
    pub(super) output_buffer: Option<BufferSlot>,
}

/// A reusable buffer slot
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BufferSlot {
    index: usize,
    size_bytes: usize,
}

impl<'a> Pipeline<'a> {
    /// Build DAG from a tree, eliminating common sub-expressions
    pub(super) fn build_dag(root: &'a dyn PlanNode) -> VortexResult<(usize, Vec<DagNode<'a>>)> {
        let mut dag = Vec::new();
        let mut hash_to_index = HashMap::new();

        // Recursive function to build DAG
        fn visit_node<'b>(
            node: &'b dyn PlanNode,
            dag: &mut Vec<DagNode<'b>>,
            hash_to_index: &mut HashMap<u64, usize>,
        ) -> usize {
            // Compute hash for this subtree
            let subtree_hash = RandomState::default().hash_one(node);

            // Check if we've seen this subtree before (sub-expression elimination)
            if let Some(&existing_index) = hash_to_index.get(&subtree_hash) {
                // Reuse existing node
                return existing_index;
            }

            // Process children first (post-order traversal)
            let child_indices: Vec<usize> = node
                .children()
                .iter()
                .map(|child| visit_node(child.as_ref(), dag, hash_to_index))
                .collect();

            // Create new DAG node
            let index = dag.len();
            let dag_node = DagNode {
                index,
                plan_node: node,
                children: child_indices.clone(),
                parents: Vec::new(), // Will be filled in later
                subtree_hash,
                output_buffer: None, // Will be assigned later
            };

            dag.push(dag_node);
            hash_to_index.insert(subtree_hash, index);

            // Store the plan node (we need to clone or move it somehow)
            // This is tricky with the current design - we might need Arc
            // For now, assume we can store a reference or recreate it

            index
        }

        // Build the DAG
        let root_index = visit_node(root, &mut dag, &mut hash_to_index);

        // Fill in parent relationships
        for i in 0..dag.len() {
            let children = dag[i].children.clone();
            for &child_idx in &children {
                dag[child_idx].parents.push(i);
            }
        }

        Ok((root_index, dag))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dag_construction() {
        // Create a tree with common sub-expressions
        // Example:
        //       root
        //      /    \
        //     A      B
        //    / \    / \
        //   C   D  C   E
        //
        // Should become DAG:
        //       root
        //      /    \
        //     A      B
        //    / \    / \
        //   C   D  /   E
        //    \    /
        //     \  /
        //      \/
        //      (C is shared)

        // let root = create_test_tree();
        // let pipeline = Pipeline::new(root).unwrap();

        // Verify DAG has fewer nodes than tree
        // assert!(pipeline.dag.len() < count_tree_nodes(root));
    }
}
