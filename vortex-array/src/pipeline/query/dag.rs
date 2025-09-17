// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::BuildHasher;

use crate::operator::{Operator, PipelinedOperator};
use crate::pipeline::query::QueryPlan;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::{HashMap, RandomState};

/// A node in our execution DAG
#[derive(Clone, Debug)]
pub(crate) struct DagNode<'a> {
    /// The original plan node
    pub(crate) plan_node: &'a dyn PipelinedOperator,
    /// Indices of children in the DAG
    pub(crate) children: Vec<usize>,
    /// Indices of parents in the DAG (for dependency tracking)
    pub(crate) parents: Vec<usize>,
}

impl<'a> QueryPlan<'a> {
    /// Build DAG from a tree, eliminating common sub-expressions
    pub(crate) fn build_dag(root: &'a dyn Operator) -> VortexResult<(usize, Vec<DagNode<'a>>)> {
        let mut dag = Vec::new();
        let mut hash_to_index = HashMap::new();

        // Recursive function to build DAG
        fn visit_node<'b>(
            node: &'b dyn PipelinedOperator,
            dag: &mut Vec<DagNode<'b>>,
            hash_to_index: &mut HashMap<u64, usize>,
            random_state: &RandomState,
        ) -> usize {
            // Compute hash for this subtree
            let subtree_hash = random_state.hash_one(node);

            // Check if we've seen this subtree before (sub-expression elimination)
            if let Some(&existing_index) = hash_to_index.get(&subtree_hash) {
                // Reuse existing node
                return existing_index;
            }

            // Process children first (post-order traversal)
            let child_indices: Vec<usize> = node
                .children()
                .iter()
                .map(|child| visit_node(child.as_ref(), dag, hash_to_index, random_state))
                .collect();

            // Create new DAG node
            let index = dag.len();
            let dag_node = DagNode {
                plan_node: node,
                children: child_indices,
                parents: Vec::new(), // Will be filled in later
            };

            dag.push(dag_node);
            hash_to_index.insert(subtree_hash, index);

            // Store the plan node (we need to clone or move it somehow)
            // This is tricky with the current design - we might need Arc
            // For now, assume we can store a reference or recreate it

            index
        }

        // Build the DAG
        let random_state = RandomState::default();
        let root_index = visit_node(root, &mut dag, &mut hash_to_index, &random_state);

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
