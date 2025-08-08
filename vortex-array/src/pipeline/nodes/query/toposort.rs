// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::pipeline::nodes::query::Pipeline;
use crate::pipeline::nodes::query::dag::DagNode;
use std::collections::VecDeque;
use vortex_error::{VortexResult, vortex_bail};

impl Pipeline<'_> {
    /// Returns the nodes of the DAG with no children.
    pub(super) fn leaf_nodes(dag: &[DagNode]) -> Vec<usize> {
        dag.iter()
            .enumerate()
            .filter(|(_, node)| node.children.is_empty())
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Topological sort for execution order
    pub(super) fn topological_sort(dag: &[DagNode]) -> VortexResult<Vec<usize>> {
        let mut in_degree = vec![0; dag.len()];
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        // Calculate in-degrees
        for node in dag {
            for &child in &node.children {
                in_degree[child] += 1;
            }
        }

        // Find nodes with no dependencies
        for (idx, &degree) in in_degree.iter().enumerate() {
            if degree == 0 {
                queue.push_back(idx);
            }
        }

        // Process nodes in topological order
        while let Some(idx) = queue.pop_front() {
            result.push(idx);

            for &child in &dag[idx].children {
                in_degree[child] -= 1;
                if in_degree[child] == 0 {
                    queue.push_back(child);
                }
            }
        }

        if result.len() != dag.len() {
            vortex_bail!(
                "Cycle detected in DAG: expected {} nodes, found {}",
                dag.len(),
                result.len()
            );
        }

        // Reverse to get a bottom-up execution order
        result.reverse();
        Ok(result)
    }
}
