// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vector allocation strategy for pipelines

use vortex_error::{VortexExpect, VortexResult};
use vortex_vector::VectorMut;

use crate::Array;
use crate::pipeline::driver::{Node, NodeId};
use crate::pipeline::{N, VectorId};

#[derive(Debug)]
pub struct VectorAllocation {
    /// Where each node writes its output
    pub(crate) output_targets: Vec<OutputTarget>,
    /// The actual allocated vectors
    pub(crate) vectors: Vec<VectorMut>,
}

// TODO(joe): support in-place view operations
// Node mutates its input in-place (input node index, vector idx)
// add variant InPlace.
/// Tracks which vector a node outputs to
#[derive(Debug, Clone)]
pub(crate) enum OutputTarget {
    /// Node writes to the top-level provided output
    ExternalOutput,
    /// Node writes to an allocated intermediate vector
    IntermediateVector(VectorId), // vector idx
}

impl OutputTarget {
    pub fn vector_id(&self) -> Option<VectorId> {
        match self {
            OutputTarget::IntermediateVector(vector_id) => Some(*vector_id),
            OutputTarget::ExternalOutput => None,
        }
    }
}

// ============================================================================
// Improved Pipeline with vector allocation
// ============================================================================

/// Allocate vectors with lifetime analysis and zero-copy optimization
pub(super) fn allocate_vectors(
    dag: &[Node],
    execution_order: &[NodeId],
) -> VortexResult<VectorAllocation> {
    let mut output_targets: Vec<Option<OutputTarget>> = vec![None; dag.len()];
    let mut allocation_types = Vec::new();

    // Process nodes in reverse execution order (top-down for output propagation)
    for &node_idx in execution_order.iter().rev() {
        let node = &dag[node_idx];
        let array = &node.array;

        // Determine output target
        let output_target = if node.parents.is_empty() {
            // Root node - always writes to external output
            OutputTarget::ExternalOutput
        } else {
            // All intermediate nodes need intermediate vector allocation
            // The previous pass-through optimization was buggy and incorrectly
            // assigned ExternalOutput to intermediate nodes

            // TODO(joe): Implement vector allocation reuse optimization here:
            // 1. Identify when intermediate nodes can safely write to ExternalOutput
            // 2. Check that ALL consumers of this node can handle external output
            // 3. Verify no conflicts with parallel execution paths
            // 4. Ensure proper vector lifetime management

            let vector_id = VectorId::new(allocation_types.len());
            allocation_types.push(array.dtype());
            OutputTarget::IntermediateVector(vector_id)
        };

        output_targets[node_idx] = Some(output_target);
    }

    Ok(VectorAllocation {
        output_targets: output_targets
            .into_iter()
            .map(|target| target.vortex_expect("missing target"))
            .collect(),
        vectors: allocation_types
            .into_iter()
            .map(|dtype| VectorMut::with_capacity(dtype, N))
            .collect(),
    })
}
