// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vector allocation strategy for pipelines

use std::cell::RefCell;

use vortex_error::{VortexExpect, VortexResult};

use crate::pipeline::operator::{NodeId, PipelineNode};
use crate::pipeline::vec::Vector;
use crate::pipeline::{VType, VectorId};

#[derive(Debug)]
pub struct VectorAllocationPlan {
    /// Where each node writes its output
    pub(crate) output_targets: Vec<OutputTarget>,
    /// The actual allocated vectors
    pub(crate) vectors: Vec<RefCell<Vector>>,
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
    IntermediateVector(usize), // vector idx
}

impl OutputTarget {
    pub fn vector_id(&self) -> Option<VectorId> {
        match self {
            OutputTarget::IntermediateVector(idx) => Some(*idx),
            OutputTarget::ExternalOutput => None,
        }
    }
}

/// Represents an allocated vector that can be reused
#[derive(Debug, Clone)]
struct VectorAllocation {
    /// Type of elements in this vector
    element_type: VType,
}

// ============================================================================
// Improved Pipeline with vector allocation
// ============================================================================

/// Allocate vectors with lifetime analysis and zero-copy optimization
pub(super) fn allocate_vectors(
    dag: &[PipelineNode],
    execution_order: &[NodeId],
) -> VortexResult<VectorAllocationPlan> {
    let mut output_targets: Vec<Option<OutputTarget>> = vec![None; dag.len()];
    let mut allocations = Vec::new();

    // Process nodes in reverse execution order (top-down for output propagation)
    for &node_idx in execution_order.iter().rev() {
        let node = &dag[node_idx];
        let operator = &node.operator;

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

            let alloc_id = allocations.len();
            allocations.push(VectorAllocation {
                element_type: operator.dtype().into(),
            });
            OutputTarget::IntermediateVector(alloc_id)
        };

        output_targets[node_idx] = Some(output_target);
    }

    Ok(VectorAllocationPlan {
        output_targets: output_targets
            .into_iter()
            .map(|target| target.vortex_expect("missing target"))
            .collect(),
        vectors: allocations
            .into_iter()
            .map(|alloc| RefCell::new(Vector::new_with_vtype(alloc.element_type)))
            .collect(),
    })
}
