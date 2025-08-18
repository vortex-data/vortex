// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vector allocation strategy for pipelines

use std::cell::RefCell;

use vortex_error::{VortexExpect, VortexResult};
use vortex_utils::aliases::hash_map::HashMap;

use crate::query::Pipeline;
use crate::query::dag::DagNode;
use crate::types::VType;
use crate::vector::{Vector, VectorId};

#[derive(Debug)]
pub(crate) struct VectorAllocationPlan {
    /// Where each node writes its output
    pub(crate) output_targets: Vec<OutputTarget>,
    /// The actual allocated vectors
    pub(crate) vectors: Vec<RefCell<Vector>>,
}

/// Tracks which vector a node outputs to
#[derive(Debug, Clone)]
pub(crate) enum OutputTarget {
    /// Node writes to the top-level provided output
    ExternalOutput,
    /// Node writes to an allocated intermediate vector
    IntermediateVector(usize), // vector idx
    /// Node mutates its input in-place (input node index, vector idx)
    InPlace(usize, usize),
}

impl OutputTarget {
    pub fn vector_id(&self) -> Option<VectorId> {
        match self {
            OutputTarget::IntermediateVector(idx) => Some(VectorId(*idx)),
            OutputTarget::InPlace(_, idx) => Some(VectorId(*idx)),
            OutputTarget::ExternalOutput => None,
        }
    }
}

/// Represents an allocated vector that can be reused
#[derive(Debug, Clone)]
struct VectorAllocation {
    /// Type of elements in this vector
    element_type: VType,
    /// When this vector becomes available for reuse (execution step)
    available_after: Option<usize>,
}

/// Lifetime information for a node's output
#[derive(Debug, Clone)]
struct NodeLifetime {
    /// When this node will be executed (earliest possible)
    earliest_execution: usize,
    /// When this node's output is last used
    last_use: usize,
    /// Which nodes consume this node's output
    consumers: Vec<usize>,
    /// Whether this node can operate in-place on its input
    can_operate_in_place: bool,
    /// Whether this node's output flows to the final output
    flows_to_output: bool,
}

// ============================================================================
// Improved Pipeline with vector allocation
// ============================================================================

impl<'a> Pipeline<'a> {
    /// Allocate vectors with lifetime analysis and zero-copy optimization
    pub(crate) fn allocate_vectors(
        dag_root: usize,
        dag: &[DagNode<'a>],
        execution_order: &[usize],
    ) -> VortexResult<VectorAllocationPlan> {
        // Step 1: Analyze node lifetimes and data flow
        let lifetimes = Self::analyze_lifetimes(dag, execution_order)?;

        // Step 2: Identify which nodes flow directly to the final output
        let output_flow = Self::trace_output_flow(dag_root, dag)?;

        // Step 3: Determine output targets for each node
        let mut output_targets: Vec<Option<OutputTarget>> = vec![None; dag.len()];
        let mut allocations = Vec::new();

        // Process nodes in reverse execution order (top-down for output propagation)
        for &node_idx in execution_order.iter().rev() {
            let node = &dag[node_idx];
            let plan_node = node.plan_node;
            let lifetime = &lifetimes[&node_idx];

            // Determine output target
            let output_target = if node.parents.is_empty() {
                // Root node - always writes to external output
                OutputTarget::ExternalOutput
            } else if output_flow.contains(&node_idx)
                && Self::can_pass_through_output(dag, node_idx, &output_targets)
            {
                // This node's output flows to the final output and all parents can pass it through
                OutputTarget::ExternalOutput
            } else if lifetime.can_operate_in_place {
                // Check if we can operate in-place on one of our inputs
                if let Some((input_idx, input_alloc)) =
                    Self::find_in_place_candidate(node, &output_targets, &lifetimes)
                {
                    OutputTarget::InPlace(input_idx, input_alloc)
                } else {
                    // Need new allocation
                    let alloc_id = allocations.len();
                    allocations.push(VectorAllocation {
                        element_type: plan_node.vtype(),
                        available_after: Some(lifetime.last_use),
                    });
                    OutputTarget::IntermediateVector(alloc_id)
                }
            } else {
                // Need new allocation
                let alloc_id = allocations.len();
                allocations.push(VectorAllocation {
                    element_type: plan_node.vtype(),
                    available_after: Some(lifetime.last_use),
                });
                OutputTarget::IntermediateVector(alloc_id)
            };

            output_targets[node_idx] = Some(output_target);
        }

        // Step 4: Optimize allocations with graph coloring
        // let optimized_allocations = Self::optimize_allocations(allocations, &lifetimes)?;

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

    /// Analyze the lifetimes of node outputs
    fn analyze_lifetimes(
        dag: &[DagNode],
        execution_order: &[usize],
    ) -> VortexResult<HashMap<usize, NodeLifetime>> {
        let mut lifetimes = HashMap::new();

        // Build execution position map
        let exec_pos: HashMap<usize, usize> = execution_order
            .iter()
            .enumerate()
            .map(|(pos, &idx)| (idx, pos))
            .collect();

        for (node_idx, node) in dag.iter().enumerate() {
            let earliest_execution = exec_pos[&node_idx];

            // Find when output is last used
            let last_use = if node.parents.is_empty() {
                // Root node - used at the very end
                execution_order.len()
            } else {
                // Last parent to execute
                node.parents
                    .iter()
                    .map(|&parent| exec_pos[&parent])
                    .max()
                    .unwrap_or(earliest_execution)
            };

            // Check if node can operate in-place
            // This would need to come from the plan node metadata
            let can_operate_in_place = false; // TODO: get from plan node

            // Check if flows to output
            let flows_to_output = node.parents.is_empty()
                || node.parents.iter().any(|&p| {
                    // Recursive check would go here
                    dag[p].parents.is_empty()
                });

            lifetimes.insert(
                node_idx,
                NodeLifetime {
                    earliest_execution,
                    last_use,
                    consumers: node.parents.clone(),
                    can_operate_in_place,
                    flows_to_output,
                },
            );
        }

        Ok(lifetimes)
    }

    /// Trace which nodes' outputs flow to the final output
    ///
    /// NOTE: we don't check for cycles here, assuming the DAG is acyclic.
    fn trace_output_flow(dag_root: usize, dag: &[DagNode]) -> VortexResult<Vec<usize>> {
        let mut flows_to_output = Vec::new();
        let mut current = dag_root;

        loop {
            let node = &dag[current];

            // Check if first child has matching type
            if let Some(&first_child_idx) = node.children.first() {
                let first_child = &dag[first_child_idx];

                if node.plan_node.vtype() == first_child.plan_node.vtype() {
                    // This node can pass through the output buffer
                    flows_to_output.push(current);
                    // Continue down the first child
                    current = first_child_idx;
                } else {
                    // Types don't match, can't pass through
                    break;
                }
            } else {
                // No children, we're done
                break;
            }
        }

        Ok(flows_to_output)
    }

    /// Check if we can pass the external output through this node
    fn can_pass_through_output(
        dag: &[DagNode],
        node_idx: usize,
        output_targets: &[Option<OutputTarget>],
    ) -> bool {
        let node = &dag[node_idx];

        // There must not be multiple parents, and it must:
        // 1. Already use external output, OR
        // 2. Be able to pass through its input
        // AND
        // 1. The input type must match the output type
        if node.parents.len() > 1 {
            return false; // Cannot pass through if multiple parents
        }
        node.parents.iter().all(|&parent| {
            if node.plan_node.vtype() != dag[parent].plan_node.vtype() {
                return false; // Type mismatch
            }
            match output_targets[parent] {
                Some(OutputTarget::ExternalOutput) => true,
                Some(OutputTarget::InPlace(..)) => true, // Can pass through
                _ => false,
            }
        })
    }

    /// Find a suitable input for in-place operation
    fn find_in_place_candidate(
        node: &DagNode,
        output_targets: &[Option<OutputTarget>],
        lifetimes: &HashMap<usize, NodeLifetime>,
    ) -> Option<(usize, usize)> {
        // Check each child
        for (input_idx, &child_node_idx) in node.children.iter().enumerate() {
            if let Some(target) = &output_targets[child_node_idx]
                && let OutputTarget::IntermediateVector(alloc_id) = target
            {
                // Check if this child's output is only used by us
                let child_lifetime = &lifetimes[&child_node_idx];
                if child_lifetime.consumers.len() == 1 && child_lifetime.consumers[0] == node.index
                {
                    // We're the only consumer - can reuse in-place
                    return Some((input_idx, *alloc_id));
                }
            }
        }
        None
    }

    // Optimize allocations using graph coloring
    // fn optimize_allocations(
    //     allocations: Vec<VectorAllocation>,
    //     lifetimes: &HashMap<usize, NodeLifetime>,
    // ) -> VortexResult<Vec<Vector>> {
    //     // Group allocations by type and size
    //     let mut allocation_groups: HashMap<(VType, usize), Vec<VectorAllocation>> = HashMap::new();
    //
    //     for alloc in allocations {
    //         let key = (alloc.element_type, alloc.size_bytes);
    //         allocation_groups.entry(key).or_default().push(alloc);
    //     }
    //
    //     // For each group, find minimum number of actual vectors needed
    //     let mut vectors = Vec::new();
    //
    //     for ((vtype, size), allocs) in allocation_groups {
    //         // Sort by availability time
    //         let mut sorted_allocs = allocs;
    //         sorted_allocs.sort_by_key(|a| a.available_after);
    //
    //         // Use interval scheduling to find minimum vectors
    //         let mut reuse_map = HashMap::new();
    //         let mut available_vectors: Vec<(usize, usize)> = Vec::new(); // (vector_id, available_after)
    //
    //         for alloc in sorted_allocs {
    //             // Find a vector that's available
    //             let vector_id = if let Some(pos) = available_vectors
    //                 .iter()
    //                 .position(|(_, avail)| *avail <= alloc.id)
    //             {
    //                 let (vid, _) = available_vectors.remove(pos);
    //                 vid
    //             } else {
    //                 // Need new vector
    //                 let vid = vectors.len();
    //                 vectors.push(Vector::new(vtype, 1024)?);
    //                 vid
    //             };
    //
    //             reuse_map.insert(alloc.id, vector_id);
    //
    //             if let Some(available_after) = alloc.available_after {
    //                 available_vectors.push((vector_id, available_after));
    //             }
    //         }
    //     }
    //
    //     Ok(vectors)
    // }
}
