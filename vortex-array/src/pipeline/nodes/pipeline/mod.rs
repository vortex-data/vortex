// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod buffers;
mod dag;
mod operators;
mod toposort;

use crate::pipeline::PipelineContext;
use crate::pipeline::bits::BitView;
use crate::pipeline::buffers::BufferId;
use crate::pipeline::nodes::operator::Operator;
use crate::pipeline::nodes::pipeline::buffers::{OutputTarget, VectorAllocationPlan};
use crate::pipeline::nodes::pipeline::dag::DagNode;
use crate::pipeline::nodes::plan::PlanNode;
use crate::pipeline::vector::Vector;
use std::cell::{Ref, RefCell};
use std::task::Poll;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

/// The idea of a pipeline is to orchestrate driving a set of operators to completion with
/// fully optimized resource usage.
///
/// During construction, the plan is analyzed to determine the optimal way to execute the nodes.
/// This includes:
/// - Sub-expression elimination: Identifying common sub-expressions and reusing them.
/// - Vector allocation: Determining how many intermediate vectors are needed.
/// - Buffer management: Managing the buffers that hold the data for each node.
///
pub struct Pipeline<'a> {
    /// Nodes in the DAG representing the execution plan with common sub-expressions eliminated.
    dag: Vec<DagNode<'a>>,
    /// The index into the `dag` of the root node (the entry point for execution).
    dag_root: usize,

    /// The topological order of `dag` nodes for execution.
    execution_order: Vec<usize>,
    /// The leaf nodes of the plan (nodes with no children).
    leaf_nodes: Vec<usize>,
    /// The operators bound to each node in the DAG.
    operators: Vec<Box<dyn Operator>>,
    /// The allocation plan for vectors used by the pipeline.
    allocation_plan: VectorAllocationPlan,

    /// The current state of each node in the DAG, indexed by position in `dag`.
    node_states: Vec<NodeState>,
    next_nodes: Vec<usize>,
}

/// Execution state for a node
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum NodeState {
    /// Node has not been executed yet
    NotStarted,
    /// Node is waiting for children to complete
    // WaitingForChildren(Vec<usize>),
    /// Node is currently executing (may return Poll::Pending)
    Executing,
    /// Node has completed execution
    Completed,
}

impl<'a> Pipeline<'a> {
    // TODO(ngates): can we pass the mask in here such that the plan can replace empty nodes?
    pub fn new(plan: &'a dyn PlanNode) -> VortexResult<Self> {
        // Step 1: Convert the plan tree to a DAG by eliminating common sub-expressions.
        let (dag_root, dag) = Self::build_dag(plan)?;

        // Step 2: Determine execution order (topological sort)
        let execution_order = Self::topological_sort(&dag)?;
        let leaf_nodes = Self::leaf_nodes(&dag);

        // Step 3: Allocate vectors
        let allocation_plan = Self::allocate_vectors(dag_root, &dag, &execution_order)?;
        // let (buffer_slots, buffers) = Self::allocate_buffers(&dag, &execution_order)?;

        // Step 4: Initialize execution state
        // Each step of the pipeline re-initializes the node states.
        let node_states = vec![NodeState::NotStarted; dag.len()];
        let next_nodes = leaf_nodes.clone();
        // But the operators are constructed once.
        let operators = Self::bind_operators(&dag)?;

        Ok(Self {
            dag,
            dag_root,
            execution_order,
            leaf_nodes,
            operators,
            allocation_plan,
            node_states,
            next_nodes,
        })
    }

    /// Step the pipeline forward
    pub fn step(
        &mut self,
        ctx: &dyn PipelineContext,
        out: &mut RefCell<Vector>,
    ) -> Poll<VortexResult<bool>> {
        let mut pending = false;

        // Loop until all the available nodes are pending.
        while let Some(node_idx) = self.next_nodes.pop() {
            self.node_states[node_idx] = NodeState::Executing;
            let operator = self.operators[node_idx].as_mut();
            let node = &self.dag[node_idx];

            // Gather input views from children
            let inputs: Vec<Ref<Vector>> = node
                .children
                .iter()
                .map(|&child_idx| {
                    match &self.allocation_plan.output_targets[child_idx] {
                        OutputTarget::ExternalOutput => {
                            // Child wrote to external output - create view
                            // out.as_view()
                            todo!("")
                        }
                        OutputTarget::IntermediateVector(vector_idx) => {
                            // Child wrote to intermediate vector
                            self.allocation_plan.vectors[*vector_idx].borrow()
                        }
                        OutputTarget::InPlace(_, vector_idx) => {
                            // Child to operate in-place
                            // TODO(ngates): that means we should not pass an input at all?
                            self.allocation_plan.vectors[*vector_idx].borrow()
                        }
                    }
                })
                .collect();

            // Determine output
            let mut output = match &self.allocation_plan.output_targets[node_idx] {
                OutputTarget::ExternalOutput => {
                    // Write directly to external output
                    out.borrow_mut()
                }
                OutputTarget::IntermediateVector(vector_idx) => {
                    // Write to an intermediate vector
                    self.allocation_plan.vectors[*vector_idx].borrow_mut()
                }
                OutputTarget::InPlace(_input_idx, vector_idx) => {
                    // Operate in-place on input
                    // This is tricky - we need mutable access to the input
                    // In practice, this might require unsafe or RefCell
                    self.allocation_plan.vectors[*vector_idx].borrow_mut()
                }
            };

            // Execute with mask (all-true for now)
            let mask = BitView::all_true();

            if let Poll::Ready(()) = operator.execute_dyn(ctx, mask, &inputs, &mut output)? {
                // If the operator completed successfully, we remove this node from the
                // next_nodes list and push its parents.
                self.node_states[node_idx] = NodeState::Completed;
                self.next_nodes.retain(|&n| n != node_idx);
                // Add parents to next_nodes if they are now ready
                for &parent_idx in &node.parents {
                    // Check if all children of the parent are completed
                    let parent = &self.dag[parent_idx];
                    let all_children_done = parent
                        .children
                        .iter()
                        .all(|&child| self.node_states[child] == NodeState::Completed);

                    if all_children_done && self.node_states[parent_idx] == NodeState::NotStarted {
                        self.node_states[parent_idx] = NodeState::Executing;
                        self.next_nodes.push(parent_idx);
                    }
                }
            } else {
                pending = true;
            }
        }

        if pending {
            // If any node is still pending, we return Poll::Pending
            Poll::Pending
        } else {
            // If all nodes are completed, we return Poll::Ready(Ok(()))
            Poll::Ready(Ok(self.next_nodes.is_empty()))
        }
    }
}

struct Context {}

impl PipelineContext for Context {
    fn buffer(&self, _buffer_id: BufferId) -> Poll<VortexResult<ByteBuffer>> {
        todo!()
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_common_sub_expressions() {}
}
