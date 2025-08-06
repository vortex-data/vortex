// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod buffers;
mod dag;
mod operators;
mod toposort;

use crate::pipeline::bits::BitView;
use crate::pipeline::buffers::BufferId;
use crate::pipeline::nodes::operator::Operator;
use crate::pipeline::nodes::pipeline::buffers::{OutputTarget, VectorAllocationPlan};
use crate::pipeline::nodes::pipeline::dag::DagNode;
use crate::pipeline::nodes::plan::PlanNode;
use crate::pipeline::vector::Vector;
use crate::pipeline::view::View;
use crate::pipeline::{PipelineContext, VectorId, VectorRef};
use std::cell::{Ref, RefCell};
use std::ops::{Deref, DerefMut};
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
        let operators = Self::bind_operators(&dag, &allocation_plan)?;

        println!("Allocation Plan: {:?}", allocation_plan);
        assert!(!next_nodes.is_empty(), "No nodes to execute");

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
    pub fn step(&mut self, out: &mut RefCell<Vector>) -> Poll<VortexResult<bool>> {
        let mut pending = false;

        // Loop until all the available nodes are pending.
        while let Some(node_idx) = self.next_nodes.pop() {
            println!("Executing node: {}", node_idx);
            self.node_states[node_idx] = NodeState::Executing;
            let operator = self.operators[node_idx].as_mut();
            let node = &self.dag[node_idx];
            //
            // // Gather input views from children
            // let inputs: Vec<Ref<Vector>> = node
            //     .children
            //     .iter()
            //     .map(|&child_idx| {
            //         match &self.allocation_plan.output_targets[child_idx] {
            //             OutputTarget::ExternalOutput => {
            //                 unreachable!("Child node cannot write to external output directly")
            //             }
            //             OutputTarget::IntermediateVector(vector_idx) => {
            //                 // Child wrote to intermediate vector
            //                 self.allocation_plan.vectors[*vector_idx].borrow()
            //             }
            //             OutputTarget::InPlace(_, vector_idx) => {
            //                 // Child to operate in-place
            //                 // TODO(ngates): that means we should not pass an input at all?
            //                 self.allocation_plan.vectors[*vector_idx].borrow()
            //             }
            //         }
            //     })
            //     .collect();

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

            let ctx: Context = Context {
                allocation_plan: &self.allocation_plan,
            };

            // Execute with mask (all-true for now)
            let mask = BitView::all_true();

            if let Poll::Ready(()) = operator.step(&ctx, mask, &mut output)? {
                // If the operator completed successfully, we remove this node from the
                // next_nodes list and push its parents.
                self.node_states[node_idx] = NodeState::Completed;
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
            if self.next_nodes.is_empty() {
                self.reset_step();
                Poll::Ready(Ok(false)) // No more work for the current step
            } else {
                Poll::Ready(Ok(true))
            }
        }
    }

    // Reset the state for the next step of the pipeline.
    fn reset_step(&mut self) {
        self.node_states.iter_mut().for_each(|state| {
            *state = NodeState::NotStarted;
        });
        self.next_nodes.clear();
        self.next_nodes.extend(self.leaf_nodes.iter().cloned());
    }
}

struct Context<'a> {
    allocation_plan: &'a VectorAllocationPlan,
}

impl<'a> PipelineContext for Context<'a> {
    fn buffer(&self, _buffer_id: BufferId) -> Poll<VortexResult<ByteBuffer>> {
        todo!()
    }

    fn vector(&self, vector_id: VectorId) -> VectorRef {
        VectorRef::new(self.allocation_plan.vectors[*vector_id].borrow())
    }
}

#[cfg(test)]
mod test {
    use crate::pipeline::N;
    use crate::pipeline::buffers::BufferHandle;
    use crate::pipeline::nodes::common::PrimitiveSource;
    use crate::pipeline::nodes::pipeline::Pipeline;
    use crate::pipeline::nodes::plan::PlanNode;
    use crate::pipeline::vector::Vector;
    use std::cell::RefCell;
    use std::task::Poll;
    use vortex_buffer::buffer;
    use vortex_error::vortex_panic;

    #[test]
    fn test_pipeline() {
        // First, let's construct a simple pipeline with a unary operator.
        let data = buffer![0..10000];
        let nchunks = data.len().next_multiple_of(N);
        let src = PrimitiveSource::new(data.len(), BufferHandle::new(data.into_byte_buffer()));

        let mut out = RefCell::new(Vector::new_with_vtype(src.output_type()));

        let mut pipeline = Pipeline::new(&src).unwrap();
        for _ in 0..nchunks {
            let mut more_work = true;
            println!("DOING WORK");
            while more_work {
                more_work = match pipeline.step(&mut out) {
                    Poll::Ready(more_work) => more_work.unwrap(),
                    Poll::Pending => {
                        vortex_panic!("Pending for in-memory pipeline")
                    }
                };
            }
        }

        assert!(false);
    }
}
