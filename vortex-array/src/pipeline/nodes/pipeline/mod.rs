// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod buffers;
mod dag;
mod operators;
mod toposort;

use crate::pipeline::bits::BitView;
use crate::pipeline::buffers::BufferId;
use crate::pipeline::nodes::pipeline::buffers::{OutputTarget, VectorAllocationPlan};
use crate::pipeline::nodes::pipeline::dag::DagNode;
use crate::pipeline::nodes::plan::PlanNode;
use crate::pipeline::vector::{Vector, VectorId, VectorRef, VectorRefMut};
use crate::pipeline::view::ViewMut;
use crate::pipeline::{Operator, PipelineContext};
use std::cell::RefCell;
use std::ops::{Deref, DerefMut};
use std::task::Poll;
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexError, VortexResult};

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

    // Pre-allocated work lists (sized to max possible nodes)
    /// The current stack of nodes to execute.
    work_stack: Vec<usize>,
    /// Nodes that returned pending during the last step.
    pending_nodes: Vec<usize>,
    /// A scratch list for pending nodes that we flip-flop with `pending_set` to avoid allocations.
    pending_nodes_next: Vec<usize>,
}

impl<'a> Pipeline<'a> {
    // TODO(ngates): can we pass the mask in here such that the plan can replace empty nodes?
    pub fn new(plan: &'a dyn PlanNode) -> VortexResult<Self> {
        // Step 1: Convert the plan tree to a DAG by eliminating common sub-expressions.
        let (dag_root, dag) = Self::build_dag(plan)?;
        let node_count = dag.len();

        // Step 2: Determine execution order (topological sort)
        let execution_order = Self::topological_sort(&dag)?;
        let leaf_nodes = Self::leaf_nodes(&dag);

        // Step 3: Allocate vectors
        let allocation_plan = Self::allocate_vectors(dag_root, &dag, &execution_order)?;
        log::info!("Allocation plan: {allocation_plan:?}");

        // let (buffer_slots, buffers) = Self::allocate_buffers(&dag, &execution_order)?;

        // Construct the operators, binding their inputs using the allocation plan.
        let operators = Self::bind_operators(&dag, &allocation_plan)?;

        Ok(Self {
            dag,
            dag_root,
            execution_order,
            leaf_nodes,
            operators,
            allocation_plan,
            node_states: vec![NodeState::NotStarted; node_count],
            // Pre-allocate work arrays
            work_stack: Vec::with_capacity(node_count),
            pending_nodes: Vec::with_capacity(node_count),
            pending_nodes_next: Vec::with_capacity(node_count),
        })
    }

    /// Step the pipeline forward
    pub fn step(&mut self, selected: BitView, out: &mut ViewMut) -> Poll<VortexResult<()>> {
        self.work_stack.clear();
        self.pending_nodes_next.clear();

        // Start with leaf nodes
        self.work_stack.extend(
            self.leaf_nodes
                .iter()
                .filter(|&&idx| self.node_states[idx] == NodeState::NotStarted)
                .copied(),
        );

        loop {
            // Retry pending nodes first
            while let Some(node_idx) = self.pending_nodes.pop() {
                match self.try_execute_node(node_idx, selected, out) {
                    ExecutionResult::Completed => {
                        // Add ready parents for cache locality
                        self.push_ready_parents(node_idx);
                    }
                    ExecutionResult::Pending => {
                        // Keep in pending set
                        self.pending_nodes_next.push(node_idx);
                    }
                    ExecutionResult::Error(e) => return Poll::Ready(Err(e)),
                    ExecutionResult::NotReady => {
                        // Dependencies not ready, skip
                    }
                }
            }

            std::mem::swap(&mut self.pending_nodes, &mut self.pending_nodes_next);
            self.pending_nodes_next.clear();

            // Process work stack
            if let Some(node_idx) = self.work_stack.pop() {
                match self.try_execute_node(node_idx, selected, out) {
                    ExecutionResult::Completed => {
                        // Execute entire parent chain for maximum cache locality
                        self.execute_parent_chain(node_idx, selected, out);
                    }
                    ExecutionResult::Pending => {
                        self.pending_nodes.push(node_idx);
                    }
                    ExecutionResult::Error(e) => return Poll::Ready(Err(e)),
                    ExecutionResult::NotReady => {}
                }
            } else if self.pending_nodes.is_empty() {
                break;
            }
        }

        if !self.pending_nodes.is_empty() {
            Poll::Pending
        } else if self.node_states[self.dag_root] == NodeState::Completed {
            self.reset_step();
            Poll::Ready(Ok(()))
        } else {
            Poll::Ready(Ok(()))
        }
    }

    /// Execute chain of ready parents while data is in cache
    #[inline]
    fn execute_parent_chain(&mut self, mut node_idx: usize, selected: BitView, out: &mut ViewMut) {
        loop {
            // Find a ready parent
            let ready_parent = self.find_ready_parent(node_idx);

            match ready_parent {
                Some(parent_idx) => {
                    match self.try_execute_node(parent_idx, selected, out) {
                        ExecutionResult::Completed => {
                            // Continue up the chain
                            node_idx = parent_idx;
                        }
                        ExecutionResult::Pending => {
                            self.pending_nodes.push(parent_idx);
                            break;
                        }
                        ExecutionResult::Error(_) | ExecutionResult::NotReady => break,
                    }
                }
                None => {
                    // No ready parent, check for other ready parents to queue
                    self.push_ready_parents(node_idx);
                    break;
                }
            }
        }
    }

    /// Find a single ready parent (for chain execution)
    #[inline]
    fn find_ready_parent(&self, node_idx: usize) -> Option<usize> {
        let node = &self.dag[node_idx];

        node.parents.iter().copied().find(|&parent_idx| {
            if self.node_states[parent_idx] != NodeState::NotStarted {
                return false;
            }

            let parent = &self.dag[parent_idx];
            parent
                .children
                .iter()
                .all(|&child| self.node_states[child] == NodeState::Completed)
        })
    }

    /// Push ready parents to work stack (no allocation - capacity pre-allocated)
    #[inline]
    fn push_ready_parents(&mut self, completed_node: usize) {
        let node = &self.dag[completed_node];

        for &parent_idx in &node.parents {
            // Skip if already processed
            if self.node_states[parent_idx] != NodeState::NotStarted {
                continue;
            }

            // Check if all children completed
            let parent = &self.dag[parent_idx];
            let all_children_done = parent
                .children
                .iter()
                .all(|&child| self.node_states[child] == NodeState::Completed);

            if all_children_done {
                // Push to work stack - won't allocate due to capacity
                self.work_stack.push(parent_idx);
            }
        }
    }

    /// Try to execute a node if ready
    #[inline]
    fn try_execute_node(
        &mut self,
        node_idx: usize,
        selected: BitView,
        out: &mut ViewMut,
    ) -> ExecutionResult {
        // Check current state
        match self.node_states[node_idx] {
            NodeState::Completed => return ExecutionResult::Completed,
            NodeState::Executing | NodeState::Pending => {
                // Try to continue execution
            }
            NodeState::NotStarted => {
                // Check if dependencies are ready
                // FIXME(ngates): is this ever not true?
                let node = &self.dag[node_idx];
                let ready = node
                    .children
                    .iter()
                    .all(|&child| self.node_states[child] == NodeState::Completed);
                if !ready {
                    return ExecutionResult::NotReady;
                }

                self.node_states[node_idx] = NodeState::Executing;
            }
        }

        // Execute the node
        match self.execute_single_node(node_idx, selected, out) {
            Poll::Ready(Ok(())) => {
                self.node_states[node_idx] = NodeState::Completed;
                ExecutionResult::Completed
            }
            Poll::Pending => {
                self.node_states[node_idx] = NodeState::Pending;
                ExecutionResult::Pending
            }
            Poll::Ready(Err(e)) => ExecutionResult::Error(e),
        }
    }

    /// Execute a single node
    #[inline]
    fn execute_single_node(
        &mut self,
        node_idx: usize,
        selected: BitView,
        external_out: &mut ViewMut,
    ) -> Poll<VortexResult<()>> {
        let operator = self.operators[node_idx].as_mut();

        let ctx = Context {
            allocation_plan: &self.allocation_plan,
        };

        match self.allocation_plan.output_targets[node_idx] {
            OutputTarget::ExternalOutput => operator.step(&ctx, selected, external_out),
            OutputTarget::IntermediateVector(vector_idx) | OutputTarget::InPlace(_, vector_idx) => {
                let mut vector_ref = self.allocation_plan.vectors[vector_idx].borrow_mut();
                let mut view = vector_ref.as_view_mut();
                operator.step(&ctx, selected, &mut view)
            }
        }
    }

    /// Reset state for next pipeline step
    #[inline]
    fn reset_step(&mut self) {
        // Reset all node states
        self.node_states.fill(NodeState::NotStarted);

        // Clear work lists (doesn't deallocate)
        self.work_stack.clear();
        self.pending_nodes.clear();
        self.pending_nodes_next.clear();
    }
}

/// Execution state for a node
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NodeState {
    /// Node has not been executed yet
    NotStarted,
    /// Node is currently executing (may return Poll::Pending)
    Executing,
    /// Node is waiting for external resources (e.g. buffers) to become available
    Pending,
    /// Node has completed execution
    Completed,
}

enum ExecutionResult {
    Completed,
    Pending,
    NotReady,
    Error(VortexError),
}

/// FIXME(ngates): this is a hack for testing
impl Operator for Pipeline<'_> {
    fn seek(&mut self, chunk_idx: usize) -> VortexResult<()> {
        todo!()
    }

    fn step(
        &mut self,
        ctx: &dyn PipelineContext,
        selected: BitView,
        out: &mut ViewMut,
    ) -> Poll<VortexResult<()>> {
        Pipeline::step(self, selected, out)
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
    use crate::pipeline::bits::BitView;
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

        let mut out = Vector::new_with_vtype(src.output_type());

        let mut pipeline = Pipeline::new(&src).unwrap();
        for i in 0..nchunks {
            match pipeline.step(BitView::all_true(), &mut out.as_view_mut()) {
                Poll::Ready(result) => result.unwrap(),
                Poll::Pending => {
                    vortex_panic!("Pending for in-memory pipeline")
                }
            }
        }

        assert!(false);
    }
}
