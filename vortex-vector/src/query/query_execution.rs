use std::task::Poll;

use vortex_error::VortexResult;

use crate::Kernel;
use crate::bits::BitView;
use crate::query::buffers::OutputTarget;
use crate::query::{Context, ExecutionResult, NodeState};
use crate::view::ViewMut;

pub struct QueryExecution {
    /// The operators bound to each node in the DAG.
    pub operators: Vec<Box<dyn Kernel>>,

    /// The current state of each node in the DAG, indexed by position in `dag`.
    pub node_states: Vec<NodeState>,

    // Pre-allocated work lists (sized to max possible nodes)
    /// The current stack of nodes to execute.
    pub work_stack: Vec<usize>,
    /// Nodes that returned pending during the last step.
    pub pending_nodes: Vec<usize>,
    /// A scratch list for pending nodes that we flip-flop with `pending_set` to avoid allocations.
    pub pending_nodes_next: Vec<usize>,
}

impl QueryExecution {
    pub fn new() -> Self {
        QueryExecution {
            operators: Vec::new(),
            node_states: Vec::new(),
            work_stack: Vec::new(),
            pending_nodes: Vec::new(),
            pending_nodes_next: Vec::new(),
        }
    }

    pub fn _seek(&mut self, chunk_idx: usize) -> VortexResult<()> {
        self.operators
            .iter_mut()
            .try_for_each(|op| op.seek(chunk_idx))
    }

    /// Step the pipeline forward
    pub fn _step(&mut self, selected: BitView, out: &mut ViewMut) -> VortexResult<()> {
        // self.work_stack.clear();
        // self.pending_nodes_next.clear();
        //
        // // Start with leaf nodes
        // self.work_stack.extend(
        //     self.leaf_nodes
        //         .iter()
        //         .filter(|&&idx| self.node_states[idx] == NodeState::NotStarted)
        //         .copied(),
        // );
        //
        // loop {
        //     // Retry pending nodes first
        //     while let Some(node_idx) = self.pending_nodes.pop() {
        //         match self.try_execute_node(node_idx, selected, out) {
        //             ExecutionResult::Completed => {
        //                 // Add ready parents for cache locality
        //                 self.push_ready_parents(node_idx);
        //             }
        //             ExecutionResult::Pending => {
        //                 // Keep in pending set
        //                 self.pending_nodes_next.push(node_idx);
        //             }
        //             ExecutionResult::Error(e) => return Err(e),
        //             ExecutionResult::NotReady => {
        //                 // Dependencies not ready, skip
        //             }
        //         }
        //     }
        //
        //     std::mem::swap(&mut self.pending_nodes, &mut self.pending_nodes_next);
        //     self.pending_nodes_next.clear();
        //
        //     // Process work stack
        //     if let Some(node_idx) = self.work_stack.pop() {
        //         match self.try_execute_node(node_idx, selected, out) {
        //             ExecutionResult::Completed => {
        //                 // Execute entire parent chain for maximum cache locality
        //                 self.execute_parent_chain(node_idx, selected, out);
        //             }
        //             ExecutionResult::Pending => {
        //                 self.pending_nodes.push(node_idx);
        //             }
        //             ExecutionResult::Error(e) => return Poll::Ready(Err(e)),
        //             ExecutionResult::NotReady => {}
        //         }
        //     } else if self.pending_nodes.is_empty() {
        //         break;
        //     }
        // }
        //
        // if !self.pending_nodes.is_empty() {
        //     Poll::Pending
        // } else if self.node_states[self.dag_root] == NodeState::Completed {
        //     self.reset_step();
        //     Poll::Ready(Ok(()))
        // } else {
        //     Poll::Ready(Ok(()))
        // }
        todo!()
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

        // FIXME(ngates): should we reset the output vector selection?

        match self.allocation_plan.output_targets[node_idx] {
            OutputTarget::ExternalOutput => operator.step(&ctx, selected, external_out),
            OutputTarget::IntermediateVector(vector_idx) | OutputTarget::InPlace(_, vector_idx) => {
                let mut vector_ref = self.allocation_plan.vectors[vector_idx].borrow_mut();
                let result = {
                    let mut view = vector_ref.as_view_mut();
                    operator.step(&ctx, selected, &mut view)
                };
                vector_ref.deref_mut().set_len(selected.true_count());
                result
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
