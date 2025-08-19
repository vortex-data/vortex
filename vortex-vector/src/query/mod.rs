// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod buffers;
mod dag;
mod operators;
mod query_execution;
mod toposort;

use std::ops::DerefMut;
use std::task::Poll;

use vortex_buffer::ByteBuffer;
use vortex_error::{VortexError, VortexResult};

use crate::bits::BitView;
use crate::buffers::BufferId;
use crate::operators::Operator;
use crate::query::buffers::{OutputTarget, VectorAllocationPlan};
use crate::query::dag::DagNode;
use crate::query::query_execution::QueryExecution;
use crate::vector::{VectorId, VectorRef};
use crate::view::ViewMut;
use crate::{Kernel, KernelContext};

/// The idea of a query-plan is to orchestrate driving a set of operators to completion with
/// fully optimized resource usage.
///
/// During construction, the plan is analyzed to determine the optimal way to execute the nodes.
/// This includes:
/// - Sub-expression elimination: Identifying common sub-expressions and reusing them.
/// - Vector allocation: Determining how many intermediate vectors are needed.
/// - Buffer management: Managing the buffers that hold the data for each node.
pub struct QueryPlan<'a> {
    /// Nodes in the DAG representing the execution plan with common sub-expressions eliminated.
    dag: Vec<DagNode<'a>>,
    /// The index into the `dag` of the root node (the entry point for execution).
    dag_root: usize,

    /// The topological order of `dag` nodes for execution.
    execution_order: Vec<usize>,
    /// The leaf nodes of the plan (nodes with no children).
    leaf_nodes: Vec<usize>,
    /// The allocation plan for vectors used by the pipeline.
    allocation_plan: VectorAllocationPlan,
}

impl<'a> QueryPlan<'a> {
    // TODO(ngates): can we pass the mask in here such that the plan can replace empty nodes?
    pub fn new(plan: &'a dyn Operator) -> VortexResult<Self> {
        // Step 1: Convert the plan tree to a DAG by eliminating common sub-expressions.
        let (dag_root, dag) = Self::build_dag(plan)?;
        let node_count = dag.len();

        // Step 2: Determine execution order (topological sort)
        let execution_order = Self::topological_sort(&dag)?;
        let leaf_nodes = Self::leaf_nodes(&dag);

        // Step 3: Allocate vectors
        let allocation_plan = Self::allocate_vectors(dag_root, &dag, &execution_order)?;

        Ok(Self {
            dag,
            dag_root,
            execution_order,
            leaf_nodes,
            allocation_plan,
        })
    }

    pub fn executable_plan(&self) -> VortexResult<QueryExecution> {
        let node_count = self.dag.len();
        // Construct the operators, binding their inputs using the allocation plan.
        let operators = Self::bind_operators(&self.dag, &self.allocation_plan)?;

        Ok(QueryExecution {
            operators: operators.clone(),
            // Pre-allocate work arrays
            node_states: vec![NodeState::NotStarted; node_count],
            work_stack: Vec::new(),
            pending_nodes: Vec::new(),
            pending_nodes_next: Vec::new(),
        })
    }
}

/// Execution state for a node
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NodeState {
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
impl Kernel for QueryExecution {
    fn seek(&mut self, chunk_idx: usize) -> VortexResult<()> {
        self._seek(chunk_idx)
    }

    fn step(
        &mut self,
        ctx: &dyn KernelContext,
        selected: BitView,
        out: &mut ViewMut,
    ) -> Poll<VortexResult<()>> {
        self._step(selected, out)
    }
}

struct Context<'a> {
    allocation_plan: &'a VectorAllocationPlan,
}

impl<'a> KernelContext for Context<'a> {
    fn buffer(&self, _buffer_id: BufferId) -> Poll<VortexResult<ByteBuffer>> {
        todo!()
    }

    fn vector(&self, vector_id: VectorId) -> VectorRef<'_> {
        VectorRef::new(self.allocation_plan.vectors[*vector_id].borrow())
    }
}
