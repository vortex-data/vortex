// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod buffers;
mod dag;
mod operators;
mod query_execution;
mod toposort;

use crate::operator::Operator;
use crate::pipeline::bits::BitView;
pub use crate::pipeline::query::buffers::VectorAllocationPlan;
use crate::pipeline::query::dag::DagNode;
use crate::pipeline::query::query_execution::QueryExecution;
use crate::pipeline::view::ViewMut;
use crate::pipeline::{Kernel, KernelContext};
use vortex_error::VortexResult;

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
    /// The topological order of `dag` nodes for execution.
    execution_order: Vec<usize>,
    /// The allocation plan for vectors used by the pipeline.
    allocation_plan: VectorAllocationPlan,
}

impl<'a> QueryPlan<'a> {
    // TODO(ngates): can we pass the mask in here such that the plan can replace empty nodes?
    pub fn new(plan: &'a dyn Operator) -> VortexResult<Self> {
        // Step 1: Convert the plan tree to a DAG by eliminating common sub-expressions.
        let (_dag_root, dag) = Self::build_dag(plan)?;
        let _node_count = dag.len();

        // Step 2: Determine execution order (topological sort)
        let execution_order = Self::topological_sort(&dag)?;

        // Step 3: Allocate vectors
        let allocation_plan = Self::allocate_vectors(&dag, &execution_order)?;

        Ok(Self {
            dag,
            execution_order,
            allocation_plan,
        })
    }

    pub fn executable_plan(self) -> VortexResult<QueryExecution> {
        // Construct the operators, binding their inputs using the allocation plan.
        let operators = Self::bind_operators(&self.dag, &self.allocation_plan)?;
        let kernel_context = KernelContext::new(self.allocation_plan.vectors);

        Ok(QueryExecution {
            operators,
            execution_schedule: self.execution_order,
            kernel_context,
            output_targets: self.allocation_plan.output_targets,
        })
    }
}

/// FIXME(ngates): this is a hack for testing
impl Kernel for QueryExecution {
    fn seek(&mut self, chunk_idx: usize) -> VortexResult<()> {
        self._seek(chunk_idx)
    }

    fn step(
        &mut self,
        _ctx: &KernelContext,
        selected: BitView,
        out: &mut ViewMut,
    ) -> VortexResult<()> {
        self._step(selected, out)
    }
}
