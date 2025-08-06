// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::pipeline::VectorId;
use crate::pipeline::nodes::operator::Operator;
use crate::pipeline::nodes::pipeline::Pipeline;
use crate::pipeline::nodes::pipeline::buffers::VectorAllocationPlan;
use crate::pipeline::nodes::pipeline::dag::DagNode;
use crate::pipeline::nodes::plan::BindContext;
use vortex_error::{VortexExpect, VortexResult};

impl Pipeline<'_> {
    pub(super) fn bind_operators(
        dag: &[DagNode],
        allocation_plan: &VectorAllocationPlan,
    ) -> VortexResult<Vec<Box<dyn Operator>>> {
        let mut operators = Vec::with_capacity(dag.len());
        for node in dag {
            let input_ids = node
                .children
                .iter()
                .map(|node_idx| {
                    allocation_plan.output_targets[*node_idx]
                        .vector_id()
                        .vortex_expect("Input node must have an output vector ID")
                })
                .collect::<Vec<_>>();
            let bind_context = PipelineBindContext { input_ids };
            let operator = node.plan_node.bind(&bind_context)?;
            operators.push(operator);
        }
        Ok(operators)
    }
}

struct PipelineBindContext {
    input_ids: Vec<VectorId>,
}

impl BindContext for PipelineBindContext {
    fn input_ids(&self) -> &[VectorId] {
        &self.input_ids
    }
}
