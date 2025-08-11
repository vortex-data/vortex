// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::{VortexExpect, VortexResult};

use crate::pipeline::Kernel;
use crate::pipeline::operators::BindContext;
use crate::pipeline::query::Pipeline;
use crate::pipeline::query::buffers::VectorAllocationPlan;
use crate::pipeline::query::dag::DagNode;
use crate::pipeline::vector::VectorId;

impl Pipeline<'_> {
    pub(in crate::pipeline) fn bind_operators(
        dag: &[DagNode],
        allocation_plan: &VectorAllocationPlan,
    ) -> VortexResult<Vec<Box<dyn Kernel>>> {
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
            let bind_context = PipelineBindContext {
                children: input_ids,
            };
            let operator = node.plan_node.bind(&bind_context)?;
            operators.push(operator);
        }
        Ok(operators)
    }
}

struct PipelineBindContext {
    children: Vec<VectorId>,
}

impl BindContext for PipelineBindContext {
    fn children(&self) -> &[VectorId] {
        &self.children
    }
}
