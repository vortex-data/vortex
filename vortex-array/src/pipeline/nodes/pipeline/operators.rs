// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::pipeline::nodes::operator::Operator;
use crate::pipeline::nodes::pipeline::Pipeline;
use crate::pipeline::nodes::pipeline::dag::DagNode;
use crate::pipeline::nodes::plan::BindContext;
use vortex_error::VortexResult;

impl Pipeline<'_> {
    pub(super) fn bind_operators(dag: &[DagNode]) -> VortexResult<Vec<Box<dyn Operator>>> {
        let mut operators = Vec::with_capacity(dag.len());
        for node in dag {
            let bind_context = PipelineBindContext;
            let operator = node.plan_node.bind(&bind_context)?;
            operators.push(operator);
        }
        Ok(operators)
    }
}

struct PipelineBindContext;

impl BindContext for PipelineBindContext {}
