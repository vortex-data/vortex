// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(dead_code)]

use vortex_error::{VortexExpect, VortexResult};

use crate::pipeline::operator::buffers::VectorAllocationPlan;
use crate::pipeline::operator::PipelineNode;
use crate::pipeline::{BatchId, BindContext, Kernel, VectorId};

pub(crate) fn bind_kernels(
    dag: &[PipelineNode],
    allocation_plan: &VectorAllocationPlan,
) -> VortexResult<Vec<Box<dyn Kernel>>> {
    let mut kernels = Vec::with_capacity(dag.len());
    for node in dag {
        let input_ids = node
            .children
            .iter()
            .map(|node_id| {
                allocation_plan.output_targets[*node_id]
                    .vector_id()
                    .vortex_expect("Input node must have an output vector ID")
            })
            .collect::<Vec<_>>();

        let bind_context = PipelineBindContext {
            children: &input_ids,
            batch_inputs: &node.batch_inputs,
        };

        let pipelined = node.operator.as_pipelined().ok_or_else(|| {
            vortex_error::vortex_err!("Operator does not support pipelined execution")
        })?;
        kernels.push(pipelined.bind(&bind_context)?);
    }
    Ok(kernels)
}

struct PipelineBindContext<'a> {
    children: &'a [VectorId],
    batch_inputs: &'a [BatchId],
}

impl BindContext for PipelineBindContext<'_> {
    fn pipelined_input(&self) -> &[VectorId] {
        self.children
    }

    fn batch_inputs(&self) -> &[BatchId] {
        self.batch_inputs
    }
}
