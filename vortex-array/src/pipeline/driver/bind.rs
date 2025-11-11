// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::{VortexExpect, VortexResult};
use vortex_vector::Vector;

use crate::array::ArrayOperator;
use crate::pipeline::driver::Node;
use crate::pipeline::driver::allocation::VectorAllocation;
use crate::pipeline::{BindContext, Kernel, VectorId};

pub(crate) fn bind_kernels(
    dag: &[Node],
    allocation_plan: &VectorAllocation,
    all_batch_inputs: &[Vector],
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

        let batch_inputs: Vec<_> = node
            .batch_inputs
            .iter()
            .map(|idx| all_batch_inputs[*idx].clone())
            .collect();

        let bind_context = PipelineBindContext {
            children: &input_ids,
            batch_inputs: &batch_inputs,
        };

        let pipelined = node
            .array
            .as_pipelined()
            .vortex_expect("Array in pipeline DAG does not support pipelined execution");

        kernels.push(pipelined.bind(&bind_context)?);
    }
    Ok(kernels)
}

struct PipelineBindContext<'a> {
    children: &'a [VectorId],
    batch_inputs: &'a [Vector],
}

impl BindContext for PipelineBindContext<'_> {
    fn pipelined_input(&self, pipelined_child_idx: usize) -> VectorId {
        self.children[pipelined_child_idx]
    }

    fn batch_input(&self, batch_child_idx: usize) -> Vector {
        self.batch_inputs[batch_child_idx].clone()
    }
}
