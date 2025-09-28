// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::{VortexExpect, VortexResult};

use crate::pipeline::operator::PipelineNode;
use crate::pipeline::operator::allocation::VectorAllocationPlan;
use crate::pipeline::{BatchId, BindContext, Kernel, VectorHandle};

pub(crate) fn bind_kernels(
    dag: &[PipelineNode],
    allocation_plan: &VectorAllocationPlan,
) -> VortexResult<Vec<Box<dyn Kernel>>> {
    let mut kernels = Vec::with_capacity(dag.len());
    for node in dag {
        let input_handles: Vec<_> = node
            .children
            .iter()
            .map(|child_idx| {
                let vector_idx = allocation_plan.output_targets[*child_idx]
                    .vector_idx()
                    .vortex_expect("Input node must have an output vector ID");
                VectorHandle::intermediate_vector(vector_idx)
            })
            .collect();

        let bind_context = PipelineBindContext {
            vector_inputs: &input_handles,
            batch_inputs: &node.batch_input_ids,
        };

        let pipelined = node.operator.as_pipelined().ok_or_else(|| {
            vortex_error::vortex_err!("Operator does not support pipelined execution")
        })?;
        kernels.push(pipelined.bind(&bind_context)?);
    }
    Ok(kernels)
}

struct PipelineBindContext<'a> {
    vector_inputs: &'a [VectorHandle],
    batch_inputs: &'a [BatchId],
}

impl BindContext for PipelineBindContext<'_> {
    /// Returns a handle to the vector input for the given child index.
    ///
    /// This handle can be used to access the vector during kernel execution by requesting it
    /// from the [`KernelContext`](crate::pipeline::operator::KernelContext).
    fn vector_input(&self, child_idx: usize) -> VectorHandle {
        self.vector_inputs[child_idx]
    }

    fn batch_inputs(&self) -> &[BatchId] {
        self.batch_inputs
    }
}
