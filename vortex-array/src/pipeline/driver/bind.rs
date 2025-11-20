// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::{VortexExpect, VortexResult};
use vortex_vector::Vector;

use crate::array::ArrayOperator;
use crate::pipeline::driver::allocation::VectorAllocation;
use crate::pipeline::driver::input::InputKernel;
use crate::pipeline::driver::{Node, NodeKind};
use crate::pipeline::{BindContext, Kernel, VectorId};

pub(crate) fn bind_kernels(
    dag: Vec<Node>,
    allocation_plan: &VectorAllocation,
    mut all_batch_inputs: Vec<Option<Vector>>,
) -> VortexResult<Vec<Box<dyn Kernel>>> {
    let mut kernels = Vec::with_capacity(dag.len());
    for node in dag {
        let input_ids = node
            .children
            .iter()
            .map(|node_id| allocation_plan.output_targets[*node_id])
            .collect::<Vec<_>>();

        let mut batch_inputs: Vec<_> = node
            .batch_inputs
            .iter()
            .map(|idx| all_batch_inputs[*idx].take())
            .collect();

        kernels.push(match node.array.as_pipelined() {
            None => {
                // If the node cannot be pipelined, it must be an input node
                assert_eq!(node.kind, NodeKind::Input);
                assert_eq!(node.batch_inputs.len(), 1);
                let batch_id = node.batch_inputs[0];

                // Release ownership of the array before trying to call into_mut on the vector.
                // This is in case the vector was constructed zero-copy from the array's data.
                drop(node.array);

                let batch = batch_inputs[batch_id]
                    .take()
                    .vortex_expect("Batch input vector has already been consumed");

                Box::new(InputKernel::new(batch))
            }
            Some(pipelined) => {
                let bind_context = PipelineBindContext {
                    children: &input_ids,
                    batch_inputs: &mut batch_inputs,
                };
                pipelined.bind(&bind_context)?
            }
        });
    }
    Ok(kernels)
}

struct PipelineBindContext<'a> {
    children: &'a [VectorId],
    batch_inputs: &'a mut [Option<Vector>],
}

impl BindContext for PipelineBindContext<'_> {
    fn pipelined_input(&self, pipelined_child_idx: usize) -> VectorId {
        self.children[pipelined_child_idx]
    }

    fn batch_input(&mut self, batch_child_idx: usize) -> Vector {
        self.batch_inputs[batch_child_idx]
            .take()
            .vortex_expect("Batch input vector has already been consumed")
    }
}
