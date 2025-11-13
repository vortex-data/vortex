// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;

use vortex_error::{VortexExpect, VortexResult};
use vortex_vector::{Vector, VectorOps};

use crate::array::ArrayOperator;
use crate::pipeline::driver::allocation::VectorAllocation;
use crate::pipeline::driver::input::InputKernel;
use crate::pipeline::driver::{Node, NodeId, NodeKind};
use crate::pipeline::{BindContext, Kernel, VectorId};

// We consume the DAG and the batch inputs such that our into_mut calls are safe
pub(crate) fn bind_kernels(
    dag: Vec<Node>,
    execution_order: &[NodeId],
    allocation_plan: &VectorAllocation,
    mut all_batch_inputs: Vec<Option<Vector>>,
) -> VortexResult<Vec<Box<dyn Kernel>>> {
    // We construct kernels in top-down order (i.e. inverse execution order) such that any arrays
    // in parent nodes are dropped prior to constructing their children. This gives us the best
    // chance of performing zero-copy into_mut calls on the pipeline input arrays.

    let mut dag: Vec<_> = dag.into_iter().map(Some).collect();
    let mut kernels: Vec<Option<Box<dyn Kernel>>> =
        iter::repeat_with(|| None).take(dag.len()).collect();

    // Note that the execution order is bottom-up, so to make sure we consume an array's parent
    // before construct the array's own kernel, we must construct kernels in reverse order.
    for node_id in execution_order.iter().copied().rev() {
        let node = dag[node_id].take().vortex_expect("already processed");

        let input_ids = node
            .children
            .iter()
            .map(|child_node_id| allocation_plan.output_targets[*child_node_id].vector_id())
            .collect::<Vec<_>>();

        let mut batch_inputs: Vec<_> = node
            .batch_inputs
            .iter()
            .map(|idx| all_batch_inputs[*idx].take())
            .collect();

        kernels[node_id] = Some(match node.array.as_pipelined() {
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
                    .vortex_expect("Batch input vector has already been consumed")
                    .into_mut();

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

    Ok(kernels
        .into_iter()
        .map(|k| k.vortex_expect("missing kernel"))
        .collect())
}

struct PipelineBindContext<'a> {
    children: &'a [Option<VectorId>],
    batch_inputs: &'a mut [Option<Vector>],
}

impl BindContext for PipelineBindContext<'_> {
    fn pipelined_input(&self, pipelined_child_idx: usize) -> VectorId {
        self.children[pipelined_child_idx]
            .vortex_expect("In-place transforms do not have VectorIDs for their zero'th child.")
    }

    fn batch_input(&mut self, batch_child_idx: usize) -> Vector {
        self.batch_inputs[batch_child_idx]
            .take()
            .vortex_expect("Batch input vector has already been consumed")
    }
}
