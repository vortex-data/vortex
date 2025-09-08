// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::DerefMut;

use vortex_error::VortexResult;

use crate::pipeline::bits::BitView;
use crate::pipeline::query::buffers::OutputTarget;
use crate::pipeline::view::ViewMut;
use crate::pipeline::{Kernel, KernelContext, N};

pub struct QueryExecution {
    /// The operators bound to each node in the DAG.
    pub operators: Vec<Box<dyn Kernel>>,
    /// Static execution order determined by topological sort
    pub execution_schedule: Vec<usize>,
    /// Vector allocation plan for intermediate results
    // pub(crate) allocation_plan: VectorAllocationPlan,
    pub(crate) kernel_context: KernelContext,
    pub(crate) output_targets: Vec<OutputTarget>,
}

impl Default for QueryExecution {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryExecution {
    pub fn new() -> Self {
        QueryExecution {
            operators: Vec::new(),
            execution_schedule: Vec::new(),
            output_targets: Vec::new(),
            kernel_context: KernelContext::default(),
        }
    }

    pub fn _seek(&mut self, chunk_idx: usize) -> VortexResult<()> {
        self.operators
            .iter_mut()
            .try_for_each(|op| op.seek(chunk_idx))
    }

    /// Step the pipeline forward - executes all nodes in static topological order
    pub fn _step(&mut self, selected: BitView, out: &mut ViewMut) -> VortexResult<()> {
        // Resut the vector length between steps.
        self.kernel_context
            .vectors
            .iter()
            .for_each(|v| v.borrow_mut().set_len(N));
        for node_idx in self.execution_schedule.iter() {
            let node_idx = *node_idx;
            let operator = self.operators[node_idx].as_mut();

            match self.output_targets[node_idx] {
                OutputTarget::ExternalOutput => {
                    operator.step(&self.kernel_context, selected, out)?
                }
                OutputTarget::IntermediateVector(vector_idx) => {
                    let mut vector_ref = self.kernel_context.vectors[vector_idx].borrow_mut();
                    let result = {
                        let mut view = vector_ref.as_view_mut();
                        assert_eq!(view.len, N);
                        operator.step(&self.kernel_context, selected, &mut view)
                    };
                    vector_ref.deref_mut().set_len(selected.true_count());
                    result?
                }
            }
        }
        Ok(())
    }
}
