// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::DerefMut;

use vortex_error::VortexResult;

use crate::Kernel;
use crate::bits::BitView;
use crate::query::Context;
use crate::query::buffers::{OutputTarget, VectorAllocationPlan};
use crate::view::ViewMut;

pub struct QueryExecution {
    /// The operators bound to each node in the DAG.
    pub operators: Vec<Box<dyn Kernel>>,
    /// Static execution order determined by topological sort
    pub execution_schedule: Vec<usize>,
    /// Vector allocation plan for intermediate results
    pub(crate) allocation_plan: VectorAllocationPlan,
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
            allocation_plan: VectorAllocationPlan {
                output_targets: Vec::new(),
                vectors: Vec::new(),
            },
        }
    }

    pub fn _seek(&mut self, chunk_idx: usize) -> VortexResult<()> {
        self.operators
            .iter_mut()
            .try_for_each(|op| op.seek(chunk_idx))
    }

    /// Step the pipeline forward - executes all nodes in static topological order
    pub fn _step(&mut self, selected: BitView, out: &mut ViewMut) -> VortexResult<()> {
        for node_idx in self.execution_schedule.iter() {
            let node_idx = *node_idx;
            let operator = self.operators[node_idx].as_mut();

            let ctx = Context {
                allocation_plan: &self.allocation_plan,
            };

            // FIXME(ngates): should we reset the output vector selection?

            match self.allocation_plan.output_targets[node_idx] {
                OutputTarget::ExternalOutput => operator.step(&ctx, selected, out)?,
                OutputTarget::IntermediateVector(vector_idx) => {
                    let mut vector_ref = self.allocation_plan.vectors[vector_idx].borrow_mut();
                    let result = {
                        let mut view = vector_ref.as_view_mut();
                        operator.step(&ctx, selected, &mut view)
                    };
                    vector_ref.deref_mut().set_len(selected.true_count());
                    result?
                }
            }
        }
        Ok(())
    }
}
