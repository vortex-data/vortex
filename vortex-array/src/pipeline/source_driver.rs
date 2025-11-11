// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_vector::{Vector, VectorMut, VectorMutOps};

use crate::pipeline::{BindContext, N, PipelineSource};

/// Temporary driver for executing a single source array in a pipelined fashion.
pub struct PipelineSourceDriver<'a> {
    array: &'a dyn PipelineSource,
}

impl<'a> PipelineSourceDriver<'a> {
    pub fn new(array: &'a dyn PipelineSource) -> Self {
        Self { array }
    }

    pub fn execute(&self) -> VortexResult<Vector> {
        // First, we compute all child vectors.
        // Since this is a pipeline source, we know that remaining children must be batch inputs,
        // and therefore we cannot push down the selection mask.
        let batch_inputs: Vec<_> = self
            .array
            .children()
            .iter()
            .map(|child| child.execute())
            .try_collect()?;

        // We now construct the source kernel.
        let mut bind_ctx = PipelineSourceBindCtx {
            batch_inputs: &batch_inputs,
        };
        let mut kernel = self.array.bind(&mut bind_ctx)?;

        // Allocate an output vector, with up to N bytes of padding to ensure every call to
        // `kernel.step(out)` has at least N bytes of capacity.
        let mut output = VectorMut::with_capacity(
            self.array.dtype(),
            // We add an extra N to ensure we have enough capacity for 2 * N elements when we
            // invoke the source kernel for the final time.
            self.array.len().next_multiple_of(N) + N,
        );

        while output.len() < self.array.len() {
            kernel.step(&mut output)?;
        }
        assert_eq!(
            output.len(),
            self.array.len(),
            "Pipeline source produced incorrect number of rows"
        );

        Ok(output.freeze())
    }
}

struct PipelineSourceBindCtx<'a> {
    batch_inputs: &'a [Vector],
}

impl BindContext for PipelineSourceBindCtx<'_> {
    fn batch_input(&self, child_idx: usize) -> Vector {
        self.batch_inputs[child_idx].clone()
    }
}
