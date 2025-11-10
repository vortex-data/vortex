// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::{VortexResult, vortex_panic};
use vortex_mask::Mask;
use vortex_vector::{Vector, VectorMut, VectorMutOps};

use crate::pipeline::bit_view::{BitView, BitViewExt};
use crate::pipeline::{BindContext, KernelContext, N, PipelineSource, VectorId};

/// Temporary driver for executing a single source array in a pipelined fashion.
pub struct PipelineSourceDriver<'a> {
    array: &'a dyn PipelineSource,
}

impl<'a> PipelineSourceDriver<'a> {
    pub fn new(array: &'a dyn PipelineSource) -> Self {
        Self { array }
    }

    pub fn execute(&self, selection: &Mask) -> VortexResult<Vector> {
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
        let kernel_ctx = KernelContext::empty();

        // Allocate an output vector, with up to N bytes of padding to ensure every call to
        // `kernel.step(out)` has at least N bytes of capacity.
        let mut output = VectorMut::with_capacity(
            self.array.dtype(),
            // We add an extra N to ensure we have enough capacity so the last chunk has 2 * N
            // elements of capacity.
            selection.true_count().next_multiple_of(N) + N,
        );

        match selection {
            Mask::AllTrue(_) => {
                // Select everything, so we can just run the kernel in a tight loop.

                // The number of _full_ chunks we need to process.
                let nchunks = selection.len() / N;
                for _ in 0..nchunks {
                    let prev_len = output.len();
                    kernel.step(&kernel_ctx, &BitView::all_true(), &mut output)?;
                    debug_assert_eq!(output.len(), prev_len + N);
                }

                // Now process the final partial chunk, if any.
                let remaining = selection.len() % N;
                if remaining > 0 {
                    let selection_view = BitView::with_prefix(remaining);

                    let prev_len = output.len();
                    kernel.step(&kernel_ctx, &selection_view, &mut output)?;
                    debug_assert_eq!(output.len(), prev_len + remaining);
                    debug_assert_eq!(output.len(), selection.len());
                }
            }
            Mask::AllFalse(_) => {
                // Select nothing, return empty output!
            }
            Mask::Values(values) => {
                // Mixed selection, so we have to process in chunks.
                let selection_bits = values.bit_buffer();
                for selection_view in selection_bits.iter_bit_views() {
                    let prev_len = output.len();
                    kernel.step(&kernel_ctx, &selection_view, &mut output)?;
                    debug_assert_eq!(output.len(), prev_len + selection_view.true_count());
                }
            }
        }

        Ok(output.freeze())
    }
}

struct PipelineSourceBindCtx<'a> {
    batch_inputs: &'a [Vector],
}

impl BindContext for PipelineSourceBindCtx<'_> {
    fn pipelined_input(&self, _child_idx: usize) -> VectorId {
        vortex_panic!("PipelineSource cannot bind pipelined inputs");
    }

    fn batch_input(&self, child_idx: usize) -> Vector {
        self.batch_inputs[child_idx].clone()
    }
}
