// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::pipeline::bits::BitView;
use crate::pipeline::{BindContext, KernelContext, PipelinedSource, VectorId, N};
use itertools::Itertools;
use vortex_error::{vortex_panic, VortexResult};
use vortex_mask::Mask;
use vortex_vector::{Vector, VectorMut, VectorMutOps};

/// Temporary driver for executing a single array in a pipelined fashion.
pub struct PipelineSourceDriver<'a> {
    array: &'a dyn PipelinedSource,
}

impl<'a> PipelineSourceDriver<'a> {
    pub fn new(array: &'a dyn PipelinedSource) -> Self {
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
        let mut kernel = self.array.bind_source(&mut bind_ctx)?;
        let kernel_ctx = KernelContext::empty();

        // Allocate an output vector, with up to N bytes of padding to ensure every call to
        // `kernel.step(out)` has at least N bytes of capacity.
        let mut output = VectorMut::with_capacity(
            self.array.dtype(),
            selection.true_count().next_multiple_of(N),
        );

        // TODO(ngates): change behaviour based on the density of the selection mask.
        let selection_buffer = selection.to_bit_buffer();
        // TODO(ngates): rewrite chunks to take an arbitrary "storage type"? Or somehow copy
        //  the chunks directly into a wider bit slice?
        let selection_chunks = selection_buffer.chunks();
        let mut selection_chunks_iter = selection_chunks.iter_padded();

        let output_len = selection.true_count();

        let mut selection_chunk = [0u64; N / u64::BITS as usize];

        let mut output_chunks = vec![];
        while output.len() < output_len {
            // Copy the next selection chunk into place.
            for word_idx in 0..selection_chunk.len() {
                selection_chunk[word_idx] = selection_chunks_iter.next().unwrap_or_else(|| 0u64);
            }

            // TODO(ngates): ideally our chunks iter would use a usize...
            let selection_chunk_usize = unsafe { std::mem::transmute(&selection_chunk) };
            let selection = BitView::new(selection_chunk_usize);

            // We know we have remaining capacity for N elements, so split off a size-N chunk.
            let remaining_output = output.split_off(N);

            kernel.step(&kernel_ctx, &selection, &mut output)?;
            assert_eq!(
                output.len(),
                selection.true_count(),
                "Kernel did not write expected number of elements"
            );

            // Now we un-split the output vector back onto its full size.
            // output.unsplit(remaining_output);
            output_chunks.push(output);
            output = remaining_output;
        }

        // Combine all output chunks back into the output vector.
        for chunk in output_chunks {
            output.unsplit(chunk);
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
