// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_vector::Datum;
use vortex_vector::VectorMutOps;
use vortex_vector::VectorOps;

use crate::pipeline::BitView;
use crate::pipeline::Kernel;
use crate::pipeline::KernelCtx;
use crate::pipeline::N;

/// A kernel that feeds a batch vector into the pipeline in chunks of size `N` with zero-copy.
pub(super) struct InputKernel {
    // The batch vector to be fed into the pipeline.
    input: Datum,
}

impl InputKernel {
    /// Create a new input kernel with the given batch vector.
    pub(super) fn new(input: Datum) -> Self {
        Self { input }
    }
}

impl Kernel for InputKernel {
    fn step(
        &mut self,
        _ctx: &KernelCtx,
        selection: &BitView,
        _out: Datum,
    ) -> VortexResult<Datum> {
        let next_chunk_len = N.min(self.input.len());

        let next_chunk = self.input.slice(0..next_chunk_len);
        self.input = self.input.slice(next_chunk_len..);

        // We must return either `N` elements, or `true_count` elements. So if we have a final
        // chunk that has fewer than `N` elements, we need to either select out the true values,
        // or pad the chunk to `N` elements.
        if next_chunk.len() < N && selection.true_count() < next_chunk.len() {
            let mut next_chunk = next_chunk.into_mut();
            // TODO(ngates): append_zeros instead
            next_chunk.append_nulls(N - next_chunk.len());
            return Ok(next_chunk.freeze());
        }

        Ok(next_chunk)
    }
}

#[cfg(test)]
mod test {
    use vortex_buffer::bitbuffer;
    use vortex_buffer::buffer;
    use vortex_dtype::PTypeDowncastExt;
    use vortex_mask::Mask;

    use crate::Array;
    use crate::ArrayOperator;
    use crate::IntoArray;
    use crate::pipeline::driver::PipelineDriver;

    #[test]
    fn test_pipeline_input() {
        let array = buffer![123u32; 8000].into_array();
        assert!(
            array.as_pipelined().is_none(),
            "We're explicitly testing non-pipelined arrays"
        );

        let selection = Mask::new_true(array.len());
        let vector = PipelineDriver::new(array)
            .execute(&selection)
            .unwrap()
            .into_primitive()
            .downcast::<u32>();
        assert_eq!(vector.elements().as_ref(), &[123u32; 8000]);
    }

    #[test]
    fn test_pipeline_input_with_selection() {
        let array = buffer![0u32, 1, 2, 3, 4].into_array();
        assert!(
            array.as_pipelined().is_none(),
            "We're explicitly testing non-pipelined arrays"
        );

        let selection = Mask::from(bitbuffer![1 0 1 0 1]);
        let vector = PipelineDriver::new(array)
            .execute(&selection)
            .unwrap()
            .into_primitive()
            .downcast::<u32>();
        assert_eq!(vector.elements().as_ref(), &[0u32, 2, 4]);
    }
}
