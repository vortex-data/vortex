// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_vector::{Vector, VectorMutOps, VectorOps};

use crate::pipeline::{BitView, Kernel, KernelCtx, N};

/// A kernel that feeds a batch vector into the pipeline in chunks of size `N` with zero-copy.
pub(super) struct InputKernel {
    // The batch vector to be fed into the pipeline.
    input: Vector,
}

impl InputKernel {
    /// Create a new input kernel with the given batch vector.
    pub(super) fn new(input: Vector) -> Self {
        Self { input }
    }
}

impl Kernel for InputKernel {
    fn step(
        &mut self,
        _ctx: &KernelCtx,
        selection: &BitView,
        _out: Vector,
    ) -> VortexResult<Vector> {
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
    use vortex_buffer::{BitBuffer, bitbuffer, buffer};
    use vortex_dtype::PTypeDowncastExt;
    use vortex_mask::Mask;

    use crate::arrays::{BoolArray, PrimitiveArray};
    use crate::pipeline::driver::PipelineDriver;
    use crate::validity::Validity;
    use crate::{Array, ArrayOperator, IntoArray};

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

    /// Ensures that we can feed an input into a pipeline with zero-copy.
    /// This can require careful book keeping to make sure we don't hold references to arrays
    /// around longer than necessary.
    #[test]
    fn test_pipeline_input_zero_copy() {
        let elements = buffer![123u32; 8000];
        let elements_ptr = elements.as_ptr();
        let validity = BitBuffer::from_iter((0..8000).map(|i| i % 2 == 0));
        let validity_ptr = validity.inner().as_ptr();

        let array = PrimitiveArray::new(
            elements,
            Validity::Array(BoolArray::from(validity).into_array()),
        )
        .into_array();
        assert!(
            array.as_pipelined().is_none(),
            "We're explicitly testing non-pipelined arrays to trigger the input case"
        );

        let selection = Mask::new_true(array.len());
        let vector = PipelineDriver::new(array)
            .execute(&selection)
            .unwrap()
            .into_primitive()
            .downcast::<u32>();

        let (vector_elements, vector_validity) = vector.into_parts();
        let vector_validity = vector_validity.into_bit_buffer().into_inner();

        assert_eq!(vector_elements.as_ptr(), elements_ptr);
        assert_eq!(vector_validity.as_ptr(), validity_ptr);
    }
}
