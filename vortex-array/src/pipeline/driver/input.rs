// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_vector::{Vector, VectorMut, VectorMutOps, VectorOps};

use crate::pipeline::bit_view::BitView;
use crate::pipeline::{Kernel, KernelCtx, N};

/// A kernel that feeds a batch vector into the pipeline in chunks of size `N` with zero-copy.
pub(super) struct InputKernel {
    // The batch vector to be fed into the pipeline.
    batch: Vector,
    // The next offset into the batch vector.
    offset: usize,
}

impl InputKernel {
    /// Create a new input kernel with the given batch vector.
    pub(super) fn new(batch: Vector) -> Self {
        Self { batch, offset: 0 }
    }
}

impl Kernel for InputKernel {
    fn step(
        &mut self,
        _ctx: &KernelCtx,
        selection: &BitView,
        _out: VectorMut,
    ) -> VortexResult<Vector> {
        let chunk_size = N.min(self.batch.len());
        let chunk = self.batch.slice(self.offset..self.offset + chunk_size);

        // We must return either N rows, or selection.true_count() rows. For the final chunk of
        // the array, if we have fewer than N rows then we can either perform a filter to produce
        // the selected rows, or we can add zeros to pad the output up to N rows.
        // TODO(ngates): if BitView carried a length, we could avoid this copy entirely.
        Ok(if chunk.len() < N && selection.true_count() < N {
            // TODO(ngates): append_zeros, not nulls
            let padding = N - chunk.len();
            let mut chunk = chunk.into_mut();
            chunk.append_nulls(padding);
            chunk.freeze()
        } else {
            chunk
        })
    }
}

#[cfg(test)]
mod test {
    use vortex_buffer::{bitbuffer, buffer, BitBuffer};
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
