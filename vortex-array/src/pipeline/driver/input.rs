// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::{VortexExpect, VortexResult};
use vortex_vector::{VectorMut, VectorMutOps, VectorOps};

use crate::pipeline::{BitView, Kernel, KernelCtx, N};

/// A kernel that feeds a batch vector into the pipeline in chunks of size `N` with zero-copy.
pub(super) struct InputKernel {
    // The batch vector to be fed into the pipeline.
    batch: Option<VectorMut>,
}

impl InputKernel {
    /// Create a new input kernel with the given batch vector.
    pub(super) fn new(batch: VectorMut) -> Self {
        Self { batch: Some(batch) }
    }
}

impl Kernel for InputKernel {
    fn step(
        &mut self,
        _ctx: &KernelCtx,
        selection: &BitView,
        out: &mut VectorMut,
    ) -> VortexResult<()> {
        let mut batch = self
            .batch
            .take()
            .vortex_expect("Input kernel has already been exhausted");
        let remaining = batch.len();

        // The ideal thing to do here is to split off a chunk of size N from our owned batch vector,
        // and then unsplit it onto the output vector. This should be a zero-copy operation in both
        // cases, regardless of whether the output vector is the root output of the pipeline or an
        // intermediate vector that gets cleared on each iteration.
        //
        // The only case this doesn't work, is when we have fewer than N elements left in our batch
        // vector, _and_ the selection vector is not simply a dense prefix. In this case, we copy
        // the remaining elements into the output.
        if remaining < N && selection.true_count() < remaining {
            // TODO(ngates): this is slow. We should instead unsplit the vector, and then manually
            //  run a compaction over the vector.
            let immutable = batch.freeze();
            selection.iter_ones(|idx| {
                out.extend_from_vector(&immutable.slice(idx..idx + 1));
            });
            return Ok(());
        }

        // We split off from our owned batch vector in chunks of size N, and then unsplit onto the
        // output vector. Both of these operations should be zero-copy.
        let mut split = batch.split_off(N.min(remaining));

        // Split-off leaves [0, at) in self.batch, and returns [at, ..)
        // So we swap the remainder back into self.batch for the next iteration
        std::mem::swap(&mut split, &mut batch);

        // If the output vector is the end of the pipeline, then each step we will be given back
        // the same output to append to, and unsplit will be zero-copy.
        // If the output vector is an intermediate vector, then it will be empty at the start of
        // each step, and unsplit will also be zero-copy.
        out.unsplit(split);

        self.batch = Some(batch);

        Ok(())
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
