// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_vector::{VectorMut, VectorMutOps};

use crate::pipeline::bit_view::BitView;
use crate::pipeline::{Kernel, KernelCtx, N};

/// A kernel that feeds a batch vector into the pipeline in chunks of size `N` with zero-copy.
pub(super) struct InputKernel {
    // The batch vector to be fed into the pipeline.
    batch: VectorMut,
}

impl InputKernel {
    /// Create a new input kernel with the given batch vector.
    pub(super) fn new(batch: VectorMut) -> Self {
        Self { batch }
    }
}

impl Kernel for InputKernel {
    fn step(
        &mut self,
        _ctx: &KernelCtx,
        _selection: &BitView,
        out: &mut VectorMut,
    ) -> VortexResult<()> {
        // We split off from our owned batch vector in chunks of size N, and then unsplit onto the
        // output vector. Both of these operations should be zero-copy.
        let mut split = self.batch.split_off(N.min(self.batch.len()));

        // Split-off leaves [0, at) in self.batch, and returns [at, ..)
        // So we swap the remainder back into self.batch for the next iteration
        std::mem::swap(&mut split, &mut self.batch);

        // If the output vector is the end of the pipeline, then each step we will be given back
        // the same output to append to, and unsplit will be zero-copy.
        // If the output vector is an intermediate vector, then it will be empty at the start of
        // each step, and unsplit will also be zero-copy.
        out.unsplit(split);

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_dtype::PTypeDowncastExt;
    use vortex_mask::Mask;

    use crate::pipeline::driver::PipelineDriver;
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
}
