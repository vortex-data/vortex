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
        // Since we can perform the split-off in-place, we don't need to clone or slice the batch.
        let mut split = self.batch.split_off(N.min(self.batch.len()));

        // Split-off leaves [0, at) in self.batch, and returns [at, ..)
        // So we swap the remainder back into self.batch for the next iteration
        std::mem::swap(&mut split, &mut self.batch);

        // Set the output to the split portion.
        *out = split;

        Ok(())
    }
}
