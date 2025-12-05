// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;
use vortex_vector::Vector;
use vortex_vector::VectorOps;

use crate::kernel::Kernel;
use crate::kernel::KernelRef;

#[derive(Debug)]
pub struct ValidateKernel {
    inner: KernelRef,
    dtype: DType,
    row_count: usize,
}

impl ValidateKernel {
    pub fn new(inner: KernelRef, dtype: DType, row_count: usize) -> Self {
        Self {
            inner,
            dtype,
            row_count,
        }
    }
}

impl Kernel for ValidateKernel {
    fn execute(self: Box<Self>) -> VortexResult<Vector> {
        let vector = self.inner.execute()?;

        vortex_ensure!(
            vector.len() == self.row_count,
            "Row count mismatch. Expected {} rows but got {}",
            self.row_count,
            vector.len()
        );
        vortex_ensure!(
            vortex_vector::vector_matches_dtype(&vector, &self.dtype),
            "Data type mismatch",
        );

        Ok(vector)
    }

    fn cost_estimate(&self, selection: &Mask) -> f64 {
        self.inner.cost_estimate(selection)
    }

    fn push_down_filter(&self, selection: &Mask) -> VortexResult<Option<KernelRef>> {
        // TODO(ngates): should this wrap back up in the ValidateKernel?
        self.inner.push_down_filter(selection)
    }
}
