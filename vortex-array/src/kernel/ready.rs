// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_vector::Vector;

use crate::kernel::Kernel;
use crate::kernel::KernelRef;

/// Create a kernel that is already computed and ready.
pub fn ready(vector: Vector) -> KernelRef {
    Box::new(ReadyKernel::new(vector))
}

/// A kernel that is already computed and ready.
#[derive(Debug)]
pub struct ReadyKernel(Vector);

impl ReadyKernel {
    pub fn new(vector: Vector) -> Self {
        Self(vector)
    }
}

impl Kernel for ReadyKernel {
    fn execute(self: Box<Self>) -> VortexResult<Vector> {
        Ok(self.0)
    }

    fn cost_estimate(&self, selection: &Mask) -> f64 {
        _ = selection;
        0.0
    }
}
