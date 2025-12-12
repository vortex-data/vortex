// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Formatter;

use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_vector::Vector;

use crate::kernel::Kernel;
use crate::kernel::KernelRef;
use crate::kernel::PushDownResult;

/// Create a kernel from a closure.
pub fn kernel<F: FnOnce() -> VortexResult<Vector> + Send + 'static>(closure: F) -> KernelRef {
    Box::new(ClosureKernel { closure })
}

pub struct ClosureKernel<F> {
    closure: F,
}

impl<F> Debug for ClosureKernel<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ClosureKernel")
    }
}

impl<F: FnOnce() -> VortexResult<Vector> + Send + 'static> Kernel for ClosureKernel<F> {
    fn execute(self: Box<Self>) -> VortexResult<Vector> {
        (self.closure)()
    }

    fn push_down_filter(self: Box<Self>, _selection: &Mask) -> VortexResult<PushDownResult> {
        Ok(PushDownResult::NotPushed(self))
    }
}
