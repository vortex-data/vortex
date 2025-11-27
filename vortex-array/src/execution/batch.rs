// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_vector::Datum;

use crate::ArrayRef;

/// Type-alias for heap-allocated batch execution kernels.
pub type BatchKernelRef = Box<dyn BatchKernel>;

/// Trait for batch execution kernels that produce a vector result.
pub trait BatchKernel: 'static + Send {
    fn execute(self: Box<Self>) -> VortexResult<Datum>;
}

/// Adapter to create a batch kernel from a closure.
pub struct BatchKernelAdapter<F>(F);
impl<F: FnOnce() -> VortexResult<Datum> + Send + 'static> BatchKernel for BatchKernelAdapter<F> {
    fn execute(self: Box<Self>) -> VortexResult<Datum> {
        self.0()
    }
}

/// Create a batch execution kernel from the given closure.
#[inline(always)]
pub fn kernel<F: FnOnce() -> VortexResult<Datum> + Send + 'static>(f: F) -> BatchKernelRef {
    Box::new(BatchKernelAdapter(f))
}

/// Context for binding batch execution kernels.
///
/// By binding child arrays through this context, we can perform common subtree elimination and
/// share canonicalized results across multiple kernels.
pub trait BindCtx {
    /// Bind the given array and optional selection to produce a batch kernel, possibly reusing
    /// previously bound results from this context.
    fn bind(
        &mut self,
        array: &ArrayRef,
        selection: Option<&ArrayRef>,
    ) -> VortexResult<BatchKernelRef>;
}
