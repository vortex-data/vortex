// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::vxo::ArrayRef;
use async_trait::async_trait;
use vortex_error::VortexResult;

pub type BatchKernelRef = Box<dyn BatchKernel>;

/// Trait for batch execution kernels.
#[async_trait]
pub trait BatchKernel {
    /// Execute the batch kernel and produce a canonicalized vector.
    async fn execute(self: Box<Self>) -> VortexResult<Vector>;
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

/// Placeholder type for canonicalized vectors.
///
/// To be replaced by the Vectors PR.
pub struct Vector;
