// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::future::BoxFuture;
use vortex_error::VortexResult;
use vortex_vector::Vector;

use crate::ArrayRef;

/// Type-alias for heap-allocated batch execution kernels.
pub type BatchKernel = BoxFuture<'static, VortexResult<Vector>>;

/// Context for binding batch execution kernels.
///
/// By binding child arrays through this context, we can perform common subtree elimination and
/// share canonicalized results across multiple kernels.
pub trait BindCtx {
    /// Bind the given array and optional selection to produce a batch kernel, possibly reusing
    /// previously bound results from this context.
    fn bind(&mut self, array: &ArrayRef, selection: Option<&ArrayRef>)
    -> VortexResult<BatchKernel>;
}
