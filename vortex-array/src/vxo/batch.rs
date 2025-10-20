// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::vxo::ArrayRef;
use async_trait::async_trait;
use futures::future::BoxFuture;
use vortex_error::VortexResult;

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

/// Type-alias for heap-allocated batch execution kernels.
pub type BatchKernelRef = Box<dyn BatchKernel>;

/// Trait for batch execution kernels.
#[async_trait]
pub trait BatchKernel {
    /// Execute the batch kernel and produce a canonicalized vector.
    ///
    /// If the kernel can return data zero-copy from its own state, then it should prefer to do so.
    /// If the kernel is producing _new_ data, it should write this data into the provided `out`
    /// vector.
    async fn execute(self: Box<Self>, out: VectorMut) -> VortexResult<Vector>;
}

type KernelFut = BoxFuture<'static, VortexResult<Vector>>;

/// Create a [`BatchKernelRef`] from a closure.
pub fn kernel<F, Fut>(f: F) -> BatchKernelRef
where
    F: FnOnce(VectorMut) -> Fut + Send + 'static,
    Fut: Future<Output = VortexResult<Vector>> + Send + 'static,
{
    Box::new(FnKernel(Box::new(move |out| Box::pin(f(out)))))
}

pub struct FnKernel(Box<dyn FnOnce(VectorMut) -> KernelFut + Send>);
#[async_trait]
impl BatchKernel for FnKernel {
    async fn execute(self: Box<Self>, out: VectorMut) -> VortexResult<Vector> {
        self.0(out).await
    }
}

/// Placeholder type for canonicalized vectors.
///
/// To be replaced by the Vectors PR.
pub struct Vector;

/// Placeholder type for mutable canonicalized vectors.
///
/// To be replaced by the Vectors PR.
pub struct VectorMut;
