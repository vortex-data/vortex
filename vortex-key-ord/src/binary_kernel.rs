// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `BinaryKernelSet<L, R>`: two-VTable analogue of
//! [`vortex_array::kernel::ParentKernelSet`] for binary stream ops.

use std::any::type_name;
use std::fmt::Debug;
use std::marker::PhantomData;

use vortex_error::VortexResult;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::VTable;

/// Typed kernel for an encoding pair `(L, R)`.
pub trait BinaryKernel<L: VTable, R: VTable>: Debug + Send + Sync + 'static {
    /// Predicate over the right input. Default: anything that downcasts
    /// to `ArrayView<'_, R>`.
    fn matches(_right: &ArrayRef) -> bool {
        true
    }

    fn execute(
        &self,
        left: ArrayView<'_, L>,
        right: ArrayView<'_, R>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Erased binary kernel: one indirect call per batch.
pub trait DynBinaryKernel<L: VTable, R: VTable>: Send + Sync {
    fn matches_right(&self, right: &ArrayRef) -> bool;
    fn execute_dyn(
        &self,
        left: ArrayView<'_, L>,
        right: ArrayView<'_, R>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// ZST bridge from [`BinaryKernel`] to [`DynBinaryKernel`]. Created by
/// [`BinaryKernelSet::lift`].
pub struct BinaryKernelAdapter<L, R, K> {
    kernel: K,
    _phantom: PhantomData<(L, R)>,
}

impl<L: VTable, R: VTable, K: BinaryKernel<L, R>> Debug for BinaryKernelAdapter<L, R, K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BinaryKernelAdapter")
            .field("left", &type_name::<L>())
            .field("right", &type_name::<R>())
            .field("kernel", &self.kernel)
            .finish()
    }
}

impl<L: VTable, R: VTable, K: BinaryKernel<L, R>> DynBinaryKernel<L, R>
    for BinaryKernelAdapter<L, R, K>
{
    fn matches_right(&self, right: &ArrayRef) -> bool {
        K::matches(right)
    }

    fn execute_dyn(
        &self,
        left: ArrayView<'_, L>,
        right: ArrayView<'_, R>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        self.kernel.execute(left, right, ctx)
    }
}

/// Collection of binary kernels for a fixed `(L, R)` encoding pair.
pub struct BinaryKernelSet<L: VTable, R: VTable> {
    kernels: &'static [&'static dyn DynBinaryKernel<L, R>],
}

impl<L: VTable, R: VTable> BinaryKernelSet<L, R> {
    pub const fn new(kernels: &'static [&'static dyn DynBinaryKernel<L, R>]) -> Self {
        Self { kernels }
    }

    /// Lift a zero-sized concrete kernel into a `&'static dyn` slot.
    pub const fn lift<K: BinaryKernel<L, R>>(
        kernel: &'static K,
    ) -> &'static dyn DynBinaryKernel<L, R> {
        const {
            assert!(
                !(size_of::<K>() != 0),
                "BinaryKernel must be zero-sized to be lifted"
            );
        }
        // SAFETY: `BinaryKernelAdapter<L, R, K>` is `#[repr(Rust)]`
        // but has the same layout as `K` because `_phantom` is
        // zero-sized and `K` is the const-asserted zero-sized first
        // field. The cast is the standard ParentKernelSet pattern.
        unsafe { &*(kernel as *const K as *const BinaryKernelAdapter<L, R, K>) }
    }

    /// First-match dispatch over the registered kernels.
    pub fn execute(
        &self,
        left: ArrayView<'_, L>,
        right_array: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        for kernel in self.kernels.iter() {
            if !kernel.matches_right(right_array) {
                continue;
            }
            let Some(right_view) = right_array.as_typed::<R>() else {
                continue;
            };
            if let Some(out) = kernel.execute_dyn(left, right_view, ctx)? {
                return Ok(Some(out));
            }
        }
        Ok(None)
    }
}
