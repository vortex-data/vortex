// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_error::VortexResult;

use crate::wgpu::kernel::WgpuKernelRef;

/// Trait that provides execution capabilities for WebGPU backend.
pub trait WgpuBackend {
    fn execute(&self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<WgpuKernelRef>;

    fn execute_parent(
        &self,
        array: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<WgpuKernelRef>> {
        let _ = (array, parent, child_idx, ctx);
        Ok(None)
    }
}
