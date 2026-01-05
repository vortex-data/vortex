// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_error::VortexResult;

use crate::wgpu::backend::WgpuBackend;
use crate::wgpu::kernel::WgpuKernelRef;

struct PrimitiveWgpuBackend {
    device: wgpu::Device,
}
impl WgpuBackend for PrimitiveWgpuBackend {
    fn execute(&self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<WgpuKernelRef> {
        todo!()
    }
}
