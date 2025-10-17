// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use cudarc::driver::CudaViewMut;
use vortex_array::Canonical;
use vortex_error::VortexResult;

pub trait GPUTask {
    // Must call `launch_task` once
    fn launch_task(&mut self) -> VortexResult<()>;

    // Must call this after launch_task
    fn export_result(&mut self) -> VortexResult<Canonical>;

    // Re can transmute as runtime
    fn output(&mut self) -> CudaViewMut<'_, u8>;

    fn len(&self) -> usize;
}
