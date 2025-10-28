// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::GpuArray;

pub trait GPUTask {
    // Must call `launch_task` once
    fn launch_task(&mut self) -> VortexResult<()>;

    // Must call this after launch_task
    fn result(&mut self) -> VortexResult<GpuArray>;
}
