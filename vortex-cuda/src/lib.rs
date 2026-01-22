// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA support for Vortex arrays.

mod device_buffer;
pub mod executor;
mod kernel;
mod session;

use std::process::Command;

pub use device_buffer::CudaBufferExt;
pub use device_buffer::CudaDeviceBuffer;
pub use executor::CudaExecutionCtx;
pub use executor::CudaKernelEvents;
use kernel::DictExecutor;
use kernel::FoRExecutor;
pub use kernel::launch_cuda_kernel_impl;
pub use session::CudaSession;
use vortex_array::arrays::DictVTable;
use vortex_fastlanes::FoRVTable;

/// Check if the NVIDIA CUDA Compiler is available.
pub fn has_nvcc() -> bool {
    Command::new("nvcc")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

/// Registers CUDA kernels.
pub fn initialize_cuda(session: &CudaSession) {
    tracing::info!("Registering CUDA kernels");
    session.register_kernel(FoRVTable::ID, &FoRExecutor);
    session.register_kernel(DictVTable::ID, &DictExecutor);
}
