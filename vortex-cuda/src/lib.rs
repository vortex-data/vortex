// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA support for Vortex arrays.

mod device_buffer;
pub mod executor;
mod for_;
mod kernel;
mod session;

use std::process::Command;

pub use device_buffer::CudaBufferExt;
pub use device_buffer::CudaDeviceBuffer;
pub use executor::CudaExecutionCtx;
pub use executor::CudaKernelEvents;
use for_::ForExecutor;
pub use session::CudaSession;

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
    session.register_kernel("fastlanes.for".into(), &ForExecutor);
    // TODO(0ax1): Register additional executors
}
