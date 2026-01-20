// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA support for Vortex arrays.

pub mod executor;
mod device_buffer;
mod for_;
mod kernel;
pub mod pinned;
pub mod pinned_allocator;
mod session;

use std::process::Command;

pub use executor::CudaExecutionCtx;
pub use executor::CudaKernelEvents;
pub use pinned::PinnedByteBuffer;
pub use pinned::PinnedByteBufferPool;
pub use pinned::PooledPinnedBuffer;
pub use pinned_allocator::PinnedBufferAllocator;
pub use pinned_allocator::PinnedDeviceAllocator;
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
