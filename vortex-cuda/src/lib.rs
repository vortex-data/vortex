// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA support for Vortex arrays.

mod device_buffer;
pub mod executor;
mod kernel;
mod session;
mod stream;

use std::process::Command;

pub use device_buffer::CudaBufferExt;
pub use device_buffer::CudaDeviceBuffer;
pub use executor::CudaExecutionCtx;
pub use executor::CudaKernelEvents;
use kernel::ALPExecutor;
use kernel::DictExecutor;
use kernel::FoRExecutor;
use kernel::ZigZagExecutor;
pub use kernel::launch_cuda_kernel_impl;
pub use session::CudaSession;
use vortex_alp::ALPVTable;
use vortex_array::arrays::DictVTable;
use vortex_decimal_byte_parts::DecimalBytePartsVTable;
use vortex_fastlanes::FoRVTable;
#[cfg(feature = "nvcomp")]
pub use vortex_nvcomp as nvcomp;
use vortex_zigzag::ZigZagVTable;

use crate::kernel::DecimalBytePartsExecutor;

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
    session.register_kernel(ALPVTable::ID, &ALPExecutor);
    session.register_kernel(FoRVTable::ID, &FoRExecutor);
    session.register_kernel(DictVTable::ID, &DictExecutor);
    session.register_kernel(ZigZagVTable::ID, &ZigZagExecutor);
    session.register_kernel(DecimalBytePartsVTable::ID, &DecimalBytePartsExecutor);
}
