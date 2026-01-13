// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA support for Vortex arrays.
//!
//! Provides support for:
//!
//! - Registering `CudaSupport` implementations for array encodings
//! - Intercepting array execution and routing them to GPU kernels if supported by an array
//! - Handling host/device memory transfers

pub mod executor;
pub mod kernel;
pub mod session;

pub use executor::CudaArrayExt;
pub use executor::CudaExecute;
pub use executor::CudaExecutor;
pub use kernel::KernelConfig;
pub use kernel::KernelRegistry;
pub use session::CudaSession;

/// Registers CUDA kernels for a given CUDA session.
pub fn initialize_cuda(session: &CudaSession) {
    tracing::info!("Initializing CUDA support");

    // Register CUDA kernel implementations for supported array encodings.
    //
    // session.register(BitPackedVTable::ID, &bitpacking::CUDA_SUPPORT);
    // session.register(RLEVTable::ID, &rle::CUDA_SUPPORT);

    tracing::debug!(
        registered_executors = session.executor_count(),
        "CUDA support initialized"
    );
}
