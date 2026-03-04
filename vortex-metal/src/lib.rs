// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Metal support for Vortex arrays.
//!
//! This crate provides GPU-accelerated array execution on Apple Silicon
//! using the Metal framework.

mod device_buffer;
mod executor;
pub mod kernel;
mod library_loader;
mod session;

pub use device_buffer::MetalBufferExt;
pub use device_buffer::MetalDeviceBuffer;
pub use executor::CanonicalMetalExt;
pub use executor::MetalArrayExt;
pub use executor::MetalExecute;
pub use executor::MetalExecutionCtx;
use kernel::FoRExecutor;
pub use library_loader::MetalLibraryLoader;
pub use session::MetalSession;
pub use session::MetalSessionExt;
use tracing::info;
use vortex::encodings::fastlanes::FoRVTable;

/// Checks if Metal is available on the system.
pub fn metal_available() -> bool {
    #[cfg(target_os = "macos")]
    {
        use objc2_metal::MTLCreateSystemDefaultDevice;
        MTLCreateSystemDefaultDevice().is_some()
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

/// Registers Metal kernels.
pub fn initialize_metal(session: &MetalSession) {
    info!("Registering Metal kernels");
    session.register_kernel(FoRVTable::ID, &FoRExecutor);
}
