// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Rust bindings to NVIDIA CUB library for GPU primitives.
//!
//! This crate provides bindings to CUB's DeviceSelect::Flagged operation,
//! which is used for GPU-accelerated filtering in Vortex.
//!
//! The library is compiled at build time and loaded at runtime via libloading.
//! This avoids link-time dependencies on CUDA.
//!
//! # Platform Support
//!
//! CUB is part of the CUDA Toolkit. This crate requires CUDA (nvcc) to be
//! available at build time.

use std::path::PathBuf;
use std::sync::OnceLock;

/// Raw FFI type definitions and dynamically-loaded function pointers from bindgen.
#[allow(non_camel_case_types, dead_code, clippy::all)]
pub mod sys;

mod error;
pub mod filter;
pub mod scan;

pub use error::CubError;

/// The loaded CUB library instance.
static CUB_LIB: OnceLock<Result<sys::CubLibrary, String>> = OnceLock::new();

fn load_cub() -> Result<sys::CubLibrary, String> {
    let lib_name = "libvortex_cub.so";
    let build_lib_dir = env!("OUT_DIR");
    let lib_path = PathBuf::from(build_lib_dir).join(lib_name);

    // SAFETY: The library at the path is a valid vortex_cub shared library
    // compiled during the build process.
    unsafe {
        sys::CubLibrary::new(&lib_path).map_err(|e| format!("Failed to load CUB library: {e}"))
    }
}

/// Gets a reference to the loaded CUB library.
///
/// The library is loaded lazily on first access. Returns an error if the
/// library cannot be found or loaded (e.g., CUDA was not available at build time).
pub fn cub_library() -> Result<&'static sys::CubLibrary, CubError> {
    CUB_LIB
        .get_or_init(load_cub)
        .as_ref()
        .map_err(|e| CubError::LibraryLoadError(e.clone()))
}

#[cfg(test)]
mod tests {
    use crate::filter;

    #[vortex_cuda_macros::test]
    fn test_filter_temp_size_u64() -> Result<(), crate::CubError> {
        let temp_bytes = filter::filter_get_temp_size_u64(1000)?;
        // CUB requires some temporary storage
        assert!(temp_bytes > 0);
        Ok(())
    }

    #[vortex_cuda_macros::test]
    fn test_filter_temp_size_f64() -> Result<(), crate::CubError> {
        let temp_bytes = filter::filter_get_temp_size_f64(10000)?;
        assert!(temp_bytes > 0);
        Ok(())
    }

    #[vortex_cuda_macros::test]
    fn test_filter_temp_size_zero_items() -> Result<(), crate::CubError> {
        // Just verify the call doesn't fail with zero items
        let _temp_bytes = filter::filter_get_temp_size_u8(0)?;
        Ok(())
    }
}
