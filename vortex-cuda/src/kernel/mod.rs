// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA kernel loading and management.

use std::path::Path;
use std::path::PathBuf;

use cudarc::driver::LaunchConfig;

mod arrays;
mod encodings;
mod filter;
mod patches;
mod slice;

pub use arrays::ConstantNumericExecutor;
pub use arrays::DictExecutor;
pub use arrays::SharedExecutor;
pub use encodings::*;
pub use filter::FilterExecutor;
pub use slice::SliceExecutor;

/// Default scalar launch configuration
pub(crate) fn scalar_launch_config(len: usize) -> LaunchConfig {
    // Kernel launch configuration constants.
    // Must match ELEMENTS_PER_THREAD in CUDA kernels (kernels/*.cu).
    const THREADS_PER_BLOCK: u32 = 64; // 2 warps
    const ELEMENTS_PER_THREAD: u32 = 32;
    const ELEMENTS_PER_BLOCK: usize = (THREADS_PER_BLOCK * ELEMENTS_PER_THREAD) as usize; // 2048

    let num_blocks =
        u32::try_from(len.div_ceil(ELEMENTS_PER_BLOCK)).expect("num_blocks cannot exceed u32::MAX");

    LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Returns the PTX file path for a given module name.
///
/// Checks for `VORTEX_CUDA_KERNELS_DIR` environment variable at runtime first,
/// falling back to the path baked in at compile time by build.rs.
///
/// # Arguments
///
/// * `module_name` - Name of the module
///
/// # Returns
///
/// The full path to the PTX file
pub(crate) fn ptx_path_for_module(module_name: &str) -> PathBuf {
    let kernels_dir = std::env::var("VORTEX_CUDA_KERNELS_DIR")
        .unwrap_or_else(|_| env!("VORTEX_CUDA_KERNELS_DIR").to_string());
    Path::new(&kernels_dir).join(format!("{}.ptx", module_name))
}
