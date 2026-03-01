// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA kernel launch wrapper for RGB→NV12 color space conversion.

use std::sync::Arc;

use cudarc::driver::CudaFunction;
use cudarc::driver::CudaStream;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::driver::sys::CUdeviceptr;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex_cuda::CudaSessionExt;

/// Loads the `rgb_to_nv12` kernel function.
pub fn load_rgb_to_nv12_kernel(
    session: &vortex::session::VortexSession,
) -> VortexResult<CudaFunction> {
    session
        .cuda_session()
        .load_function_with_suffixes("rgb_to_nv12", &[])
}

/// Launches the RGB→NV12 conversion kernel on the GPU.
///
/// # Arguments
///
/// * `stream` - CUDA stream to launch on
/// * `func` - The loaded `rgb_to_nv12` kernel function
/// * `r_ptr` - Device pointer to R plane (width * height bytes)
/// * `g_ptr` - Device pointer to G plane (width * height bytes)
/// * `b_ptr` - Device pointer to B plane (width * height bytes)
/// * `nv12_ptr` - Device pointer to NV12 output (width * height * 3/2 bytes)
/// * `width` - Frame width in pixels
/// * `height` - Frame height in pixels
#[allow(clippy::too_many_arguments)]
pub fn rgb_to_nv12_launch(
    stream: &Arc<CudaStream>,
    func: &CudaFunction,
    r_ptr: CUdeviceptr,
    g_ptr: CUdeviceptr,
    b_ptr: CUdeviceptr,
    nv12_ptr: CUdeviceptr,
    width: u32,
    height: u32,
) -> VortexResult<()> {
    let block_x = 16u32;
    let block_y = 16u32;
    let grid_x = width.div_ceil(block_x);
    let grid_y = height.div_ceil(block_y);

    let config = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (block_x, block_y, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = stream.launch_builder(func);
    builder.arg(&r_ptr);
    builder.arg(&g_ptr);
    builder.arg(&b_ptr);
    builder.arg(&nv12_ptr);
    builder.arg(&width);
    builder.arg(&height);

    // SAFETY: All device pointers are valid and properly sized.
    unsafe {
        builder
            .launch(config)
            .map_err(|e| vortex_err!("Failed to launch rgb_to_nv12 kernel: {e}"))?;
    }

    Ok(())
}
