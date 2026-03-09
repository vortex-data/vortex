// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA kernel launch wrapper for in-place posterization of uint8 buffers.

use std::sync::Arc;

use cudarc::driver::CudaFunction;
use cudarc::driver::CudaStream;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::driver::sys::CUdeviceptr;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex_cuda::CudaSessionExt;

/// Loads the `posterize` kernel function.
pub fn load_posterize_kernel(
    session: &vortex::session::VortexSession,
) -> VortexResult<CudaFunction> {
    session
        .cuda_session()
        .load_function_with_suffixes("posterize", &[])
}

/// Launches the posterize kernel on a GPU buffer in-place.
///
/// Quantizes each uint8 value to one of `levels` evenly spaced steps.
/// For example, `levels=4` maps to {0, 85, 170, 255}.
pub fn posterize_launch(
    stream: &Arc<CudaStream>,
    func: &CudaFunction,
    data_ptr: CUdeviceptr,
    len: u32,
    levels: u32,
) -> VortexResult<()> {
    let threads = 256u32;
    let blocks = len.div_ceil(threads);

    let config = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = stream.launch_builder(func);
    builder.arg(&data_ptr);
    builder.arg(&len);
    builder.arg(&levels);

    unsafe {
        builder
            .launch(config)
            .map_err(|e| vortex_err!("Failed to launch posterize kernel: {e}"))?;
    }

    Ok(())
}
