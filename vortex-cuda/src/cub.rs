// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA wrappers around CUB scan primitives.

use std::ffi::c_void;

use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex_cub::scan;
use vortex_cub::scan::cudaStream_t;

use crate::CudaExecutionCtx;

pub(crate) fn exclusive_sum_i32(
    input: &CudaSlice<i32>,
    len: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<CudaSlice<i32>> {
    let len_i64 = i64::try_from(len)?;
    let temp_bytes = scan::exclusive_sum_i32_temp_size(len_i64)
        .map_err(|err| vortex_err!("CUB scan_exclusive_sum_i32_temp_size failed: {err}"))?;

    let mut temp = ctx.device_alloc::<u8>(temp_bytes.max(1))?;
    let mut output = ctx.device_alloc::<i32>(len)?;
    let stream = ctx.stream();
    let stream_ptr = stream.cu_stream() as cudaStream_t;
    let (input_ptr, record_input) = input.device_ptr(stream);
    let (output_ptr, record_output) = output.device_ptr_mut(stream);
    let (temp_ptr, record_temp) = temp.device_ptr_mut(stream);

    ctx.launch_external(len, || unsafe {
        scan::exclusive_sum_i32(
            temp_ptr as *mut c_void,
            temp_bytes,
            input_ptr as *const i32,
            output_ptr as *mut i32,
            len_i64,
            stream_ptr,
        )
        .map_err(|err| vortex_err!("CUB scan_exclusive_sum_i32 failed: {err}"))
    })?;
    drop((record_input, record_output, record_temp));

    Ok(output)
}
