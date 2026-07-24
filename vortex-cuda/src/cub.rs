// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA wrappers around CUB scan primitives.

use std::ffi::c_void;

use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use vortex::array::buffer::BufferHandle;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex_cub::onpair;
use vortex_cub::scan;
use vortex_cub::scan::cudaStream_t;

use crate::CudaBufferExt;
use crate::CudaExecutionCtx;

/// Regenerate the OnPair decode kernel's per-batch output offsets in one
/// fused sweep (see `cub/kernels/onpair.cu`): the per-batch decoded-size
/// reduction and the exclusive scan over the sizes run in a single kernel via
/// decoupled look-back. Returns `num_batches + 1` offsets; the last is the
/// total decoded byte count. A code outside the dictionary raises `status`
/// to 1; the caller must check the flag before trusting the offsets.
pub(crate) fn onpair_batch_offsets(
    codes: &BufferHandle,
    lens: &BufferHandle,
    dict_size: u32,
    num_tokens: usize,
    num_batches: usize,
    status: &mut CudaSlice<u32>,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<CudaSlice<u64>> {
    let num_batches_i64 = i64::try_from(num_batches)?;
    let temp_bytes = onpair::batch_offsets_temp_size(num_batches_i64)
        .map_err(|err| vortex_err!("CUB onpair_batch_offsets_temp_size failed: {err}"))?;

    let mut temp = ctx.device_alloc::<u8>(temp_bytes.max(1))?;
    let mut chunk_offsets = ctx.device_alloc::<u64>(num_batches + 1)?;
    let codes_ptr = codes.cuda_device_ptr()?;
    let lens_ptr = lens.cuda_device_ptr()?;
    let total_tokens = u64::try_from(num_tokens)?;
    let stream = ctx.stream();
    let stream_ptr = stream.cu_stream() as cudaStream_t;
    let (status_ptr, record_status) = status.device_ptr_mut(stream);
    let (offsets_ptr, record_offsets) = chunk_offsets.device_ptr_mut(stream);
    let (temp_ptr, record_temp) = temp.device_ptr_mut(stream);

    ctx.launch_external(num_tokens, || unsafe {
        onpair::batch_offsets(
            temp_ptr as *mut c_void,
            temp_bytes,
            codes_ptr as *const u16,
            lens_ptr as *const u8,
            dict_size,
            total_tokens,
            offsets_ptr as *mut u64,
            status_ptr as *mut u32,
            num_batches_i64,
            stream_ptr,
        )
        .map_err(|err| vortex_err!("CUB onpair_batch_offsets failed: {err}"))
    })?;
    drop((record_status, record_offsets, record_temp));

    Ok(chunk_offsets)
}

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
