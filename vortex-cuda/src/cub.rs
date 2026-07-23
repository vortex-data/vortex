// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA wrappers around CUB scan primitives.

use std::ffi::c_void;

use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use vortex::array::buffer::BufferHandle;
use vortex::dtype::PType;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex_cub::scan;
use vortex_cub::scan::LengthType;
use vortex_cub::scan::cudaStream_t;

use crate::CudaBufferExt;
use crate::CudaExecutionCtx;

fn length_type(ptype: PType) -> VortexResult<LengthType> {
    Ok(match ptype {
        PType::U8 => LengthType::U8,
        PType::I8 => LengthType::I8,
        PType::U16 => LengthType::U16,
        PType::I16 => LengthType::I16,
        PType::U32 => LengthType::U32,
        PType::I32 => LengthType::I32,
        PType::U64 => LengthType::U64,
        PType::I64 => LengthType::I64,
        other => vortex_bail!("unsupported lengths ptype {other} for fused exclusive sum"),
    })
}

/// Fused widen + CUB `DeviceScan::ExclusiveSum` over device-resident per-row
/// lengths of any integer width.
///
/// One CUB dispatch scans the lengths through a widening transform iterator
/// and produces `num_rows + 1` u64 offsets whose last element is the total —
/// no separate widen kernel and no materialized u64 input. A negative length
/// raises `status` to 2 and contributes zero bytes; the caller must check the
/// flag before trusting the offsets.
pub(crate) fn exclusive_sum_lengths_u64(
    lengths: &BufferHandle,
    ptype: PType,
    num_rows: usize,
    status: &mut CudaSlice<u32>,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<CudaSlice<u64>> {
    let ty = length_type(ptype)?;
    let num_offsets = i64::try_from(num_rows + 1)?;
    let temp_bytes = scan::exclusive_sum_lengths_temp_size(ty, num_offsets)
        .map_err(|err| vortex_err!("CUB scan_exclusive_sum_lengths_temp_size failed: {err}"))?;

    let mut temp = ctx.device_alloc::<u8>(temp_bytes.max(1))?;
    let mut output = ctx.device_alloc::<u64>(num_rows + 1)?;
    let lengths_ptr = lengths.cuda_device_ptr()?;
    let stream = ctx.stream();
    let stream_ptr = stream.cu_stream() as cudaStream_t;
    let (status_ptr, record_status) = status.device_ptr_mut(stream);
    let (output_ptr, record_output) = output.device_ptr_mut(stream);
    let (temp_ptr, record_temp) = temp.device_ptr_mut(stream);

    ctx.launch_external(num_rows + 1, || unsafe {
        scan::exclusive_sum_lengths(
            ty,
            temp_ptr as *mut c_void,
            temp_bytes,
            lengths_ptr as *const c_void,
            output_ptr as *mut u64,
            status_ptr as *mut u32,
            num_offsets,
            stream_ptr,
        )
        .map_err(|err| vortex_err!("CUB scan_exclusive_sum_lengths failed: {err}"))
    })?;
    drop((record_status, record_output, record_temp));

    Ok(output)
}

/// CUB `DeviceScan::ExclusiveSum` over device-resident `u64` values.
///
/// Runs through the `i64` CUB instantiation: callers pass non-negative counts
/// whose prefix sums stay below `i64::MAX`, where two's complement `i64` and
/// `u64` addition produce identical bit patterns.
pub(crate) fn exclusive_sum_u64(
    input: &CudaSlice<u64>,
    len: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<CudaSlice<u64>> {
    let len_i64 = i64::try_from(len)?;
    let temp_bytes = scan::exclusive_sum_i64_temp_size(len_i64)
        .map_err(|err| vortex_err!("CUB scan_exclusive_sum_i64_temp_size failed: {err}"))?;

    let mut temp = ctx.device_alloc::<u8>(temp_bytes.max(1))?;
    let mut output = ctx.device_alloc::<u64>(len)?;
    let stream = ctx.stream();
    let stream_ptr = stream.cu_stream() as cudaStream_t;
    let (input_ptr, record_input) = input.device_ptr(stream);
    let (output_ptr, record_output) = output.device_ptr_mut(stream);
    let (temp_ptr, record_temp) = temp.device_ptr_mut(stream);

    ctx.launch_external(len, || unsafe {
        scan::exclusive_sum_i64(
            temp_ptr as *mut c_void,
            temp_bytes,
            input_ptr as *const i64,
            output_ptr as *mut i64,
            len_i64,
            stream_ptr,
        )
        .map_err(|err| vortex_err!("CUB scan_exclusive_sum_i64 failed: {err}"))
    })?;
    drop((record_input, record_output, record_temp));

    Ok(output)
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
