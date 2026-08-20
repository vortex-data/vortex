// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Device construction of standard Arrow `i32` offsets.

use std::sync::Arc;

use cudarc::driver::DeviceRepr;
use cudarc::driver::PushKernelArg;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::primitive::PrimitiveDataParts;
use vortex::array::buffer::BufferHandle;
use vortex::array::match_each_integer_ptype;
use vortex::buffer::Buffer;
use vortex::dtype::NativePType;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;

use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::cub::exclusive_sum_i32;
use crate::executor::CudaExecutionCtx;

pub(crate) struct I32Offsets {
    pub(crate) buffer: BufferHandle,
    pub(crate) total: usize,
}

/// Build Arrow-compatible offsets from a canonical integer length array.
///
/// Length conversion, prefix sum, and overflow validation stay on the active CUDA stream. Only the
/// final offset and status word are copied to the host because callers need the total to size their
/// output allocation.
pub(crate) async fn i32_offsets_from_lengths(
    lengths: PrimitiveArray,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<I32Offsets> {
    let len = lengths.len();
    let ptype = lengths.ptype();
    let PrimitiveDataParts { buffer, .. } = lengths.into_data_parts();
    let lengths = ctx.ensure_on_device(buffer).await?;

    match_each_integer_ptype!(ptype, |L| {
        i32_offsets_from_lengths_typed::<L>(&lengths, len, ctx).await
    })
}

async fn i32_offsets_from_lengths_typed<L>(
    lengths: &BufferHandle,
    len: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<I32Offsets>
where
    L: NativePType + DeviceRepr + Send + Sync + 'static,
{
    let scan_len = len
        .checked_add(1)
        .ok_or_else(|| vortex_err!("Arrow offset count overflow"))?;
    let mut status = ctx.device_alloc::<u32>(1)?;
    ctx.stream()
        .memset_zeros(&mut status)
        .map_err(|err| vortex_err!("Failed to zero Arrow offset status buffer: {err}"))?;
    let lengths_view = lengths.cuda_view::<L>()?;
    let mut scan_input = ctx.device_alloc::<i32>(scan_len)?;
    let ptype = L::PTYPE.to_string();
    let scan_kernel =
        ctx.load_function_with_suffixes("arrow_offsets", &["from", "lengths", &ptype])?;
    let len_u64 = u64::try_from(len)?;

    ctx.launch_kernel(&scan_kernel, scan_len, |args| {
        args.arg(&lengths_view)
            .arg(&mut scan_input)
            .arg(&mut status)
            .arg(&len_u64);
    })?;

    let offsets = exclusive_sum_i32(&scan_input, scan_len, ctx)?;
    let scan_len_u64 = u64::try_from(scan_len)?;
    let validate_kernel = ctx.load_function_with_suffixes("arrow_offsets", &["validate"])?;
    ctx.launch_kernel(&validate_kernel, scan_len, |args| {
        args.arg(&offsets).arg(&mut status).arg(&scan_len_u64);
    })?;

    let offsets = BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(offsets)));
    let status = BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(status)));
    let status_copy = status.try_to_host()?;
    let total_copy = offsets.slice_typed::<i32>(len..scan_len).try_to_host()?;
    let (status_bytes, total_bytes) = futures::try_join!(status_copy, total_copy)?;
    match Buffer::<u32>::from_byte_buffer(status_bytes)[0] {
        0 => {}
        1 => vortex_bail!("cannot build Arrow offsets from a negative length"),
        2 => vortex_bail!("length sum exceeds Arrow i32 offset range"),
        status => vortex_bail!("unexpected Arrow offset status {status}"),
    }

    Ok(I32Offsets {
        buffer: offsets,
        total: usize::try_from(Buffer::<i32>::from_byte_buffer(total_bytes)[0])?,
    })
}
