// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA executor for zstd-buffers decompression using nvcomp.

use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::buffer::BufferHandle;
use vortex::array::buffer::DeviceBuffer;
use vortex::buffer::Alignment;
use vortex::buffer::Buffer;
use vortex::encodings::zstd::ZstdBuffers;
use vortex::encodings::zstd::ZstdBuffersArray;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex_nvcomp::sys;
use vortex_nvcomp::sys::nvcompStatus_t;
use vortex_nvcomp::zstd as nvcomp_zstd;

use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;

#[derive(Debug)]
pub(crate) struct ZstdBuffersExecutor;

#[async_trait]
impl CudaExecute for ZstdBuffersExecutor {
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let zstd_buffers = array
            .try_downcast::<ZstdBuffers>()
            .map_err(|_| vortex_err!("expected zstd buffers array"))?;
        decode_zstd_buffers(zstd_buffers, ctx).await
    }
}

async fn decode_zstd_buffers(
    array: ZstdBuffersArray,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical> {
    let plan = array.decode_plan()?;
    let compressed_buffers = plan.compressed_buffers();

    if compressed_buffers.is_empty() {
        let inner_array = ZstdBuffers::build_inner(&array, &[], ctx.session())?;
        return inner_array.execute_cuda(ctx).await;
    }

    let nvcomp_temp_buffer_size = nvcomp_zstd::get_decompress_temp_size(
        plan.num_frames(),
        plan.output_size_max(),
        plan.output_size_total(),
    )
    .map_err(|e| vortex_err!("nvcomp get_decompress_temp_size failed: {}", e))?;

    let device_frame_handles = move_frames_to_device(compressed_buffers, ctx).await?;
    let mut device_output = ctx.device_alloc::<u8>(plan.output_size_total())?;

    let frame_views = device_frame_handles
        .iter()
        .map(|handle| handle.cuda_view::<u8>())
        .collect::<VortexResult<Vec<_>>>()?;
    let mut frame_ptr_records = Vec::with_capacity(frame_views.len());
    let mut frame_ptrs = Vec::with_capacity(frame_views.len());
    for view in &frame_views {
        let (ptr, record_frame_ptr) = view.device_ptr(ctx.stream());
        frame_ptrs.push(ptr);
        frame_ptr_records.push(record_frame_ptr);
    }

    let output_ptrs = {
        // We only need the allocation address to assemble output pointer metadata.
        // The actual device write is tracked by `record_device_output` around
        // `decompress_async`, so this guard can be dropped immediately.
        let (base_ptr, _) = device_output.device_ptr(ctx.stream());
        plan.output_offsets()
            .iter()
            .map(|offset| base_ptr + *offset as u64)
            .collect::<Vec<_>>()
    };

    // We need to copy nvcomp args to the device; these come from the
    // host-resident decode plan.
    let (frame_ptrs_handle, frame_sizes_handle, output_sizes_handle, output_ptrs_handle) = futures::try_join!(
        ctx.copy_to_device(frame_ptrs)?,
        ctx.copy_to_device(plan.frame_sizes())?,
        ctx.copy_to_device(plan.output_sizes())?,
        ctx.copy_to_device(output_ptrs)?
    )?;

    let mut device_actual_sizes: CudaSlice<usize> = ctx.device_alloc(plan.num_frames())?;
    let mut device_statuses: CudaSlice<nvcompStatus_t> = ctx.device_alloc(plan.num_frames())?;
    let mut nvcomp_temp_buffer: CudaSlice<u8> = ctx.device_alloc(nvcomp_temp_buffer_size)?;
    let stream = ctx.stream();
    let frame_ptrs_view = frame_ptrs_handle.cuda_view::<u64>()?;
    let frame_sizes_view = frame_sizes_handle.cuda_view::<usize>()?;
    let output_sizes_view = output_sizes_handle.cuda_view::<usize>()?;
    let output_ptrs_view = output_ptrs_handle.cuda_view::<u64>()?;

    let (frame_ptrs_ptr, record_frame_ptrs) = frame_ptrs_view.device_ptr(stream);
    let (frame_sizes_ptr, record_frame_sizes) = frame_sizes_view.device_ptr(stream);
    let (output_sizes_ptr, record_output_sizes) = output_sizes_view.device_ptr(stream);
    let (output_ptrs_ptr, record_output_ptrs) = output_ptrs_view.device_ptr(stream);

    // Track writes to the output allocation at the actual enqueue point.
    // This guard intentionally outlives the pointer-metadata construction above.
    let (_device_output_ptr, record_device_output) = device_output.device_ptr_mut(stream);
    let (device_actual_sizes_ptr, record_actual_sizes) = device_actual_sizes.device_ptr_mut(stream);
    let (nvcomp_temp_buffer_ptr, record_temp) = nvcomp_temp_buffer.device_ptr_mut(stream);
    let (device_statuses_ptr, record_statuses) = device_statuses.device_ptr_mut(stream);

    ctx.launch_external(plan.output_size_total(), || {
        // SAFETY: Pointer and size parameters are derived from validated decode plan inputs.
        unsafe {
            nvcomp_zstd::decompress_async(
                frame_ptrs_ptr as _,
                frame_sizes_ptr as _,
                output_sizes_ptr as _,
                device_actual_sizes_ptr as _,
                plan.num_frames(),
                nvcomp_temp_buffer_ptr as _,
                nvcomp_temp_buffer_size,
                output_ptrs_ptr as _,
                device_statuses_ptr as _,
                stream.cu_stream().cast(),
            )
            .map_err(|e| vortex_err!("nvcomp decompress_async failed: {}", e))
        }
    })?;
    drop(frame_ptr_records);
    drop(frame_views);
    drop((
        record_frame_ptrs,
        record_frame_sizes,
        record_output_sizes,
        record_output_ptrs,
        record_device_output,
        record_actual_sizes,
        record_temp,
        record_statuses,
    ));

    validate_decompress_results(&plan, device_actual_sizes, device_statuses).await?;

    let output_handle = BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(device_output)));
    let decompressed_buffers = plan.split_output_handle(&output_handle)?;

    let inner_array = ZstdBuffers::build_inner(&array, &decompressed_buffers, ctx.session())?;
    inner_array.execute_cuda(ctx).await
}

async fn move_frames_to_device(
    compressed_buffers: &[BufferHandle],
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Vec<BufferHandle>> {
    let mut results = Vec::with_capacity(compressed_buffers.len());
    for frame in compressed_buffers {
        results.push(ctx.ensure_on_device(frame.clone()).await?);
    }
    Ok(results)
}

// This performs D2H to retrieve the lengths and status arrays.
async fn validate_decompress_results(
    plan: &vortex::encodings::zstd::ZstdBuffersDecodePlan,
    device_actual_sizes: CudaSlice<usize>,
    device_statuses: CudaSlice<nvcompStatus_t>,
) -> VortexResult<()> {
    let actual_sizes_host = CudaDeviceBuffer::new(device_actual_sizes)
        .copy_to_host(Alignment::of::<usize>())?
        .await?;
    let statuses_host = CudaDeviceBuffer::new(device_statuses)
        .copy_to_host(Alignment::of::<nvcompStatus_t>())?
        .await?;

    let actual_sizes = Buffer::<usize>::from_byte_buffer(actual_sizes_host);
    let statuses = Buffer::<nvcompStatus_t>::from_byte_buffer(statuses_host);
    let expected_sizes = plan.output_sizes();

    for (idx, ((&status, &actual), &expected)) in statuses
        .as_slice()
        .iter()
        .zip(actual_sizes.as_slice().iter())
        .zip(expected_sizes.iter())
        .enumerate()
    {
        if status != sys::nvcompStatus_t_nvcompSuccess {
            return Err(vortex_err!(
                "nvcomp chunk {} failed with status {}",
                idx,
                status
            ));
        }
        if actual != expected {
            return Err(vortex_err!(
                "nvcomp chunk {} decompressed size mismatch: expected {}, got {}",
                idx,
                expected,
                actual
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::VarBinViewArray;
    use vortex::array::assert_arrays_eq;
    use vortex::error::VortexExpect;
    use vortex::error::VortexResult;
    use vortex::session::VortexSession;

    use super::*;
    use crate::CanonicalCudaExt;
    use crate::session::CudaSession;

    #[crate::test]
    async fn test_cuda_zstd_buffers_decompression_primitive() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let input = PrimitiveArray::from_iter(0i64..1024).into_array();
        let compressed = ZstdBuffers::compress(&input, 3, &VortexSession::empty())?;

        let cpu_result = compressed.clone().into_array().to_canonical()?;
        let gpu_result = ZstdBuffersExecutor
            .execute(compressed.into_array(), &mut cuda_ctx)
            .await?
            .into_host()
            .await?;

        assert_arrays_eq!(cpu_result.into_array(), gpu_result.into_array());
        Ok(())
    }

    #[crate::test]
    async fn test_cuda_zstd_buffers_decompression_varbinview() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let input = VarBinViewArray::from_iter_str([
            "hello",
            "world",
            "this is a longer string for testing zstd_buffers",
            "foo",
            "bar",
            "baz",
        ])
        .into_array();
        let compressed = ZstdBuffers::compress(&input, 3, &VortexSession::empty())?;

        let cpu_result = compressed.clone().into_array().to_canonical()?;
        let gpu_result = ZstdBuffersExecutor
            .execute(compressed.into_array(), &mut cuda_ctx)
            .await?
            .into_host()
            .await?;

        assert_arrays_eq!(cpu_result.into_array(), gpu_result.into_array());
        Ok(())
    }
}
