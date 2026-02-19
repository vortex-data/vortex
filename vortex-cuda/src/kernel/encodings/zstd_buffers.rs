// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA executor for zstd-buffers decompression using nvcomp.

use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use futures::future::BoxFuture;
use futures::future::try_join_all;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::buffer::BufferHandle;
use vortex_array::buffer::DeviceBuffer;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_cuda_macros::cuda_tests;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_nvcomp::sys;
use vortex_nvcomp::sys::nvcompStatus_t;
use vortex_nvcomp::zstd as nvcomp_zstd;
use vortex_zstd::ZstdBuffersArray;
use vortex_zstd::ZstdBuffersVTable;

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
            .try_into::<ZstdBuffersVTable>()
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
        let inner_array = array.build_inner(&[], ctx.session())?;
        return inner_array.execute_cuda(ctx).await;
    }

    let nvcomp_temp_buffer_size = nvcomp_zstd::get_decompress_temp_size(
        plan.num_frames(),
        plan.output_size_max(),
        plan.output_size_total(),
    )
    .map_err(|e| vortex_err!("nvcomp get_decompress_temp_size failed: {}", e))?;

    let device_frame_handles = move_frames_to_device(compressed_buffers, ctx).await?;
    let device_output = ctx.device_alloc::<u8>(plan.output_size_total())?;

    macro_rules! device_ptr {
        ($handle:expr, $type:ty) => {
            $handle.cuda_view::<$type>()?.device_ptr(ctx.stream()).0
        };
    }

    let frame_ptrs = device_frame_handles
        .iter()
        .map(|handle| Ok(device_ptr!(handle, u8)))
        .collect::<VortexResult<Vec<_>>>()?;

    let output_ptrs = {
        let base_ptr = device_output.device_ptr(ctx.stream()).0;
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

    let frame_ptrs_ptr = device_ptr!(frame_ptrs_handle, u64);
    let frame_sizes_ptr = device_ptr!(frame_sizes_handle, usize);
    let output_sizes_ptr = device_ptr!(output_sizes_handle, usize);
    let output_ptrs_ptr = device_ptr!(output_ptrs_handle, u64);

    let stream = ctx.stream();
    unsafe {
        nvcomp_zstd::decompress_async(
            frame_ptrs_ptr as _,
            frame_sizes_ptr as _,
            output_sizes_ptr as _,
            device_actual_sizes.device_ptr_mut(stream).0 as _,
            plan.num_frames(),
            nvcomp_temp_buffer.device_ptr_mut(stream).0 as _,
            nvcomp_temp_buffer_size,
            output_ptrs_ptr as _,
            device_statuses.device_ptr_mut(stream).0 as _,
            stream.cu_stream().cast(),
        )
        .map_err(|e| vortex_err!("nvcomp decompress_async failed: {}", e))?;
    }

    validate_decompress_results(&plan, device_actual_sizes, device_statuses).await?;

    let output_handle = BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(device_output)));
    let decompressed_buffers = plan.split_output_handle(&output_handle)?;

    let inner_array = array.build_inner(&decompressed_buffers, ctx.session())?;
    inner_array.execute_cuda(ctx).await
}

async fn move_frames_to_device(
    compressed_buffers: &[BufferHandle],
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Vec<BufferHandle>> {
    let move_futures = compressed_buffers
        .iter()
        .map(
            |frame| -> VortexResult<BoxFuture<'static, VortexResult<BufferHandle>>> {
                if frame.is_on_device() {
                    let frame = frame.clone();
                    Ok(Box::pin(async move { Ok(frame) }))
                } else {
                    ctx.move_to_device(frame.clone())
                }
            },
        )
        .collect::<VortexResult<Vec<_>>>()?;

    try_join_all(move_futures).await
}

// This performs D2H to retrieve the lengths and status arrays.
async fn validate_decompress_results(
    plan: &vortex_zstd::ZstdBuffersDecodePlan,
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

#[cuda_tests]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;
    use vortex_zstd::ZstdBuffersArray;

    use super::*;
    use crate::CanonicalCudaExt;
    use crate::session::CudaSession;

    #[tokio::test]
    async fn test_cuda_zstd_buffers_decompression_primitive() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let input = PrimitiveArray::from_iter(0i64..1024).into_array();
        let compressed = ZstdBuffersArray::compress(&input, 3)?;

        let cpu_result = compressed.clone().into_array().to_canonical()?;
        let gpu_result = ZstdBuffersExecutor
            .execute(compressed.into_array(), &mut cuda_ctx)
            .await?
            .into_host()
            .await?;

        assert_arrays_eq!(cpu_result.into_array(), gpu_result.into_array());
        Ok(())
    }

    #[tokio::test]
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
        let compressed = ZstdBuffersArray::compress(&input, 3)?;

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
