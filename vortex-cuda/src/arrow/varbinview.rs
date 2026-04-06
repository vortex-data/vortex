// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! We force conversion to VarBin from VarBinView. We parallelize
//! all the necessary string copying using some kernels.

use std::sync::Arc;

use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::arrays::varbinview::VarBinViewArrayParts;
use vortex::array::buffer::BufferHandle;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_err;

use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::CudaExecutionCtx;
use crate::arrow::check_validity_empty;

/// Parts of a binary array (VarBin).
///
/// We return them as buffer handles directly since we don't need the full VarBin array.
pub(crate) struct BinaryParts {
    pub(crate) offsets: BufferHandle,
    pub(crate) bytes: BufferHandle,
}

pub(crate) async fn copy_varbinview_to_varbin(
    array: VarBinViewArray,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<BinaryParts> {
    let len = array.len();
    let VarBinViewArrayParts {
        views,
        buffers,
        validity,
        ..
    } = array.into_data().into_parts();

    // TODO(aduffy): handle nulls
    check_validity_empty(&validity)?;

    // copy all buffers over to device.
    let views = ctx.ensure_on_device(views).await?;
    // before string copying, we must copy all string data buffers to the device.
    let mut device_buffers = vec![];
    for buffer in buffers.iter() {
        device_buffers.push(ctx.ensure_on_device(buffer.clone()).await?);
    }

    let buffer_ptrs = device_buffers
        .iter()
        .map(|b| b.cuda_device_ptr())
        .collect::<VortexResult<Vec<_>>>()?;

    // clone the buffer_ptrs to device so that we can pass it as an `uint8_t**` to the kernels
    let buffer_ptrs_device = ctx
        .stream()
        .clone_htod(&buffer_ptrs)
        .map_err(|e| vortex_err!("failed copying buffer_ptrs to device: {e}"))?;

    // single-threaded, launch the kernel for building the assets
    let compute_offsets = ctx.load_function("varbinview_compute_offsets", &[])?;

    // allocate the final offsets buffer.
    let offsets = ctx.device_alloc::<i32>(len + 1)?;
    let len_i64 = len as i64;

    let views_view = views.cuda_view::<i128>()?;
    let offsets_view = offsets.as_view();

    let last_offset_device = ctx.device_alloc::<i32>(1)?;
    let last_offset_device_view = last_offset_device.as_view();

    let mut kernel = ctx.launch_builder(&compute_offsets);
    kernel.arg(&views_view);
    kernel.arg(&len_i64);
    kernel.arg(&offsets_view);
    kernel.arg(&last_offset_device_view);

    let single_threaded_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    // Launch the kernel
    // SAFETY: we do not access any of the buffers we passed in until after the
    // kernel completes and we synchronize the stream.
    unsafe {
        kernel
            .launch(single_threaded_cfg)
            .map_err(|d| vortex_err!("compute_offsets kernel failure: {d}"))?;
    }

    // synchronize so the offset writes complete
    // now it is safe to read the memory again.
    ctx.stream()
        .synchronize()
        .map_err(|d| vortex_err!("synchronize stream failed: {d}"))?;

    let last_offset_host = ctx
        .stream()
        .clone_dtoh(&last_offset_device)
        .map_err(|e| vortex_err!("failed reading last_offset_device back to host: {e}"))?;

    // allocate a string buffer large enough to hold all strings.
    let data_buf = ctx.device_alloc::<u8>(last_offset_host[0] as usize)?;

    // now setup and launch the parallel string copy kernel
    let copy_strings = ctx.load_function("varbinview_copy_strings", &[])?;

    let buffer_ptrs_view = buffer_ptrs_device.as_view();
    let data_buf_view = data_buf.as_view();

    let mut kernel = ctx.launch_builder(&copy_strings);
    kernel.arg(&len_i64);
    kernel.arg(&views_view);
    kernel.arg(&buffer_ptrs_view);
    kernel.arg(&data_buf_view);
    kernel.arg(&offsets_view);

    // we do a fully parallel string copy. each thread is responsible for issuing copy
    // for a single string.
    let threads_per_blocks = 256u32;
    let n_blocks = len
        .div_ceil(threads_per_blocks as usize)
        .try_into()
        .vortex_expect("n_blocks should never overflow u32");
    let fully_parallel_cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (threads_per_blocks, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: downstream callers should synchronize the stream before accessing the values.
    unsafe {
        kernel
            .launch(fully_parallel_cfg)
            .map_err(|d| vortex_err!("copy_strings kernel failure: {d}"))?;
    }

    // synchronize?
    ctx.stream()
        .synchronize()
        .map_err(|e| vortex_err!("synchronize failure: {e}"))?;

    // now, offsets should contain the final offsets, and data_buf should contain all the
    // string data.
    let bytes_handle = BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(data_buf)));
    let offsets_handle = BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(offsets)));

    Ok(BinaryParts {
        bytes: bytes_handle,
        offsets: offsets_handle,
    })
}
