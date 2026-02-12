// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! We force conversion to VarBin from VarBinView. We parallelize
//! all the necessary string copying using some kernels.

use std::ptr::addr_of_mut;
use std::sync::Arc;

use cudarc::driver::LaunchConfig;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::VarBinViewArrayParts;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::CudaDeviceBuffer;
use crate::CudaExecutionCtx;
use crate::arrow::check_validity_empty;
use crate::arrow::ensure_device_resident;
use crate::launcher::Function;
use crate::launcher::Kernel;
use crate::launcher::Launcher;

/// Parts of a binary array (VarBin).
///
/// We return them as buffer handles directly since we don't need the full VarBin array.
pub(crate) struct BinaryParts {
    pub(crate) offsets: BufferHandle,
    pub(crate) bytes: BufferHandle,
}

struct ComputeOffsetsKernel {
    function: Function,
}

impl Kernel for ComputeOffsetsKernel {
    type Args = (CudaDeviceBuffer, i64, CudaDeviceBuffer, CudaDeviceBuffer);

    fn new(function: Function) -> Self {
        Self { function }
    }

    unsafe fn launch(self, args: Self::Args, launcher: &Arc<dyn Launcher>) -> VortexResult<()> {
        let single_threaded_cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        let (views, mut len, offsets, last_offset) = args;

        unsafe {
            launcher.launch(
                self.function,
                single_threaded_cfg,
                vec![
                    views.device_ptr(),
                    std::ptr::addr_of_mut!(len).cast(),
                    offsets.device_ptr(),
                    last_offset.device_ptr(),
                ],
            )
        }
    }
}

struct CopyStrings {
    function: Function,
}

impl Kernel for CopyStrings {
    type Args = (
        i64,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
        CudaDeviceBuffer,
    );

    fn new(function: Function) -> Self {
        Self { function }
    }

    unsafe fn launch(self, args: Self::Args, launcher: &Arc<dyn Launcher>) -> VortexResult<()> {
        let (mut len, views, buffer_ptrs, data_buf, offsets) = args;

        // we do a fully parallel string copy. each thread is responsible for issuing copy
        // for a single string.
        let threads_per_blocks = 256u32;
        let n_blocks = (len as usize)
            .div_ceil(threads_per_blocks as usize)
            .try_into()
            .vortex_expect("n_blocks should never overflow u32");

        let fully_parallel_cfg = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (threads_per_blocks, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            launcher.launch(
                self.function,
                fully_parallel_cfg,
                vec![
                    addr_of_mut!(len).cast(),
                    views.device_ptr(),
                    buffer_ptrs.device_ptr(),
                    data_buf.device_ptr(),
                    offsets.device_ptr(),
                ],
            )
        }
    }
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
    } = array.into_parts();

    // TODO(aduffy): handle nulls
    check_validity_empty(&validity)?;

    // copy all buffers over to device.
    let views = ensure_device_resident(views, ctx).await?;
    // before string copying, we must copy all string data buffers to the device.
    let mut device_buffers = vec![];
    for buffer in buffers.iter() {
        device_buffers.push(ensure_device_resident(buffer.clone(), ctx).await?);
    }

    let buffer_ptrs = device_buffers
        .iter()
        .map(|b| b.device_ptr() as u64)
        .collect::<Vec<_>>();

    // clone the buffer_ptrs to device so that we can pass it as an `uint8_t**` to the kernels
    let buffer_ptrs_device = ctx
        .stream()
        .clone_htod(&buffer_ptrs)
        .map_err(|e| vortex_err!("failed copying buffer_ptrs to device: {e}"))?;
    let buffer_ptrs_device = CudaDeviceBuffer::new(buffer_ptrs_device);

    let compute_offsets_kernel =
        ctx.load_module("varbinview_compute_offsets")?
            .load::<ComputeOffsetsKernel>("varbinview_compute_offsets".to_string())?;

    // allocate the final offsets buffer.
    let offsets = ctx.device_alloc::<i32>(len + 1)?;
    let offsets = CudaDeviceBuffer::new(offsets);
    let len_i64 = len as i64;

    let last_offset_device = ctx.device_alloc::<i32>(1)?;
    let last_offset_device = CudaDeviceBuffer::new(last_offset_device);

    ctx.launch(
        compute_offsets_kernel,
        (
            views.clone(),
            len_i64,
            offsets.clone(),
            last_offset_device.clone(),
        ),
    )?;

    let last_offset_host =
        Buffer::<i32>::from_byte_buffer(ctx.copy_to_host(&last_offset_device)?.await?);

    // allocate a string buffer large enough to hold all strings.
    let data_buf = ctx.device_alloc::<u8>(last_offset_host[0] as usize)?;
    let data_buf = CudaDeviceBuffer::new(data_buf);

    // now setup and launch the parallel string copy kernel
    let copy_strings_kernel = ctx
        .load_module("varbincopy_strings")?
        .load::<CopyStrings>("varbincopy_strings".to_string())?;

    ctx.launch(
        copy_strings_kernel,
        (
            len_i64,
            views,
            buffer_ptrs_device,
            data_buf.clone(),
            offsets.clone(),
        ),
    )?;

    Ok(BinaryParts {
        bytes: data_buf.into(),
        offsets: offsets.into(),
    })
}
