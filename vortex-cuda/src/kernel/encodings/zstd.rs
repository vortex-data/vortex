// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA executor for ZSTD decompression using nvcomp.

use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use futures::future::try_join_all;
use tracing::debug;
use tracing::instrument;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::arrays::varbinview::BinaryView;
use vortex::array::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use vortex::array::buffer::BufferHandle;
use vortex::array::buffer::DeviceBuffer;
use vortex::array::vtable::child_to_validity;
use vortex::buffer::Alignment;
use vortex::buffer::Buffer;
use vortex::buffer::ByteBuffer;
use vortex::dtype::DType;
use vortex::encodings::zstd::Zstd;
use vortex::encodings::zstd::ZstdArray;
use vortex::encodings::zstd::ZstdDataParts;
use vortex::encodings::zstd::ZstdMetadata;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::mask::AllOr;
use vortex_nvcomp::sys::nvcompStatus_t;
use vortex_nvcomp::zstd as nvcomp_zstd;

use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;

/// ZSTD kernel execution parameters prepared from a compressed array.
pub struct ZstdKernelPrep {
    /// Device pointer to array of pointers to compressed frames.
    pub frame_ptrs_ptr: u64,
    /// Device pointer to array of compressed frame sizes.
    pub frame_sizes_ptr: u64,
    /// Device pointer to array of expected uncompressed sizes.
    pub output_sizes_ptr: u64,
    /// Device pointer to array of pointers to output buffers.
    pub output_ptrs_ptr: u64,
    /// Number of frames to decompress.
    pub num_frames: usize,
    /// Size of temporary buffer for nvcomp.
    pub nvcomp_temp_buffer_size: usize,
    /// Device buffer for actual decompressed sizes (output).
    pub device_actual_sizes: CudaSlice<usize>,
    /// Device buffer for per-chunk status codes.
    pub device_statuses: CudaSlice<nvcompStatus_t>,
    /// Temporary workspace buffer for nvcomp.
    pub nvcomp_temp_buffer: CudaSlice<u8>,
    /// Contiguous output buffer for all decompressed data.
    pub device_output: CudaSlice<u8>,
    /// Handles to device memory for compressed frames (kept alive to prevent deallocation).
    pub device_frame_handles: Vec<BufferHandle>,
    /// Handle to device memory for frame pointers array.
    pub frame_ptrs_handle: BufferHandle,
    /// Handle to device memory for frame sizes array.
    pub frame_sizes_handle: BufferHandle,
    /// Handle to device memory for output sizes array.
    pub output_sizes_handle: BufferHandle,
    /// Handle to device memory for output pointers array.
    pub output_ptrs_handle: BufferHandle,
}

/// Prepares ZSTD kernel metadata and device buffers for decompression.
///
/// Returns the handles and metadata needed for kernel execution.
///
/// # Arguments
///
/// * `frames` - The compressed ZSTD frames (must not be empty)
/// * `metadata` - The compression metadata containing frame sizes
/// * `ctx` - The CUDA execution context
pub async fn zstd_kernel_prepare(
    frames: Vec<ByteBuffer>,
    metadata: &ZstdMetadata,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<ZstdKernelPrep> {
    // Gather frames' metadata.
    let frame_sizes: Vec<usize> = frames.iter().map(|f| f.len()).collect();
    let output_sizes: Vec<usize> = metadata
        .frames
        .iter()
        .map(|m| {
            usize::try_from(m.uncompressed_size)
                .vortex_expect("uncompressed size must fit in usize")
        })
        .collect();
    let output_size_total: usize = output_sizes.iter().sum();
    let output_size_max = output_sizes.iter().copied().max().unwrap_or(0);

    let num_frames = frames.len();

    // Temporary internal buffer size for nvCOMP ZSTD decompression.
    let nvcomp_temp_buffer_size =
        nvcomp_zstd::get_decompress_temp_size(num_frames, output_size_max, output_size_total)
            .map_err(|e| vortex_err!("nvcomp get_decompress_temp_size failed: {}", e))?;

    // Async copy frames to the device.
    let frame_futs = frames
        .into_iter()
        .map(|frame| ctx.copy_to_device(frame))
        .collect::<VortexResult<Vec<_>>>()?;

    let device_frame_handles = try_join_all(frame_futs).await?;

    // Allocate contiguous output buffer for all decompressed data.
    let device_output = ctx.device_alloc::<u8>(output_size_total)?;

    // Device pointers for all compressed frames.
    let frame_ptrs = device_frame_handles
        .iter()
        .map(|handle| handle.cuda_device_ptr())
        .collect::<VortexResult<Vec<_>>>()?;

    // Build output_ptrs from output base pointer + offsets.
    let output_ptrs = {
        // We only need the allocation address here to build pointer metadata.
        // The actual device write is tracked by `record_device_output` around
        // `decompress_async`, so this guard can be dropped immediately.
        let (base_ptr, _) = device_output.device_ptr(ctx.stream());
        output_sizes
            .iter()
            .scan(0u64, |offset, &size| {
                let ptr = base_ptr + *offset;
                *offset += size as u64;
                Some(ptr)
            })
            .collect::<Vec<_>>()
    };

    // Copy metadata asynchronously to the device.
    let (frame_ptrs_handle, frame_sizes_handle, output_sizes_handle, output_ptrs_handle) = futures::try_join!(
        ctx.copy_to_device(frame_ptrs)?,
        ctx.copy_to_device(frame_sizes)?,
        ctx.copy_to_device(output_sizes)?,
        ctx.copy_to_device(output_ptrs)?
    )?;

    // Allocate working buffers
    let device_actual_sizes: CudaSlice<usize> = ctx.device_alloc(num_frames)?;
    let device_statuses: CudaSlice<nvcompStatus_t> = ctx.device_alloc(num_frames)?;
    let nvcomp_temp_buffer: CudaSlice<u8> = ctx.device_alloc(nvcomp_temp_buffer_size)?;

    let frame_ptrs_ptr = frame_ptrs_handle.cuda_device_ptr()?;
    let frame_sizes_ptr = frame_sizes_handle.cuda_device_ptr()?;
    let output_sizes_ptr = output_sizes_handle.cuda_device_ptr()?;
    let output_ptrs_ptr = output_ptrs_handle.cuda_device_ptr()?;

    // Return device pointers and handles to keep device memory alive
    Ok(ZstdKernelPrep {
        frame_ptrs_ptr,
        frame_sizes_ptr,
        output_sizes_ptr,
        output_ptrs_ptr,
        num_frames,
        nvcomp_temp_buffer_size,
        device_actual_sizes,
        device_statuses,
        nvcomp_temp_buffer,
        device_output,
        device_frame_handles,
        frame_ptrs_handle,
        frame_sizes_handle,
        output_sizes_handle,
        output_ptrs_handle,
    })
}

/// CUDA executor for ZSTD decompression using nvCOMP.
#[derive(Debug)]
pub(crate) struct ZstdExecutor;

impl ZstdExecutor {
    fn try_specialize(array: ArrayRef) -> Option<ZstdArray> {
        array.try_downcast::<Zstd>().ok()
    }
}

#[async_trait]
impl CudaExecute for ZstdExecutor {
    #[instrument(level = "trace", skip_all, fields(executor = ?self))]
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let zstd = Self::try_specialize(array).ok_or_else(|| vortex_err!("Expected ZstdArray"))?;

        match zstd.dtype() {
            DType::Binary(_) | DType::Utf8(_) => decode_zstd(zstd, ctx).await,
            _other => {
                debug!(
                    dtype = %_other,
                    "Only Binary/Utf8 ZSTD arrays supported on GPU, falling back to CPU"
                );
                Zstd::decompress(&zstd, ctx.execution_ctx())?.to_canonical()
            }
        }
    }
}

async fn decode_zstd(array: ZstdArray, ctx: &mut CudaExecutionCtx) -> VortexResult<Canonical> {
    let dtype = array.dtype().clone();
    let validity = child_to_validity(&array.as_ref().slots()[0], dtype.nullability());
    let ZstdDataParts {
        frames,
        metadata,
        validity,
        n_rows,
        dictionary,
        slice_start,
        slice_stop,
    } = array.into_data().into_parts(validity);

    // nvCOMP doesn't support ZSTD dictionaries.
    if dictionary.is_some() {
        return Err(vortex_err!("ZSTD dictionary not supported on GPU"));
    }

    if frames.is_empty() {
        let result = unsafe {
            VarBinViewArray::new_unchecked(
                Buffer::<BinaryView>::empty(),
                Arc::from([]),
                dtype,
                validity,
            )
        };

        return Ok(Canonical::VarBinView(result));
    }

    let mut exec = zstd_kernel_prepare(frames, &metadata, ctx).await?;

    let stream = ctx.stream();
    let frame_views = exec
        .device_frame_handles
        .iter()
        .map(|handle| handle.cuda_view::<u8>())
        .collect::<VortexResult<Vec<_>>>()?;
    let mut frame_ptr_records = Vec::with_capacity(frame_views.len());
    for view in &frame_views {
        let (_frame_ptr, record_frame_ptr) = view.device_ptr(stream);
        frame_ptr_records.push(record_frame_ptr);
    }

    let frame_ptrs_view = exec.frame_ptrs_handle.cuda_view::<u64>()?;
    let frame_sizes_view = exec.frame_sizes_handle.cuda_view::<usize>()?;
    let output_sizes_view = exec.output_sizes_handle.cuda_view::<usize>()?;
    let output_ptrs_view = exec.output_ptrs_handle.cuda_view::<u64>()?;

    let (frame_ptrs_ptr, record_frame_ptrs) = frame_ptrs_view.device_ptr(stream);
    let (frame_sizes_ptr, record_frame_sizes) = frame_sizes_view.device_ptr(stream);
    let (output_sizes_ptr, record_output_sizes) = output_sizes_view.device_ptr(stream);
    let (output_ptrs_ptr, record_output_ptrs) = output_ptrs_view.device_ptr(stream);

    // Track writes to the output allocation at the actual enqueue point.
    // This guard intentionally outlives the pointer-metadata construction above.
    let (_device_output_ptr, record_device_output) = exec.device_output.device_ptr_mut(stream);
    let (device_actual_sizes_ptr, record_actual_sizes) =
        exec.device_actual_sizes.device_ptr_mut(stream);
    let (nvcomp_temp_buffer_ptr, record_temp) = exec.nvcomp_temp_buffer.device_ptr_mut(stream);
    let (device_statuses_ptr, record_statuses) = exec.device_statuses.device_ptr_mut(stream);

    ctx.launch_external(n_rows, || {
        // SAFETY: zstd_kernel_prepare makes sure to return valid kernel params.
        unsafe {
            nvcomp_zstd::decompress_async(
                frame_ptrs_ptr as _,
                frame_sizes_ptr as _,
                output_sizes_ptr as _,
                device_actual_sizes_ptr as _,
                exec.num_frames,
                nvcomp_temp_buffer_ptr as _,
                exec.nvcomp_temp_buffer_size,
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

    // Unconditionally copy back to the host as Zstd arrays are fully
    // self-contained. They neither have any parent or child encodings.
    //
    // TODO(0ax1): Don't copy back to host once VarBinView supports buffer handles.
    let host_buffer = CudaDeviceBuffer::new(exec.device_output)
        .copy_to_host(Alignment::new(1))?
        .await?;

    let slice_value_indices = validity
        .execute_mask(n_rows, ctx.execution_ctx())?
        .valid_counts_for_indices(&[slice_start, slice_stop]);
    let slice_value_idx_start = slice_value_indices[0];
    let slice_value_idx_stop = slice_value_indices[1];

    let sliced_validity = validity.slice(slice_start..slice_stop)?;

    match sliced_validity
        .execute_mask(slice_stop - slice_start, ctx.execution_ctx())?
        .indices()
    {
        AllOr::All => {
            let (buffers, all_views) =
                vortex::encodings::zstd::reconstruct_views(&host_buffer, MAX_BUFFER_LEN);
            let sliced_views = all_views.slice(slice_value_idx_start..slice_value_idx_stop);

            Ok(Canonical::VarBinView(unsafe {
                VarBinViewArray::new_unchecked(
                    sliced_views,
                    Arc::from(buffers),
                    dtype,
                    sliced_validity,
                )
            }))
        }
        _ => {
            vortex_bail!("CUDA ZSTD decompression does not yet support arrays with nulls")
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex::array::IntoArray;
    use vortex::array::arrays::VarBinViewArray;
    use vortex::array::assert_arrays_eq;
    use vortex::encodings::zstd::Zstd;
    use vortex::error::VortexResult;
    use vortex::session::VortexSession;

    use super::*;
    use crate::CanonicalCudaExt;
    use crate::executor::CudaArrayExt;
    use crate::session::CudaSession;

    #[crate::test]
    async fn test_cuda_zstd_decompression_utf8() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let strings = VarBinViewArray::from_iter_str([
            "hello",
            "world",
            "this is a longer string for testing",
            "foo",
            "bar",
            "baz",
        ]);

        let zstd_array = Zstd::from_var_bin_view(&strings, 3, 0)?;

        let cpu_result = Zstd::decompress(&zstd_array, cuda_ctx.execution_ctx())?.to_canonical()?;
        let gpu_result = ZstdExecutor
            .execute(zstd_array.into_array(), &mut cuda_ctx)
            .await?;

        assert_arrays_eq!(cpu_result.into_array(), gpu_result.into_array());
        Ok(())
    }

    #[crate::test]
    async fn test_cuda_zstd_decompression_multiple_frames() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let strings = VarBinViewArray::from_iter_str([
            "the quick brown fox jumps over the lazy dog",
            "hello world",
            "lorem ipsum dolor sit amet",
            "foo bar baz",
            "testing multiple frames",
            "another string here",
            "more data for testing",
            "short",
            "a bit longer string value",
            "final test string",
            "extra string one",
            "extra string two",
            "extra string three",
            "last one",
        ]);

        // Compress with ZSTD using values_per_frame=3 to create multiple frames.
        // 14 strings and 3 values per frame = ceil(14/3) = 5 frames.
        let zstd_array = Zstd::from_var_bin_view(&strings, 3, 3)?;

        let cpu_result = Zstd::decompress(&zstd_array, cuda_ctx.execution_ctx())?.to_canonical()?;
        let gpu_result = ZstdExecutor
            .execute(zstd_array.into_array(), &mut cuda_ctx)
            .await?;

        assert_arrays_eq!(cpu_result.into_array(), gpu_result.into_array());
        Ok(())
    }

    #[crate::test]
    async fn test_cuda_zstd_decompression_sliced() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let strings = VarBinViewArray::from_iter_str([
            "the quick brown fox jumps over the lazy dog",
            "hello world",
            "lorem ipsum dolor sit amet",
            "foo bar baz",
            "testing sliced arrays",
            "another string here",
            "more data for testing",
            "short",
            "a bit longer string value",
            "final test string",
        ]);

        let zstd_array = Zstd::from_var_bin_view(&strings, 3, 0)?;

        // Slice the array to get a subset (indices 2..7)
        let sliced_zstd = zstd_array.slice(2..7)?;

        let cpu_result = sliced_zstd.to_canonical()?;
        let gpu_result = ZstdExecutor
            .execute(sliced_zstd.clone(), &mut cuda_ctx)
            .await?;

        assert_arrays_eq!(cpu_result.into_array(), gpu_result.into_array());
        Ok(())
    }

    /// Zstd with nullable data — the GPU kernel does not yet support nulls,
    /// so `execute_cuda` should gracefully fall back to CPU and produce
    /// correct results instead of panicking.
    #[crate::test]
    async fn test_cuda_zstd_nullable_falls_back_to_cpu() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let strings = VarBinViewArray::from_iter_nullable_str([
            Some("hello"),
            None,
            Some("world"),
            None,
            Some("testing nullable zstd"),
            Some("another string"),
        ]);

        let zstd_array = Zstd::from_var_bin_view(&strings, 3, 0)?;

        let cpu_result = zstd_array.to_canonical()?.into_array();

        // execute_cuda should fall back to CPU and still produce the correct result.
        let gpu_result = zstd_array
            .into_array()
            .execute_cuda(&mut cuda_ctx)
            .await?
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(cpu_result, gpu_result);
        Ok(())
    }
}
