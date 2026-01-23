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
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::arrays::BinaryView;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::buffer::DeviceBuffer;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_mask::AllOr;
use vortex_nvcomp::sys::nvcompStatus_t;
use vortex_nvcomp::zstd as nvcomp_zstd;
use vortex_zstd::ZstdArray;
use vortex_zstd::ZstdArrayParts;
use vortex_zstd::ZstdVTable;

use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;

/// CUDA executor for ZSTD decompression using nvCOMP.
#[derive(Debug)]
pub struct ZstdExecutor;

impl ZstdExecutor {
    fn try_specialize(array: ArrayRef) -> Option<ZstdArray> {
        array.as_opt::<ZstdVTable>().cloned()
    }
}

#[async_trait]
impl CudaExecute for ZstdExecutor {
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let zstd = Self::try_specialize(array).ok_or_else(|| vortex_err!("Expected ZstdArray"))?;

        match zstd.as_ref().dtype() {
            DType::Binary(_) | DType::Utf8(_) => decode_zstd(zstd, ctx).await,
            other => {
                tracing::debug!(
                    dtype = %other,
                    "Only Binary/Utf8 ZSTD arrays supported on GPU, falling back to CPU"
                );
                zstd.decompress()?.to_canonical()
            }
        }
    }
}

async fn decode_zstd(array: ZstdArray, ctx: &mut CudaExecutionCtx) -> VortexResult<Canonical> {
    let ZstdArrayParts {
        frames,
        metadata,
        dtype,
        validity,
        n_rows,
        dictionary,
        slice_start,
        slice_stop,
    } = array.into_parts();

    // nvCOMP doesn't support ZSTD dictionaries.
    if dictionary.is_some() {
        vortex_panic!("ZSTD dictionary not supported on GPU, falling back to CPU");
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
        .map(|handle| {
            handle
                .cuda_view::<u8>()
                .map(|view| view.device_ptr(ctx.stream()).0)
        })
        .collect::<VortexResult<Vec<_>>>()?;

    // Build output_ptrs from output base pointer + cumulative offsets.
    let output_ptrs = {
        let base_ptr = device_output.device_ptr(ctx.stream()).0;
        output_sizes
            .iter()
            .scan(0u64, |offset, &size| {
                let ptr = base_ptr + *offset;
                *offset += size as u64;
                Some(ptr)
            })
            .collect::<Vec<_>>()
    };

    let stream = ctx.stream();
    macro_rules! dev_ptr {
        ($slice:expr) => {
            $slice.device_ptr(stream).0 as _
        };
        (mut $slice:expr) => {
            $slice.device_ptr_mut(stream).0 as _
        };
    }

    // Copy metadata asynchronously to the device.
    let (frame_ptrs_handle, frame_sizes_handle, output_sizes_handle, output_ptrs_handle) = futures::try_join!(
        ctx.copy_to_device(frame_ptrs)?,
        ctx.copy_to_device(frame_sizes)?,
        ctx.copy_to_device(output_sizes)?,
        ctx.copy_to_device(output_ptrs)?
    )?;

    // Note that `device_alloc` returns immediately and does not wait for the allocation to complete.
    let mut device_actual_sizes: CudaSlice<usize> = ctx.device_alloc(num_frames)?;
    let mut device_statuses: CudaSlice<nvcompStatus_t> = ctx.device_alloc(num_frames)?;
    let mut nvcomp_temp_buffer: CudaSlice<u8> = ctx.device_alloc(nvcomp_temp_buffer_size)?;

    unsafe {
        nvcomp_zstd::decompress_async(
            // Device pointer to array of pointers to compressed chunks
            dev_ptr!(frame_ptrs_handle.cuda_view::<u64>()?),
            // Device pointer to array of compressed chunk sizes
            dev_ptr!(frame_sizes_handle.cuda_view::<usize>()?),
            // Device pointer to array of expected uncompressed sizes
            dev_ptr!(output_sizes_handle.cuda_view::<usize>()?),
            // Device pointer to array for actual uncompressed sizes (output)
            dev_ptr!(mut device_actual_sizes),
            // Number of frames to decompress
            num_frames,
            // Device pointer to temporary workspace buffer
            dev_ptr!(mut nvcomp_temp_buffer),
            // Size of temporary buffer in bytes
            nvcomp_temp_buffer_size,
            // Device pointer to array of pointers to output buffers
            dev_ptr!(output_ptrs_handle.cuda_view::<u64>()?),
            // Device pointer to array for per-chunk status codes
            dev_ptr!(mut device_statuses),
            // CUDA stream to execute on
            stream.cu_stream().cast(),
        )
        .map_err(|e| vortex_err!("nvcomp decompress_async failed: {}", e))?;
    }

    // Copy decompressed data back to host.
    let host_buffer = CudaDeviceBuffer::new(device_output)
        .copy_to_host(Alignment::new(1))?
        .await?;

    let slice_value_indices = validity
        .to_mask(n_rows)
        .valid_counts_for_indices(&[slice_start, slice_stop]);
    let slice_value_idx_start = slice_value_indices[0];
    let slice_value_idx_stop = slice_value_indices[1];

    let sliced_validity = validity.slice(slice_start..slice_stop)?;

    match sliced_validity.to_mask(slice_stop - slice_start).indices() {
        AllOr::All => {
            let all_views = vortex_zstd::reconstruct_views(&host_buffer);
            let sliced_views = all_views.slice(slice_value_idx_start..slice_value_idx_stop);

            Ok(Canonical::VarBinView(unsafe {
                VarBinViewArray::new_unchecked(
                    sliced_views,
                    Arc::from([host_buffer]),
                    dtype,
                    sliced_validity,
                )
            }))
        }
        _ => {
            unimplemented!("CUDA ZSTD decompression does not yet support arrays with nulls")
        }
    }
}

#[cfg(test)]
#[cfg(cuda_available)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;
    use vortex_zstd::ZstdArray;

    use super::*;
    use crate::session::CudaSession;

    #[tokio::test]
    async fn test_cuda_zstd_decompression_utf8() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let strings = VarBinViewArray::from_iter_str([
            "hello",
            "world",
            "this is a longer string for testing",
            "foo",
            "bar",
            "baz",
        ]);

        let zstd_array = ZstdArray::from_var_bin_view(&strings, 3, 0)?;

        let cpu_result = zstd_array.decompress()?.to_canonical()?;
        let gpu_result = ZstdExecutor
            .execute(zstd_array.into_array(), &mut cuda_ctx)
            .await?;

        assert_arrays_eq!(cpu_result.into_array(), gpu_result.into_array());
        Ok(())
    }

    #[tokio::test]
    async fn test_cuda_zstd_decompression_multiple_frames() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(VortexSession::empty())
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
        let zstd_array = ZstdArray::from_var_bin_view(&strings, 3, 3)?;

        let cpu_result = zstd_array.decompress()?.to_canonical()?;
        let gpu_result = ZstdExecutor
            .execute(zstd_array.into_array(), &mut cuda_ctx)
            .await?;

        assert_arrays_eq!(cpu_result.into_array(), gpu_result.into_array());
        Ok(())
    }

    #[tokio::test]
    async fn test_cuda_zstd_decompression_sliced() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(VortexSession::empty())
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

        let zstd_array = ZstdArray::from_var_bin_view(&strings, 3, 0)?;

        // Slice the array to get a subset (indices 2..7)
        let sliced_zstd = zstd_array.slice(2..7)?;

        let cpu_result = sliced_zstd.to_canonical()?;
        let gpu_result = ZstdExecutor
            .execute(sliced_zstd.clone(), &mut cuda_ctx)
            .await?;

        assert_arrays_eq!(cpu_result.into_array(), gpu_result.into_array());
        Ok(())
    }
}
