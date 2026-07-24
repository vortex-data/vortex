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

#[cfg(test)]
mod tests {
    use vortex::error::VortexExpect;

    use super::*;
    use crate::session::CudaSession;

    /// Upload synthetic codes and lengths, regenerate the chunk offsets, and
    /// read them back together with the status flag.
    async fn batch_offsets_roundtrip(
        codes: Vec<u16>,
        lens: Vec<u8>,
    ) -> VortexResult<(Vec<u64>, u32)> {
        let mut ctx = CudaSession::create_execution_ctx(&crate::cuda_session())?;
        let num_tokens = codes.len();
        let num_batches = num_tokens.div_ceil(128);
        let dict_size = u32::try_from(lens.len())?;

        let codes_dev = ctx.copy_to_device(codes)?.await?;
        let lens_dev = ctx.copy_to_device(lens)?.await?;
        let mut status = ctx.device_alloc::<u32>(1)?;
        ctx.stream()
            .memset_zeros(&mut status)
            .map_err(|e| vortex_err!("Failed to zero status flag: {e}"))?;

        let offsets = onpair_batch_offsets(
            &codes_dev,
            &lens_dev,
            dict_size,
            num_tokens,
            num_batches,
            &mut status,
            &mut ctx,
        )?;

        let offsets = ctx
            .stream()
            .clone_dtoh(&offsets)
            .map_err(|e| vortex_err!("Failed to copy offsets to host: {e}"))?;
        let status = ctx
            .stream()
            .clone_dtoh(&status)
            .map_err(|e| vortex_err!("Failed to copy status to host: {e}"))?;
        Ok((offsets, status[0]))
    }

    /// The exclusive prefix at 128-token boundaries, plus the trailing total.
    fn host_reference(codes: &[u16], lens: &[u8]) -> Vec<u64> {
        let mut expected = Vec::with_capacity(codes.len().div_ceil(128) + 1);
        expected.push(0u64);
        let mut acc = 0u64;
        for (i, &code) in codes.iter().enumerate() {
            acc += u64::from(lens[code as usize]);
            if (i + 1) % 128 == 0 {
                expected.push(acc);
            }
        }
        if !codes.len().is_multiple_of(128) {
            expected.push(acc);
        }
        expected
    }

    /// A single partial batch: one tile, no look-back.
    #[crate::test]
    async fn test_onpair_batch_offsets_single_batch() -> VortexResult<()> {
        let lens: Vec<u8> = (1..=16).collect();
        let codes: Vec<u16> = (0..100u16).map(|i| i % 16).collect();
        let expected = host_reference(&codes, &lens);

        let (offsets, status) = batch_offsets_roundtrip(codes, lens).await?;
        assert_eq!(status, 0);
        assert_eq!(offsets, expected);
        Ok(())
    }

    /// Many look-back tiles with a ragged tail batch; the offsets must match
    /// a host prefix sum sampled at 128-token boundaries. Regression test for
    /// the look-back prefix being defined only in lane 0.
    #[crate::test]
    async fn test_onpair_batch_offsets_multi_tile() -> VortexResult<()> {
        let lens: Vec<u8> = (1..=16u8).cycle().take(300).collect();
        let codes: Vec<u16> = (0..2000u32 * 128 - 57)
            .map(|i| u16::try_from(i * 31 % 300).vortex_expect("bounded by dictionary size"))
            .collect();
        let expected = host_reference(&codes, &lens);

        let (offsets, status) = batch_offsets_roundtrip(codes, lens).await?;
        assert_eq!(status, 0);
        assert_eq!(offsets, expected);
        Ok(())
    }

    /// A code outside the dictionary raises the status flag and contributes
    /// zero bytes.
    #[crate::test]
    async fn test_onpair_batch_offsets_flags_out_of_range_code() -> VortexResult<()> {
        let lens = vec![2u8; 4];
        let mut codes = vec![1u16; 200];
        codes[130] = 9;

        let (offsets, status) = batch_offsets_roundtrip(codes, lens).await?;
        assert_eq!(status, 1);
        assert_eq!(offsets, vec![0, 256, 256 + 71 * 2]);
        Ok(())
    }
}
