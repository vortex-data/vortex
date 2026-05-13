// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA executor for FSST decompression.

use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::PushKernelArg;
use tracing::instrument;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::arrays::varbinview::BinaryView;
use vortex::array::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use vortex::array::arrays::varbinview::build_views::build_views;
use vortex::array::buffer::BufferHandle;
use vortex::array::buffer::DeviceBuffer;
use vortex::buffer::Alignment;
use vortex::buffer::Buffer;
use vortex::dtype::PType;
use vortex::encodings::fsst::FSST;
use vortex::encodings::fsst::FSSTArray;
use vortex::encodings::fsst::FSSTArrayExt;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;

use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;

/// Default target compressed-byte size per split used by [`FSSTExecutor`].
/// Each split is a code-boundary-aligned chunk of the compressed stream that
/// one GPU thread decodes. Smaller → more parallel units + more coalescing,
/// larger → lower per-thread overhead.
pub const DEFAULT_SPLIT_COMPRESSED_BYTES: usize = 32;

/// FSST kernel execution parameters prepared from a compressed array.
///
/// The kernel's parallelism unit is a **split**: a code-boundary-aligned chunk
/// of the compressed stream, ~`target_split_bytes` compressed bytes long. Each
/// GPU thread decodes one split. This is the GSST-style layout, computed at
/// prep time by a CPU pre-pass over `codes_bytes`.
pub struct FsstKernelPrep {
    /// Compressed codes byte stream on device.
    pub codes_bytes: BufferHandle,
    /// Per-split start offsets into `codes_bytes` (I32). Length = num_splits + 1.
    pub split_in_offsets: BufferHandle,
    /// Per-split start offsets into the output buffer (I32). Length = num_splits + 1.
    pub split_out_offsets: BufferHandle,
    /// Symbol table (256 × u64).
    pub symbols: BufferHandle,
    /// Per-symbol byte length (256 × u8).
    pub symbol_lengths: BufferHandle,
    /// Preallocated device output buffer.
    pub device_output: CudaSlice<u8>,
    /// Number of splits the kernel will decode in parallel.
    pub num_splits: usize,
    /// Total decoded bytes (sum of uncompressed_lens).
    pub total_size: usize,
    /// Canonicalized uncompressed_lens PrimitiveArray, kept for the post-kernel `build_views` call.
    pub uncompressed_lens_array: PrimitiveArray,
}

/// Prepare FSST kernel parameters and device buffers for decompression.
///
/// Runs a CPU pre-pass over `codes_bytes` tracking `in_pos` + `out_pos`,
/// emitting a split boundary at the next code boundary every time consumed
/// compressed bytes reach `target_split_bytes`. Splits never land mid-escape.
///
/// Asserts the stored ptype for `uncompressed_lengths` is `I32`. File-loaded
/// FSST arrays with narrower slot ptypes (U8 / U16 / U32) are not yet
/// supported and will return an error.
pub async fn fsst_kernel_prepare(
    fsst: FSSTArray,
    target_split_bytes: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<FsstKernelPrep> {
    if target_split_bytes == 0 {
        vortex_bail!("target_split_bytes must be > 0");
    }

    let uncompressed_lens_array = fsst
        .uncompressed_lengths()
        .clone()
        .execute::<PrimitiveArray>(ctx.execution_ctx())?;

    if uncompressed_lens_array.ptype() != PType::I32 {
        vortex_bail!(
            "CUDA FSST decode: expected uncompressed_lens ptype I32, got {} (not yet implemented)",
            uncompressed_lens_array.ptype()
        );
    }

    let codes_bytes_host = fsst.codes_bytes().as_slice();
    let symbol_lengths_buf = fsst.symbol_lengths();
    let symbol_lengths_host = symbol_lengths_buf.as_slice();

    let capacity = codes_bytes_host.len() / target_split_bytes + 2;
    let mut split_in_offsets = Vec::<i32>::with_capacity(capacity);
    let mut split_out_offsets = Vec::<i32>::with_capacity(capacity);
    split_in_offsets.push(0);
    split_out_offsets.push(0);

    let mut in_pos: usize = 0;
    let mut out_pos: i64 = 0;
    let mut split_start_in: usize = 0;

    while in_pos < codes_bytes_host.len() {
        let code = codes_bytes_host[in_pos];
        if code == 255 {
            in_pos += 2;
            out_pos += 1;
        } else {
            let len = i64::from(symbol_lengths_host[code as usize]);
            in_pos += 1;
            out_pos += len;
        }

        if in_pos - split_start_in >= target_split_bytes {
            split_in_offsets.push(
                i32::try_from(in_pos).map_err(|_| vortex_err!("codes_bytes exceeds i32::MAX"))?,
            );
            split_out_offsets.push(i32::try_from(out_pos).map_err(|_| {
                vortex_err!(
                    "FSST decoded output size exceeds MAX_BUFFER_LEN ({})",
                    MAX_BUFFER_LEN
                )
            })?);
            split_start_in = in_pos;
        }
    }
    if in_pos > split_start_in {
        split_in_offsets
            .push(i32::try_from(in_pos).map_err(|_| vortex_err!("codes_bytes exceeds i32::MAX"))?);
        split_out_offsets.push(i32::try_from(out_pos).map_err(|_| {
            vortex_err!(
                "FSST decoded output size exceeds MAX_BUFFER_LEN ({})",
                MAX_BUFFER_LEN
            )
        })?);
    }

    let num_splits = split_in_offsets.len() - 1;
    let total_size = usize::try_from(out_pos).map_err(|_| vortex_err!("output size overflow"))?;

    let symbols_u64: Vec<u64> = fsst.symbols().iter().map(|s| s.to_u64()).collect();
    let symbol_lengths = fsst.symbol_lengths().clone();
    let codes_bytes_handle = fsst.codes_bytes_handle().clone();

    let symbols_fut = ctx.copy_to_device(symbols_u64)?;
    let symbol_lengths_fut = ctx.copy_to_device(symbol_lengths)?;
    let split_in_fut = ctx.copy_to_device(split_in_offsets)?;
    let split_out_fut = ctx.copy_to_device(split_out_offsets)?;

    let codes_bytes = ctx.ensure_on_device(codes_bytes_handle).await?;

    let symbols = symbols_fut.await?;
    let symbol_lengths = symbol_lengths_fut.await?;
    let split_in_offsets = split_in_fut.await?;
    let split_out_offsets = split_out_fut.await?;

    let device_output = ctx.device_alloc::<u8>(total_size)?;

    Ok(FsstKernelPrep {
        codes_bytes,
        split_in_offsets,
        split_out_offsets,
        symbols,
        symbol_lengths,
        device_output,
        num_splits,
        total_size,
        uncompressed_lens_array,
    })
}

/// CUDA decoder for FSST.
#[derive(Debug)]
pub(crate) struct FSSTExecutor;

impl FSSTExecutor {
    fn try_specialize(array: ArrayRef) -> Option<FSSTArray> {
        array.try_downcast::<FSST>().ok()
    }
}

#[async_trait]
impl CudaExecute for FSSTExecutor {
    #[instrument(level = "trace", skip_all, fields(executor = ?self))]
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let fsst = Self::try_specialize(array).ok_or_else(|| vortex_err!("Expected FSSTArray"))?;

        let dtype = fsst.dtype().clone();
        let validity = fsst.codes().validity()?;

        if fsst.is_empty() {
            let empty = unsafe {
                VarBinViewArray::new_unchecked(
                    Buffer::<BinaryView>::empty(),
                    Arc::from([]),
                    dtype,
                    validity,
                )
            };
            return Ok(Canonical::VarBinView(empty));
        }

        let prep = fsst_kernel_prepare(fsst, DEFAULT_SPLIT_COMPRESSED_BYTES, ctx).await?;

        // Scope the views so they're dropped before `prep.device_output` is moved.
        {
            let codes_bytes_view = prep.codes_bytes.cuda_view::<u8>()?;
            let split_in_view = prep.split_in_offsets.cuda_view::<i32>()?;
            let split_out_view = prep.split_out_offsets.cuda_view::<i32>()?;
            let symbols_view = prep.symbols.cuda_view::<u64>()?;
            let symbol_lengths_view = prep.symbol_lengths.cuda_view::<u8>()?;

            // The kernel checks store alignment relative to the base via
            // `out_pos % N`, so the base must satisfy the widest store (u128 → 16).
            let (output_base_ptr, _) = prep.device_output.device_ptr(ctx.stream());
            assert_eq!(
                output_base_ptr % 16,
                0,
                "device_output base not 16-aligned: {output_base_ptr:#x}",
            );

            let cuda_function = ctx.load_function("fsst", &[])?;
            let num_splits_u64 = prep.num_splits as u64;
            ctx.launch_kernel(&cuda_function, prep.num_splits, |args| {
                args.arg(&codes_bytes_view)
                    .arg(&split_in_view)
                    .arg(&split_out_view)
                    .arg(&symbols_view)
                    .arg(&symbol_lengths_view)
                    .arg(&prep.device_output)
                    .arg(&num_splits_u64);
            })?;
        }

        let FsstKernelPrep {
            device_output,
            total_size,
            uncompressed_lens_array,
            ..
        } = prep;

        let host_bytes = CudaDeviceBuffer::new(device_output)
            .copy_to_host(Alignment::new(1))?
            .await?;
        let host_bytes = host_bytes.slice(0..total_size);

        let (buffers, views) = build_views(
            0,
            MAX_BUFFER_LEN,
            host_bytes.into_mut(),
            uncompressed_lens_array.as_slice::<i32>(),
        );

        Ok(Canonical::VarBinView(unsafe {
            VarBinViewArray::new_unchecked(views, Arc::from(buffers), dtype, validity)
        }))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex::array::IntoArray;
    use vortex::array::arrays::VarBinArray;
    use vortex::array::assert_arrays_eq;
    use vortex::dtype::DType;
    use vortex::dtype::Nullability;
    use vortex::encodings::fsst::fsst_compress;
    use vortex::encodings::fsst::fsst_train_compressor;
    use vortex::error::VortexExpect;
    use vortex::session::VortexSession;

    use super::*;
    use crate::CanonicalCudaExt;
    use crate::session::CudaSession;

    #[rstest]
    #[case::non_null(
        vec![Some(&b"the quick brown fox"[..]),
             Some(&b"jumps over the lazy dog"[..]),
             Some(&b"hello world"[..]),
             Some(&b"vortex fsst test string"[..])],
        Nullability::NonNullable,
    )]
    #[case::partial_nulls(
        vec![Some(&b"alpha"[..]), None, Some(&b"gamma"[..]), None, Some(&b"epsilon"[..])],
        Nullability::Nullable,
    )]
    #[case::all_nulls(
        vec![None, None, None, None, None],
        Nullability::Nullable,
    )]
    #[crate::test]
    async fn test_cuda_fsst_decompression_roundtrip(
        #[case] strings: Vec<Option<&'static [u8]>>,
        #[case] nullability: Nullability,
    ) -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let varbin = VarBinArray::from_iter(strings, DType::Binary(nullability));
        let compressor = fsst_train_compressor(&varbin);
        let dtype = varbin.dtype().clone();
        let len = varbin.len();
        let fsst_array =
            fsst_compress(&varbin, len, &dtype, &compressor, cuda_ctx.execution_ctx()).into_array();

        let cpu_result = crate::canonicalize_cpu(fsst_array.clone())?;
        let gpu_result = FSSTExecutor
            .execute(fsst_array, &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(cpu_result.into_array(), gpu_result);
        Ok(())
    }

    /// Exercises the multi-block grid-stride path on a larger dataset. With
    /// 100K strings and ~17 compressed bytes each, the launch produces
    /// thousands of splits spanning many blocks.
    #[crate::test]
    async fn test_cuda_fsst_decompression_roundtrip_large() -> VortexResult<()> {
        use vortex_fsst::test_utils::make_fsst_clickbench_urls;

        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let fsst_array = make_fsst_clickbench_urls(100_000, cuda_ctx.execution_ctx()).into_array();

        let cpu_result = crate::canonicalize_cpu(fsst_array.clone())?;
        let gpu_result = FSSTExecutor
            .execute(fsst_array, &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(cpu_result.into_array(), gpu_result);
        Ok(())
    }
}
