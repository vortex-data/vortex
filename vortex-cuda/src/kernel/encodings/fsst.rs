// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA executor for FSST decompression.

use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::CudaSlice;
use cudarc::driver::PushKernelArg;
use tracing::instrument;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::arrays::primitive::PrimitiveDataParts;
use vortex::array::arrays::varbin::VarBinArrayExt;
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

/// FSST kernel execution parameters prepared from a compressed array.
pub struct FsstKernelPrep {
    /// Compressed codes byte stream on device.
    pub codes_bytes: BufferHandle,
    /// Per-string compressed offsets into `codes_bytes` (I32).
    pub codes_offsets: BufferHandle,
    /// Symbol table (256 × u64).
    pub symbols: BufferHandle,
    /// Per-symbol byte length (256 × u8).
    pub symbol_lengths: BufferHandle,
    /// Per-string prefix-summed output offsets (I32).
    pub output_offsets: BufferHandle,
    /// Preallocated device output buffer (sized `total_size + 7`).
    pub device_output: CudaSlice<u8>,
    /// Number of strings in the array.
    pub num_strings: usize,
    /// Total decoded bytes (sum of uncompressed_lens).
    pub total_size: usize,
    /// Canonicalized uncompressed_lens PrimitiveArray, kept for the post-kernel `build_views` call.
    pub uncompressed_lens_array: PrimitiveArray,
}

/// Prepare FSST kernel parameters and device buffers for decompression.
///
/// Asserts the stored ptypes for `codes_offsets` and `uncompressed_lengths` are
/// both `I32`. File-loaded FSST arrays with narrower slot ptypes (U8 / U16 / U32)
/// are not yet supported and will return an error.
pub async fn fsst_kernel_prepare(
    fsst: FSSTArray,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<FsstKernelPrep> {
    let num_strings = fsst.len();

    let codes_offsets = fsst
        .codes()
        .offsets()
        .clone()
        .execute::<PrimitiveArray>(ctx.execution_ctx())?;
    let uncompressed_lens_array = fsst
        .uncompressed_lengths()
        .clone()
        .execute::<PrimitiveArray>(ctx.execution_ctx())?;

    if codes_offsets.ptype() != PType::I32 {
        vortex_bail!(
            "CUDA FSST decode: expected codes_offsets ptype I32, got {} (not yet implemented)",
            codes_offsets.ptype()
        );
    }
    if uncompressed_lens_array.ptype() != PType::I32 {
        vortex_bail!(
            "CUDA FSST decode: expected uncompressed_lens ptype I32, got {} (not yet implemented)",
            uncompressed_lens_array.ptype()
        );
    }

    // CPU-side prefix sum of uncompressed lengths → Vec<i32> output offsets.
    let lens_slice = uncompressed_lens_array.as_slice::<i32>();
    let mut output_offsets = Vec::with_capacity(num_strings + 1);
    let mut acc: i64 = 0;
    output_offsets.push(0i32);
    for &len in lens_slice {
        acc += i64::from(len);
        let off = i32::try_from(acc).map_err(|_| {
            vortex_err!(
                "FSST decoded output size exceeds MAX_BUFFER_LEN ({})",
                MAX_BUFFER_LEN
            )
        })?;
        output_offsets.push(off);
    }
    let total_size = usize::try_from(acc)
        .map_err(|_| vortex_err!("FSST output size overflow (unreachable — bounded above)"))?;

    let symbols_u64: Vec<u64> = fsst.symbols().iter().map(|s| s.to_u64()).collect();
    let symbol_lengths = fsst.symbol_lengths().clone();
    let codes_bytes_handle = fsst.codes_bytes_handle().clone();
    let PrimitiveDataParts {
        buffer: codes_offsets_handle,
        ..
    } = codes_offsets.into_data_parts();

    let symbols_fut = ctx.copy_to_device(symbols_u64)?;
    let symbol_lengths_fut = ctx.copy_to_device(symbol_lengths)?;
    let output_offsets_fut = ctx.copy_to_device(output_offsets)?;

    let codes_bytes = ctx.ensure_on_device(codes_bytes_handle).await?;
    let codes_offsets = ctx.ensure_on_device(codes_offsets_handle).await?;

    let symbols = symbols_fut.await?;
    let symbol_lengths = symbol_lengths_fut.await?;
    let output_offsets = output_offsets_fut.await?;

    let device_output = ctx.device_alloc::<u8>(total_size + 7)?;

    Ok(FsstKernelPrep {
        codes_bytes,
        codes_offsets,
        symbols,
        symbol_lengths,
        output_offsets,
        device_output,
        num_strings,
        total_size,
        uncompressed_lens_array,
    })
}

#[derive(Debug)]
pub(crate) struct FsstExecutor;

impl FsstExecutor {
    fn try_specialize(array: ArrayRef) -> Option<FSSTArray> {
        array.try_downcast::<FSST>().ok()
    }
}

#[async_trait]
impl CudaExecute for FsstExecutor {
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

        let prep = fsst_kernel_prepare(fsst, ctx).await?;

        // Scope the views so they're dropped before `prep.device_output` is moved.
        {
            let codes_bytes_view = prep.codes_bytes.cuda_view::<u8>()?;
            let codes_offsets_view = prep.codes_offsets.cuda_view::<i32>()?;
            let symbols_view = prep.symbols.cuda_view::<u64>()?;
            let symbol_lengths_view = prep.symbol_lengths.cuda_view::<u8>()?;
            let output_offsets_view = prep.output_offsets.cuda_view::<i32>()?;

            let cuda_function = ctx.load_function("fsst_decompress", &[])?;
            let num_strings_u64 = prep.num_strings as u64;
            ctx.launch_kernel(&cuda_function, prep.num_strings, |args| {
                args.arg(&codes_bytes_view)
                    .arg(&codes_offsets_view)
                    .arg(&symbols_view)
                    .arg(&symbol_lengths_view)
                    .arg(&output_offsets_view)
                    .arg(&prep.device_output)
                    .arg(&num_strings_u64);
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

// This test will FAIL until the `fsst_decode` kernel body in
// `vortex-cuda/kernels/src/fsst.cu` is filled in. The scaffolding stub writes
// zeros across each thread's output range, so the GPU-side canonical output is
// all-zero bytes and won't match the CPU reference.
#[cfg(test)]
mod tests {
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

    #[crate::test]
    async fn test_cuda_fsst_decompression_roundtrip() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let strings: Vec<&[u8]> = vec![
            b"the quick brown fox",
            b"jumps over the lazy dog",
            b"hello world",
            b"vortex fsst test string",
        ];
        let varbin = VarBinArray::from_iter(
            strings.iter().map(|s| Some(*s)),
            DType::Binary(Nullability::NonNullable),
        );
        let compressor = fsst_train_compressor(&varbin);
        let dtype = varbin.dtype().clone();
        let len = varbin.len();
        let fsst_array =
            fsst_compress(&varbin, len, &dtype, &compressor, cuda_ctx.execution_ctx()).into_array();

        let cpu_result = crate::canonicalize_cpu(fsst_array.clone())?;
        let gpu_result = FsstExecutor
            .execute(fsst_array, &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(cpu_result.into_array(), gpu_result);
        Ok(())
    }

    /// Exercises the multi-block grid-stride path. With 100K strings, the launch
    /// spans ~49 blocks of 2048 strings each, so every thread cycles through its
    /// 32-string share. A bug where threads only handle their first string (or
    /// the wrong stride) would leave most outputs zero and fail this assertion.
    #[crate::test]
    async fn test_cuda_fsst_decompression_roundtrip_large() -> VortexResult<()> {
        use vortex_fsst::test_utils::make_fsst_clickbench_urls;

        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let fsst_array = make_fsst_clickbench_urls(100_000, cuda_ctx.execution_ctx()).into_array();

        let cpu_result = crate::canonicalize_cpu(fsst_array.clone())?;
        let gpu_result = FsstExecutor
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
