// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA executor for OnPair decompression.
//!
//! Decoding runs on the GPU over the flat token stream:
//!
//! 1. `onpair_batch_offsets` (in the CUB shim) regenerates the per-batch
//!    output offsets (`chunk_offsets`) the decode kernel positions its writes
//!    with: one fused sweep reduces every 128-token batch's decoded size and
//!    exclusive-scans the sizes in-kernel via decoupled look-back.
//! 2. `onpair_shmem_4tpt_split8read` gathers each token's bytes from the
//!    split dictionary layout and scatters them to the output byte stream.
//!
//! The total decoded size comes from the GPU offsets regeneration; the
//! lengths child is materialised on host once, and each output path derives
//! its row offsets from it — validating them against that total before they
//! index the decoded heap. The result is exposed either as a canonical
//! `VarBinView` (views built on-device by `onpair_build_views` from
//! host-prefix-summed offsets, or on host for heaps that exceed a single
//! backing buffer) or as Arrow-compatible i32 offsets plus values via
//! [`decode_onpair_varbin`], which builds the offsets on device with
//! [`i32_offsets_from_lengths`] — mirroring the FSST varbin path.

use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use tracing::instrument;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use vortex::array::arrays::varbinview::build_views::build_views;
use vortex::array::buffer::BufferHandle;
use vortex::array::buffer::DeviceBuffer;
use vortex::array::builtins::ArrayBuiltins;
use vortex::array::match_each_integer_ptype;
use vortex::array::validity::Validity;
use vortex::buffer::Alignment;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex_onpair::DictionaryView;
use vortex_onpair::MAX_TOKEN_SIZE;
use vortex_onpair::OnPair;
use vortex_onpair::OnPairArray;
use vortex_onpair::OnPairArrayExt;
use vortex_onpair::OnPairArraySlotsExt;
use vortex_onpair::code_boundary_at;
use vortex_onpair::dict_view;

use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::arrow::I32Offsets;
use crate::arrow::i32_offsets_from_lengths;
use crate::cub::onpair_batch_offsets;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;
use crate::kernel::encodings::DecodedVarBin;

// The kernels fix the dictionary row stride at 16 bytes (two `uint2` reads).
const _: () = assert!(MAX_TOKEN_SIZE == 16);

/// Tokens per decode batch: one decode-kernel warp emits 128 tokens (4 per
/// thread). Must match `ONPAIR_TOKENS_PER_BATCH` in `cub/kernels/onpair.cu`.
const TOKENS_PER_BATCH: usize = 128;
/// Threads per block for the warp-per-batch kernels (16 warps).
const BLOCK_THREADS: u32 = 512;
const WARPS_PER_BLOCK: usize = (BLOCK_THREADS / 32) as usize;

/// Launch config for the warp-per-batch kernels: one warp per 128-token batch.
fn batch_launch_config(num_batches: usize) -> VortexResult<LaunchConfig> {
    let grid_dim = u32::try_from(num_batches.div_ceil(WARPS_PER_BLOCK))?;
    Ok(LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    })
}

/// CUDA decoder for OnPair.
#[derive(Debug)]
pub(crate) struct OnPairExecutor;

#[async_trait]
impl CudaExecute for OnPairExecutor {
    #[instrument(level = "trace", skip_all, fields(executor = ?self))]
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let onpair = array
            .try_downcast::<OnPair>()
            .map_err(|_| vortex_err!("Expected OnPairArray"))?;
        decode_onpair(onpair, ctx).await
    }
}

/// Host sum of the per-row decoded lengths, rejecting negatives.
fn sum_lengths(lengths: &PrimitiveArray) -> VortexResult<u64> {
    match_each_integer_ptype!(lengths.ptype(), |P| {
        let mut acc = 0u64;
        #[allow(clippy::unnecessary_cast)]
        for &length in lengths.as_slice::<P>() {
            let length = u64::try_from(length as i128)
                .map_err(|_| vortex_err!("OnPair uncompressed length cannot be negative"))?;
            acc = acc
                .checked_add(length)
                .ok_or_else(|| vortex_err!("OnPair decoded size overflow"))?;
        }
        VortexResult::Ok(acc)
    })
}

/// All-empty output: `num_rows` inline empty views and no backing buffers.
async fn empty_views(
    num_rows: usize,
    dtype: DType,
    validity: Validity,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical> {
    let views = ctx.copy_to_device(vec![0i128; num_rows])?.await?;
    Ok(Canonical::VarBinView(unsafe {
        VarBinViewArray::new_handle_unchecked(views, Arc::from([]), dtype, validity)
    }))
}

/// The device-staged compressed token stream: the u16 code window, the split
/// dictionary layout, and the regenerated per-batch output offsets.
struct StagedCodes {
    codes: BufferHandle,
    dict_s8: BufferHandle,
    dict_padded: BufferHandle,
    lens: BufferHandle,
    /// Exclusive per-batch output offsets, `num_batches + 1` entries; the last
    /// is the total decoded byte count of the code window.
    chunk_offsets: CudaSlice<u64>,
    num_batches: usize,
    num_tokens: usize,
    launch_config: LaunchConfig,
}

/// The shared result of the OnPair GPU decode pipeline.
struct OnPairDecoded {
    /// The flat decoded byte stream.
    bytes: CudaSlice<u8>,
    /// Total decoded byte count, computed on device by the offsets
    /// regeneration.
    total_size: usize,
    /// Host-resident per-row lengths. Each output path derives what it needs:
    /// the views fast path prefix-sums them into row offsets, the varbin path
    /// builds Arrow i32 offsets from them on device, and the rollover path
    /// consumes them directly.
    lengths: PrimitiveArray,
}

/// Stage this array's code window and dictionary on the device and regenerate
/// the decode kernel's per-batch output offsets from them in one fused sweep
/// (see [`onpair_batch_offsets`]).
async fn stage_codes(
    onpair: &OnPairArray,
    code_start: usize,
    code_end: usize,
    status: &mut CudaSlice<u32>,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<StagedCodes> {
    // Widen this array's code window to the decode kernel's u16 ABI.
    let codes = onpair
        .codes()
        .slice(code_start..code_end)?
        .cast(DType::Primitive(PType::U16, Nullability::NonNullable))?
        .execute::<PrimitiveArray>(ctx.execution_ctx())?
        .into_buffer::<u16>();
    let num_tokens = codes.len();

    // Stage the dictionary in the decode kernel's split layout: fixed 16-byte
    // rows (`dict_padded`, the rare `len > 8` read), the first 8 bytes of every
    // row (`dict_s8`, the common-case read), and the per-code lengths.
    let dict = dict_view(onpair.as_view(), ctx.execution_ctx())?;
    let dict_size = dict.num_tokens();
    let dict_size_u32 = u32::try_from(dict_size)?;
    let mut dict_padded = vec![0u8; dict_size * MAX_TOKEN_SIZE];
    let mut dict_s8 = vec![0u8; dict_size * 8];
    let mut lens = vec![0u8; dict_size];
    for code in 0..dict_size {
        let token =
            dict.token(u16::try_from(code).vortex_expect("dictionary has at most 2^16 tokens"));
        let len = token.len();
        lens[code] = u8::try_from(len).vortex_expect("token length is at most MAX_TOKEN_SIZE");
        dict_padded[code * MAX_TOKEN_SIZE..code * MAX_TOKEN_SIZE + len].copy_from_slice(token);
        let head = len.min(8);
        dict_s8[code * 8..code * 8 + head].copy_from_slice(&token[..head]);
    }

    let (codes_dev, s8_dev, padded_dev, lens_dev) = futures::try_join!(
        ctx.copy_to_device(codes)?,
        ctx.copy_to_device(dict_s8)?,
        ctx.copy_to_device(dict_padded)?,
        ctx.copy_to_device(lens)?,
    )?;

    let num_batches = num_tokens.div_ceil(TOKENS_PER_BATCH);
    let launch_config = batch_launch_config(num_batches)?;
    let chunk_offsets = onpair_batch_offsets(
        &codes_dev,
        &lens_dev,
        dict_size_u32,
        num_tokens,
        num_batches,
        status,
        ctx,
    )?;

    Ok(StagedCodes {
        codes: codes_dev,
        dict_s8: s8_dev,
        dict_padded: padded_dev,
        lens: lens_dev,
        chunk_offsets,
        num_batches,
        num_tokens,
        launch_config,
    })
}

/// Run the OnPair decode pipeline: sum the per-row lengths on host,
/// regenerate the per-batch output offsets on the device, validate the
/// compressed stream, and decode the flat byte stream. Returns `Ok(None)`
/// when the array decodes to zero bytes.
async fn decode_onpair_bytes(
    onpair: &OnPairArray,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Option<OnPairDecoded>> {
    let num_rows = onpair.len();

    // Materialise the lengths child once; each output path derives its row
    // offsets from it and validates them against the GPU-computed total.
    let lengths = onpair
        .uncompressed_lengths()
        .clone()
        .execute::<PrimitiveArray>(ctx.execution_ctx())?;

    // `codes_offsets` may be a sliced view of the original; its first and last
    // boundaries bound the contiguous run of `codes` belonging to this array's
    // rows (`slice` keeps the full `codes` child and only narrows the offsets).
    let code_start = code_boundary_at(onpair.codes_offsets(), 0, ctx.execution_ctx())?;
    let code_end = code_boundary_at(onpair.codes_offsets(), num_rows, ctx.execution_ctx())?;
    vortex_ensure!(
        code_start <= code_end,
        "OnPair codes_offsets must be nondecreasing"
    );
    vortex_ensure!(
        code_end <= onpair.codes().len(),
        "OnPair codes_offsets end {} exceeds codes len {}",
        code_end,
        onpair.codes().len()
    );

    if code_start == code_end {
        // No codes: the array must decode to zero bytes.
        let total = sum_lengths(&lengths)?;
        vortex_ensure!(
            total == 0,
            "OnPair records {total} decoded bytes but has no codes"
        );
        return Ok(None);
    }

    // Corruption flag raised by the batch-sizes kernel for a code outside the
    // dictionary; checked before the unchecked decode kernel is allowed to run.
    let mut status = ctx.device_alloc::<u32>(1)?;
    ctx.stream()
        .memset_zeros(&mut status)
        .map_err(|e| vortex_err!("Failed to zero OnPair status flag: {e}"))?;

    let staged = stage_codes(onpair, code_start, code_end, &mut status, ctx).await?;

    // One synchronizing readback gates the decode kernel — whose dictionary
    // gathers and output scatters are unchecked — and yields the GPU-computed
    // total that sizes the output. The lengths child is validated against it
    // by whichever output path materialises row offsets.
    let status = ctx
        .stream()
        .clone_dtoh(&status)
        .map_err(|e| vortex_err!("Failed to copy OnPair status flag to host: {e}"))?;
    if status.first().copied().unwrap_or(1) != 0 {
        vortex_bail!("OnPair code out of dictionary range");
    }
    let chunk_total = ctx
        .stream()
        .clone_dtoh(
            &staged
                .chunk_offsets
                .slice(staged.num_batches..staged.num_batches + 1),
        )
        .map_err(|e| vortex_err!("Failed to copy OnPair decoded size to host: {e}"))?
        .first()
        .copied()
        .ok_or_else(|| vortex_err!("OnPair batch offset scan returned no total"))?;
    let total_size = usize::try_from(chunk_total)?;
    // A conformant dictionary has no zero-length tokens, so a non-empty code
    // window decodes to at least one byte.
    vortex_ensure!(total_size > 0, "OnPair has codes but decodes to zero bytes");

    // Decode. The kernel's drain gates 16-byte stores on `out_start % 16`
    // relative to the buffer base, so the base must be 16-aligned.
    let mut bytes = ctx.device_alloc::<u8>(total_size)?;
    let (bytes_base_ptr, _) = bytes.device_ptr(ctx.stream());
    assert_eq!(
        bytes_base_ptr % 16,
        0,
        "output base not 16-aligned: {bytes_base_ptr:#x}",
    );

    let num_tokens_u64 = u64::try_from(staged.num_tokens)?;
    let codes_view = staged.codes.cuda_view::<u16>()?;
    let s8_view = staged.dict_s8.cuda_view::<u8>()?;
    let padded_view = staged.dict_padded.cuda_view::<u8>()?;
    let lens_view = staged.lens.cuda_view::<u8>()?;
    let decode_fn = ctx.load_function_with_suffixes("onpair_shmem_4tpt_split8read", &[])?;
    ctx.launch_kernel_config(
        &decode_fn,
        staged.launch_config,
        staged.num_tokens,
        |args| {
            args.arg(&codes_view)
                .arg(&staged.chunk_offsets)
                .arg(&s8_view)
                .arg(&padded_view)
                .arg(&lens_view)
                .arg(&mut bytes)
                .arg(&num_tokens_u64);
        },
    )?;

    Ok(Some(OnPairDecoded {
        bytes,
        total_size,
        lengths,
    }))
}

async fn decode_onpair(onpair: OnPairArray, ctx: &mut CudaExecutionCtx) -> VortexResult<Canonical> {
    let dtype = onpair.dtype().clone();
    let validity = onpair.array_validity();
    let num_rows = onpair.len();

    if onpair.is_empty() {
        return Ok(Canonical::empty(&dtype));
    }

    if validity.definitely_all_null() {
        return empty_views(num_rows, dtype, validity, ctx).await;
    }

    let Some(decoded) = decode_onpair_bytes(&onpair, ctx).await? else {
        return empty_views(num_rows, dtype, validity, ctx).await;
    };
    let OnPairDecoded {
        bytes,
        total_size,
        lengths,
    } = decoded;

    // Fast path: the decoded heap fits a single BinaryView backing buffer, so
    // the per-row views build on-device. Only this path needs the u64 row
    // offsets: prefix-sum the lengths here and stage them on device.
    if total_size <= MAX_BUFFER_LEN {
        let row_offsets: Vec<u64> = match_each_integer_ptype!(lengths.ptype(), |P| {
            let mut offsets = Vec::with_capacity(lengths.len() + 1);
            let mut acc = 0u64;
            offsets.push(0u64);
            #[allow(clippy::unnecessary_cast)]
            for &length in lengths.as_slice::<P>() {
                let length = u64::try_from(length as i128)
                    .map_err(|_| vortex_err!("OnPair uncompressed length cannot be negative"))?;
                acc = acc
                    .checked_add(length)
                    .ok_or_else(|| vortex_err!("OnPair decoded size overflow"))?;
                offsets.push(acc);
            }
            VortexResult::Ok(offsets)
        })?;
        // The views index the decoded heap, so the lengths must account for
        // exactly the bytes the codes decoded to.
        let row_total = *row_offsets
            .last()
            .vortex_expect("row_offsets has at least one entry");
        vortex_ensure!(
            row_total == total_size as u64,
            "OnPair codes decode to {total_size} bytes but uncompressed_lengths records {row_total}"
        );
        let row_offsets_dev = ctx.copy_to_device(row_offsets)?.await?;
        let row_offsets_view = row_offsets_dev.cuda_view::<u64>()?;
        let mut device_views = ctx.device_alloc::<i128>(num_rows)?;
        let num_rows_u64 = u64::try_from(num_rows)?;
        let build_views_fn = ctx.load_function_with_suffixes("onpair", &["build_views"])?;
        ctx.launch_kernel(&build_views_fn, num_rows, |args| {
            args.arg(&row_offsets_view)
                .arg(&bytes)
                .arg(&mut device_views)
                .arg(&num_rows_u64);
        })?;

        let views = BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(device_views)));
        let bytes = BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(bytes)));
        return Ok(Canonical::VarBinView(unsafe {
            VarBinViewArray::new_handle_unchecked(views, Arc::from([bytes]), dtype, validity)
        }));
    }

    // BinaryView offsets are u32. Heaps that need multiple backing buffers
    // roll the decoded bytes over on host, mirroring the CPU canonical path.
    // The host views index the copied heap, so validate the lengths first.
    let row_total = sum_lengths(&lengths)?;
    vortex_ensure!(
        row_total == total_size as u64,
        "OnPair codes decode to {total_size} bytes but uncompressed_lengths records {row_total}"
    );
    let host_bytes = CudaDeviceBuffer::new(bytes)
        .copy_to_host(Alignment::new(1))?
        .await?;
    let host_bytes = host_bytes.slice(0..total_size);

    let (buffers, views) = match_each_integer_ptype!(lengths.ptype(), |P| {
        build_views(
            0,
            MAX_BUFFER_LEN,
            host_bytes.into_mut(),
            lengths.as_slice::<P>(),
        )
    });

    Ok(Canonical::VarBinView(unsafe {
        VarBinViewArray::new_unchecked(views, Arc::from(buffers), dtype, validity)
    }))
}

/// Decode OnPair directly into Arrow-compatible i32 offsets and contiguous
/// values on device, mirroring the FSST varbin path.
pub(crate) async fn decode_onpair_varbin(
    onpair: OnPairArray,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<DecodedVarBin> {
    let dtype = onpair.dtype().clone();
    let validity = onpair.array_validity();
    let len = onpair.len();

    let decoded = if onpair.is_empty() || validity.definitely_all_null() {
        None
    } else {
        decode_onpair_bytes(&onpair, ctx).await?
    };

    let Some(decoded) = decoded else {
        // Zero decoded bytes: all-zero offsets and an empty values heap.
        let offsets = ctx.copy_to_device(vec![0i32; len + 1])?.await?;
        let allocation = CudaDeviceBuffer::new(ctx.device_alloc::<u8>(1)?);
        let values = BufferHandle::new_device(allocation.slice(0..0));
        return Ok(DecodedVarBin {
            dtype,
            len,
            offsets,
            values,
            validity,
        });
    };

    // Build the Arrow i32 offsets from the lengths on device; this also
    // rejects heaps beyond Arrow's i32 offset range.
    let I32Offsets {
        buffer: offsets,
        total,
    } = i32_offsets_from_lengths(decoded.lengths.clone(), ctx).await?;
    // The Arrow offsets index the decoded heap, so the lengths must account
    // for exactly the bytes the codes decoded to.
    vortex_ensure!(
        total == decoded.total_size,
        "OnPair codes decode to {} bytes but uncompressed_lengths records {total}",
        decoded.total_size
    );

    Ok(DecodedVarBin {
        dtype,
        len,
        offsets,
        values: BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(decoded.bytes))),
        validity,
    })
}

#[cfg(test)]
mod tests {
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use rstest::rstest;
    use vortex::array::IntoArray;
    use vortex::array::arrays::VarBinArray;
    use vortex::array::assert_arrays_eq;
    use vortex::buffer::Buffer;
    use vortex::error::VortexExpect;
    use vortex_array::VortexSessionExecute;
    use vortex_onpair::DEFAULT_DICT12_CONFIG;
    use vortex_onpair::onpair_compress;

    use super::*;
    use crate::CanonicalCudaExt;
    use crate::arrow::DeviceArrayExt;
    use crate::arrow::release_device_array;
    use crate::arrow::release_schema;
    use crate::session::CudaSession;
    use crate::session::VarBinExportLayout;

    fn cuda_ctx_with_varbin_layout(layout: VarBinExportLayout) -> VortexResult<CudaExecutionCtx> {
        let session = vortex::array::array_session()
            .with_some(CudaSession::try_default()?.with_varbin_export_layout(layout));
        CudaSession::create_execution_ctx(&session)
    }

    fn assert_device_resident(canonical: &Canonical) {
        let varbinview = canonical.as_varbinview();
        assert!(varbinview.views_handle().is_on_device());
        assert!(
            varbinview
                .data_buffers()
                .iter()
                .all(BufferHandle::is_on_device)
        );
    }

    fn compress_onpair(
        strings: Vec<Option<&'static [u8]>>,
        dtype: DType,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let varbin = VarBinArray::from_iter(strings, dtype).into_array();
        let onpair = onpair_compress(&varbin, DEFAULT_DICT12_CONFIG, ctx.execution_ctx())?;
        vortex_ensure!(
            onpair.as_opt::<OnPair>().is_some(),
            "expected OnPair array, got {}",
            onpair.encoding_id()
        );
        Ok(onpair)
    }

    #[rstest]
    #[case::binary_non_null(
        vec![Some(&b"the quick brown fox"[..]),
             Some(&b"jumps over the lazy dog"[..]),
             Some(&b"hello world"[..]),
             Some(&b"vortex onpair test string"[..])],
        DType::Binary(Nullability::NonNullable),
    )]
    #[case::utf8_non_null(
        vec![Some(&b"the quick brown fox"[..]),
             Some(&b"jumps over the lazy dog"[..]),
             Some(&b"hello world"[..]),
             Some(&b"vortex onpair test string"[..])],
        DType::Utf8(Nullability::NonNullable),
    )]
    #[case::utf8_inline_boundary(
        vec![Some(&b""[..]),
             Some(&b"123456789012"[..]),
             Some(&b"1234567890123"[..]),
             Some(&b"this is another outlined value"[..])],
        DType::Utf8(Nullability::NonNullable),
    )]
    #[case::utf8_partial_nulls(
        vec![Some(&b"alpha"[..]), None, Some(&b"gamma"[..]), None, Some(&b"epsilon"[..])],
        DType::Utf8(Nullability::Nullable),
    )]
    #[case::binary_all_empty(
        vec![Some(&b""[..]), Some(&b""[..]), Some(&b""[..])],
        DType::Binary(Nullability::NonNullable),
    )]
    #[crate::test]
    async fn test_cuda_onpair_decompression_roundtrip(
        #[case] strings: Vec<Option<&'static [u8]>>,
        #[case] dtype: DType,
    ) -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create execution context");

        let onpair = compress_onpair(strings, dtype.clone(), &mut cuda_ctx)?;

        let gpu_result = OnPairExecutor
            .execute(onpair.clone(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed");
        assert_eq!(gpu_result.dtype(), &dtype);
        assert_device_resident(&gpu_result);

        let host_result = gpu_result.into_host().await?.into_array();
        assert_arrays_eq!(onpair, host_result, &mut ctx);
        Ok(())
    }

    /// A slice keeps the whole `codes` child and narrows only `codes_offsets`,
    /// so this exercises the nonzero `code_start` window.
    #[crate::test]
    async fn test_cuda_onpair_decompression_sliced() -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create execution context");
        let values = vec![
            Some(&b"before the window"[..]),
            None,
            Some(&b"the quick brown fox"[..]),
            None,
            Some(&b"after the window"[..]),
        ];
        let onpair = compress_onpair(values, DType::Utf8(Nullability::Nullable), &mut cuda_ctx)?;
        let sliced = onpair.slice(1..4)?;

        let gpu_result = OnPairExecutor
            .execute(sliced.clone(), &mut cuda_ctx)
            .await?;
        assert_device_resident(&gpu_result);
        let host_result = gpu_result.into_host().await?.into_array();
        assert_arrays_eq!(sliced, host_result, &mut ctx);
        Ok(())
    }

    /// A slice covering only null rows decodes zero bytes and takes the
    /// empty-views path.
    #[crate::test]
    async fn test_cuda_onpair_decompression_null_slice() -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create execution context");
        let values = vec![Some(&b"alpha"[..]), None, None, Some(&b"omega"[..])];
        let onpair = compress_onpair(values, DType::Utf8(Nullability::Nullable), &mut cuda_ctx)?;
        let sliced = onpair.slice(1..3)?;

        let gpu_result = OnPairExecutor
            .execute(sliced.clone(), &mut cuda_ctx)
            .await?;
        assert_device_resident(&gpu_result);
        let host_result = gpu_result.into_host().await?.into_array();
        assert_arrays_eq!(sliced, host_result, &mut ctx);
        Ok(())
    }

    /// Exercises many 128-token batches and the multi-block decode grid.
    #[crate::test]
    async fn test_cuda_onpair_decompression_roundtrip_large() -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create execution context");

        let strings: Vec<String> = (0..100_000)
            .map(|i| format!("https://www.example.com/path/{i}/segment?q={}", i % 97))
            .collect();
        let varbin = VarBinArray::from_iter(
            strings.iter().map(|s| Some(s.as_str())),
            DType::Utf8(Nullability::NonNullable),
        )
        .into_array();
        let onpair = onpair_compress(&varbin, DEFAULT_DICT12_CONFIG, cuda_ctx.execution_ctx())?;

        let gpu_result = OnPairExecutor
            .execute(onpair.clone(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed");
        assert_device_resident(&gpu_result);

        let host_result = gpu_result.into_host().await?.into_array();
        assert_arrays_eq!(onpair, host_result, &mut ctx);
        Ok(())
    }

    #[crate::test]
    async fn test_cuda_onpair_direct_varbin_output() -> VortexResult<()> {
        let mut cuda_ctx = cuda_ctx_with_varbin_layout(VarBinExportLayout::VarBin)?;
        let values: [&[u8]; 3] = [
            b"",
            b"short",
            b"this value is stored directly in the values buffer",
        ];
        let onpair = compress_onpair(
            values.iter().map(|v| Some(*v)).collect(),
            DType::Utf8(Nullability::NonNullable),
            &mut cuda_ctx,
        )?
        .try_downcast::<OnPair>()
        .map_err(|array| vortex_err!("expected OnPair array, got {}", array.encoding_id()))?;

        let output = decode_onpair_varbin(onpair, &mut cuda_ctx).await?;
        assert_eq!(output.dtype, DType::Utf8(Nullability::NonNullable));
        assert_eq!(output.len, values.len());
        assert!(output.offsets.is_on_device());
        assert!(output.values.is_on_device());

        let offsets = Buffer::<i32>::from_byte_buffer(output.offsets.try_to_host()?.await?);
        assert_eq!(
            offsets.as_slice(),
            &[0, 0, 5, i32::try_from(5 + values[2].len())?,]
        );
        assert_eq!(
            output.values.try_to_host()?.await?.as_ref(),
            values.concat()
        );
        Ok(())
    }

    #[rstest]
    #[case::binary(
        DType::Binary(Nullability::NonNullable),
        VarBinExportLayout::VarBin,
        DataType::Binary,
        3
    )]
    #[case::utf8(
        DType::Utf8(Nullability::NonNullable),
        VarBinExportLayout::VarBin,
        DataType::Utf8,
        3
    )]
    #[case::binary_view(
        DType::Binary(Nullability::NonNullable),
        VarBinExportLayout::VarBinView,
        DataType::BinaryView,
        4
    )]
    #[case::utf8_view(
        DType::Utf8(Nullability::NonNullable),
        VarBinExportLayout::VarBinView,
        DataType::Utf8View,
        4
    )]
    #[crate::test]
    async fn test_cuda_onpair_arrow_export_uses_dtype_layout(
        #[case] dtype: DType,
        #[case] layout: VarBinExportLayout,
        #[case] expected_data_type: DataType,
        #[case] expected_n_buffers: i64,
    ) -> VortexResult<()> {
        let mut cuda_ctx = cuda_ctx_with_varbin_layout(layout)?;
        let values = vec![
            Some(&b"short"[..]),
            Some(&b"this value is stored out of line"[..]),
        ];
        let onpair = compress_onpair(values, dtype, &mut cuda_ctx)?;

        let mut exported = onpair
            .export_device_array_with_schema(&mut cuda_ctx)
            .await?;
        assert_eq!(
            Field::try_from(&exported.schema)?,
            Field::new("", expected_data_type, false)
        );
        assert_eq!(exported.array.array.n_buffers, expected_n_buffers);

        release_device_array(&mut exported.array);
        release_schema(&mut exported.schema);
        Ok(())
    }
}
