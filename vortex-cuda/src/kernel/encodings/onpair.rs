// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA executor for OnPair decompression.
//!
//! Decoding runs on the GPU over the full token stream — a sliced array keeps
//! its whole `codes` child, and buffers never round-trip between host and
//! device to cut a window out of it:
//!
//! 1. `onpair_batch_offsets` (in the CUB shim) regenerates the per-batch
//!    output offsets (`chunk_offsets`) the decode kernel positions its writes
//!    with: one fused sweep reduces every 128-token batch's decoded size and
//!    exclusive-scans the sizes in-kernel via decoupled look-back.
//! 2. `onpair_window_offsets` reads this array's token window from the
//!    device-resident `codes_offsets` child — the offsets are nondecreasing,
//!    so the window's min and max are its first and last elements — and
//!    resolves the window's byte positions inside the decoded stream in the
//!    same launch: the whole-batch prefix from `chunk_offsets` plus a
//!    partial-batch reduction over each boundary batch's head. No boundary is
//!    read on host before the single gating readback.
//! 3. `onpair_shmem_4tpt_split8read` gathers each token's bytes from the
//!    split dictionary layout and scatters them to the output byte stream;
//!    the window is then exposed as a zero-copy device slice of the heap.
//!
//! Every kernel that reads the codes is instantiated for the two widths
//! OnPair stores (u16 natively, u8 when the compressor narrowed the codes),
//! so the code stream is decompressed on device and never widened.
//!
//! The heap size and window bounds come from the GPU; each output path builds
//! its row offsets from the lengths child with [`i32_offsets_from_lengths`]
//! on device and validates them against the window size before they index the
//! decoded bytes. The result is exposed either as a canonical `VarBinView`
//! (views built on-device by `onpair_build_views`, or on host for windows
//! that exceed a single backing buffer — the only path that materialises the
//! lengths) or as Arrow-compatible i32 offsets plus values via
//! [`decode_onpair_varbin`] — mirroring the FSST varbin path.

use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DeviceRepr;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use num_traits::AsPrimitive;
use tracing::instrument;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::arrays::primitive::PrimitiveDataParts;
use vortex::array::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use vortex::array::arrays::varbinview::build_views::build_views;
use vortex::array::buffer::BufferHandle;
use vortex::array::buffer::DeviceBuffer;
use vortex::array::match_each_integer_ptype;
use vortex::array::validity::Validity;
use vortex::dtype::DType;
use vortex::dtype::NativePType;
use vortex::dtype::PType;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex_array::ArrayView;
use vortex_onpair::DictionaryView;
use vortex_onpair::MAX_TOKEN_SIZE;
use vortex_onpair::OnPair;
use vortex_onpair::OnPairArray;
use vortex_onpair::OnPairArrayExt;
use vortex_onpair::OnPairArraySlotsExt;
use vortex_onpair::dict_view;

use crate::CanonicalCudaExt;
use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::arrow::I32Offsets;
use crate::arrow::i32_offsets_from_lengths;
use crate::cub::onpair_batch_offsets;
use crate::executor::CudaArrayExt;
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
            .as_typed::<OnPair>()
            .ok_or_else(|| vortex_err!("Expected OnPairArray"))?;
        decode_onpair(onpair, ctx).await
    }
}

async fn decode_onpair(
    onpair: ArrayView<'_, OnPair>,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical> {
    let dtype = onpair.dtype().clone();
    let validity = onpair.array_validity();
    let num_rows = onpair.len();

    if onpair.is_empty() {
        return Ok(Canonical::empty(&dtype));
    }

    let Some(decoded) = decode_onpair_bytes(onpair, ctx).await? else {
        return empty_views(num_rows, dtype, validity, ctx).await;
    };
    let OnPairDecoded {
        bytes,
        total_size,
        lengths,
    } = decoded;

    // Fast path: the decoded window fits a single BinaryView backing buffer
    // (`MAX_BUFFER_LEN`, i32::MAX), so the per-row offsets fit Arrow's i32
    // range and build on device from the device-resident lengths — nothing
    // touches the host.
    if total_size <= MAX_BUFFER_LEN {
        let I32Offsets {
            buffer: row_offsets,
            total,
        } = i32_offsets_from_lengths(lengths, ctx).await?;
        let row_total = u64::try_from(total)?;
        vortex_ensure!(
            row_total == total_size as u64,
            "OnPair codes decode to {total_size} bytes but uncompressed_lengths records {row_total}"
        );
        let row_offsets_view = row_offsets.cuda_view::<i32>()?;
        let bytes_view = bytes.cuda_view::<u8>()?;
        let mut device_views = ctx.device_alloc::<i128>(num_rows)?;
        let num_rows_u64 = u64::try_from(num_rows)?;
        let build_views_fn = ctx.load_function_with_suffixes("onpair", &["build_views"])?;
        ctx.launch_kernel(&build_views_fn, num_rows, |args| {
            args.arg(&row_offsets_view)
                .arg(&bytes_view)
                .arg(&mut device_views)
                .arg(&num_rows_u64);
        })?;

        let views = BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(device_views)));
        return Ok(Canonical::VarBinView(unsafe {
            VarBinViewArray::new_handle_unchecked(views, Arc::from([bytes]), dtype, validity)
        }));
    }

    // BinaryView offsets are u32. Windows that need multiple backing buffers
    // roll the decoded bytes over on host, mirroring the CPU canonical path;
    // only here do the lengths leave the device. The host views index the
    // copied window, so validate the lengths first.
    let lengths = Canonical::Primitive(lengths)
        .into_host()
        .await?
        .into_primitive();
    let row_total = sum_lengths(&lengths)?;
    vortex_ensure!(
        row_total == total_size as u64,
        "OnPair codes decode to {total_size} bytes but uncompressed_lengths records {row_total}"
    );
    let host_bytes = bytes.try_to_host()?.await?;

    let (buffers, views) = match_each_integer_ptype!(lengths.ptype(), |P| {
        build_views(0, MAX_BUFFER_LEN, host_bytes, lengths.as_slice::<P>())
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

    let Some(decoded) = decode_onpair_bytes(onpair.as_view(), ctx).await? else {
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

    let OnPairDecoded {
        bytes,
        total_size,
        lengths,
    } = decoded;

    // Build the Arrow i32 offsets from the lengths on device; this also
    // rejects windows beyond Arrow's i32 offset range.
    let I32Offsets {
        buffer: offsets,
        total,
    } = i32_offsets_from_lengths(lengths, ctx).await?;
    let row_total = u64::try_from(total)?;
    vortex_ensure!(
        row_total == total_size as u64,
        "OnPair codes decode to {total_size} bytes but uncompressed_lengths records {row_total}"
    );

    Ok(DecodedVarBin {
        dtype,
        len,
        offsets,
        values: bytes,
        validity,
    })
}

/// The shared result of the OnPair GPU decode pipeline.
struct OnPairDecoded {
    /// This array's rows' decoded bytes: a zero-copy device slice of the full
    /// decoded heap, bounded by the on-device window-offsets resolution.
    bytes: BufferHandle,
    /// Byte size of the window, computed on device.
    total_size: usize,
    /// Per-row lengths. The varbin path and the canonical fast path build
    /// their row offsets from them on device; only the host rollover path
    /// materialises them.
    lengths: PrimitiveArray,
}

/// Run the OnPair decode pipeline over the full token stream: stage the codes
/// and dictionary, regenerate the per-batch output offsets on the device,
/// validate the compressed stream, and decode the flat byte stream. A sliced
/// array keeps its whole `codes` child, so the decode runs unwindowed and this
/// array's rows are exposed as a zero-copy device slice of the decoded heap,
/// bounded by the on-device `onpair_window_offsets` resolution — the codes
/// never round-trip through the host. Returns `Ok(None)` when there is
/// nothing to decode: the array is empty, every row is null, or the code
/// window is empty.
async fn decode_onpair_bytes(
    onpair: ArrayView<'_, OnPair>,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Option<OnPairDecoded>> {
    let num_rows = onpair.len();

    if num_rows == 0 {
        return Ok(None);
    }

    // Every row null (cheap metadata check): nothing to decode. A sliced
    // all-null window usually carries a validity child instead of the
    // `AllInvalid` marker and is caught when the token window resolves empty.
    if onpair.array_validity().definitely_all_null() {
        return Ok(None);
    }

    // No codes at all (e.g. every row empty): the child's length is host
    // metadata, so this early-out costs no device read.
    if onpair.codes().is_empty() {
        let lengths = decode_primitive_child(onpair.uncompressed_lengths().clone(), ctx).await?;
        ensure_zero_lengths(lengths).await?;
        return Ok(None);
    }

    // The three children and the dictionary staging all run on `ctx`'s
    // stream: the per-row lengths (consumed only by the output paths), the
    // per-row code boundaries (the token window is resolved from them by a
    // kernel, never by host scalar reads), the codes themselves, and the
    // split dictionary staging.
    let lengths = decode_primitive_child(onpair.uncompressed_lengths().clone(), ctx).await?;
    let codes_offsets = decode_primitive_child(onpair.codes_offsets().clone(), ctx).await?;
    let codes = decode_primitive_child(onpair.codes().clone(), ctx).await?;
    let dict = stage_dict(onpair, ctx).await?;

    // The kernels are instantiated for the two widths OnPair stores — u16
    // natively, u8 when the compressor narrowed the codes — so no widening
    // pass is needed.
    match codes.ptype() {
        PType::U8 => decode_window::<u8>(codes, codes_offsets, lengths, dict, ctx).await,
        PType::U16 => decode_window::<u16>(codes, codes_offsets, lengths, dict, ctx).await,
        other => vortex_bail!("OnPair codes must decompress to u8 or u16, got {other}"),
    }
}

/// The dictionary staged device-resident in the decode kernel's split layout.
struct StagedDict {
    dict_s8: BufferHandle,
    dict_padded: BufferHandle,
    lens: BufferHandle,
    dict_size: u32,
}

/// Stage the dictionary in the decode kernel's split layout: fixed 16-byte
/// rows (`dict_padded`, the rare `len > 8` read), the first 8 bytes of every
/// row (`dict_s8`, the common-case read), and the per-code lengths.
async fn stage_dict(
    onpair: ArrayView<'_, OnPair>,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<StagedDict> {
    let dict = dict_view(onpair, ctx.execution_ctx())?;
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

    let (s8_dev, padded_dev, lens_dev) = futures::try_join!(
        ctx.copy_to_device(dict_s8)?,
        ctx.copy_to_device(dict_padded)?,
        ctx.copy_to_device(lens)?,
    )?;

    Ok(StagedDict {
        dict_s8: s8_dev,
        dict_padded: padded_dev,
        lens: lens_dev,
        dict_size: dict_size_u32,
    })
}

/// Stage the codes at their native width `C`, resolve the window bounds on
/// device, validate the stream, and decode the full token stream. Returns
/// `Ok(None)` when the token window turns out to be empty.
async fn decode_window<C>(
    codes: PrimitiveArray,
    codes_offsets: PrimitiveArray,
    lengths: PrimitiveArray,
    dict: StagedDict,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Option<OnPairDecoded>>
where
    C: NativePType + DeviceRepr + Send + Sync + 'static,
{
    // Corruption flag raised by the batch-sizes kernel for a code outside the
    // dictionary; checked before the unchecked decode kernel is allowed to run.
    let mut status = ctx.device_alloc::<u32>(1)?;
    ctx.stream()
        .memset_zeros(&mut status)
        .map_err(|e| vortex_err!("Failed to zero OnPair status flag: {e}"))?;

    let staged = stage_codes(codes, &dict, &mut status, ctx).await?;

    let ptype = C::PTYPE.to_string();
    let num_tokens_u64 = u64::try_from(staged.num_tokens)?;
    let codes_view = staged.codes.cuda_view::<C>()?;
    let s8_view = dict.dict_s8.cuda_view::<u8>()?;
    let padded_view = dict.dict_padded.cuda_view::<u8>()?;
    let lens_view = dict.lens.cuda_view::<u8>()?;

    // The readback scratch: the fused window-offsets kernel writes the token
    // window into slots 0..2, the byte window into slots 2..4, and packs the
    // sweep's outputs — the decoded heap size and the corruption status flag
    // — into slots 4..6, so one readback gates the decode kernel.
    let mut scratch = ctx.device_alloc::<u64>(6)?;

    // Token and byte bounds of this array's rows, resolved on device by one
    // fused launch — one warp per boundary. Each warp reads its token
    // boundary from the (possibly slice-narrowed) `codes_offsets` child (the
    // offsets are nondecreasing, so the window's min and max are its first
    // and last elements) and resolves its byte position inside the decoded
    // stream: the whole-batch prefix from `chunk_offsets` plus a
    // partial-batch reduction over the boundary batch's head. The kernel is
    // instantiated per (code width, offsets ptype) pair.
    let offsets_ptype = codes_offsets.ptype();
    let last_boundary = u64::try_from(codes_offsets.len().saturating_sub(1))?;
    let PrimitiveDataParts {
        buffer: offsets_buffer,
        ..
    } = codes_offsets.into_data_parts();
    let offsets_dev = ctx.ensure_on_device(offsets_buffer).await?;
    let window_fn = ctx.load_function_with_suffixes(
        "onpair",
        &["window_offsets", &ptype, &offsets_ptype.to_string()],
    )?;
    match_each_integer_ptype!(offsets_ptype, |O| {
        let offsets_view = offsets_dev.cuda_view::<O>()?;
        ctx.launch_kernel_config(
            &window_fn,
            LaunchConfig {
                grid_dim: (2, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
            2,
            |args| {
                args.arg(&codes_view)
                    .arg(&lens_view)
                    .arg(&dict.dict_size)
                    .arg(&staged.chunk_offsets)
                    .arg(&num_tokens_u64)
                    .arg(&offsets_view)
                    .arg(&last_boundary)
                    .arg(&status)
                    .arg(&mut scratch);
            },
        )?;
    });

    // One synchronizing readback gates the decode kernel — whose dictionary
    // gathers and output scatters are unchecked — and yields the GPU-computed
    // heap size, token window, byte window, and corruption status in a single
    // round trip. The lengths child is validated against the window size by
    // whichever output path materialises row offsets.
    let scratch = ctx
        .stream()
        .clone_dtoh(&scratch)
        .map_err(|e| vortex_err!("Failed to copy OnPair window scratch to host: {e}"))?;
    let [
        token_start,
        token_end,
        byte_start,
        byte_end,
        chunk_total,
        status,
    ] = scratch[..]
    else {
        vortex_bail!("OnPair window resolution returned no bounds");
    };
    if status != 0 {
        vortex_bail!("OnPair code out of dictionary range");
    }
    let heap_size = usize::try_from(chunk_total)?;
    vortex_ensure!(
        token_start <= token_end,
        "OnPair codes_offsets must be nondecreasing"
    );
    vortex_ensure!(
        token_end <= num_tokens_u64,
        "OnPair codes_offsets end {token_end} exceeds codes len {num_tokens_u64}"
    );
    if token_start == token_end {
        // No codes in the window (e.g. a slice covering only null rows).
        ensure_zero_lengths(lengths).await?;
        return Ok(None);
    }
    let byte_start = usize::try_from(byte_start)?;
    let byte_end = usize::try_from(byte_end)?;
    vortex_ensure!(
        byte_start <= byte_end && byte_end <= heap_size,
        "OnPair window bounds [{byte_start}, {byte_end}) exceed decoded heap size {heap_size}"
    );
    let total_size = byte_end - byte_start;
    // A conformant dictionary has no zero-length tokens, so a non-empty code
    // window decodes to at least one byte.
    vortex_ensure!(total_size > 0, "OnPair has codes but decodes to zero bytes");

    // Decode the full stream. The kernel's drain gates 16-byte stores on
    // `out_start % 16` relative to the buffer base, so the base must be
    // 16-aligned.
    let mut bytes = ctx.device_alloc::<u8>(heap_size)?;
    let (bytes_base_ptr, _) = bytes.device_ptr(ctx.stream());
    assert_eq!(
        bytes_base_ptr % 16,
        0,
        "output base not 16-aligned: {bytes_base_ptr:#x}",
    );

    let decode_fn = ctx.load_function_with_suffixes("onpair_shmem_4tpt_split8read", &[&ptype])?;
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

    // This array's rows as a zero-copy device slice of the decoded heap.
    let heap = CudaDeviceBuffer::new(bytes);
    Ok(Some(OnPairDecoded {
        bytes: BufferHandle::new_device(heap.slice(byte_start..byte_end)),
        total_size,
        lengths,
    }))
}

/// The device-staged compressed token stream: the full code stream at its
/// native width and the regenerated per-batch output offsets.
struct StagedCodes {
    codes: BufferHandle,
    /// Exclusive per-batch output offsets, `num_batches + 1` entries; the last
    /// is the total decoded byte count of the full code stream.
    chunk_offsets: CudaSlice<u64>,
    num_tokens: usize,
    launch_config: LaunchConfig,
}

/// Stage this array's device-decompressed codes and regenerate the decode
/// kernel's per-batch output offsets from them and the staged dictionary in
/// one fused sweep (see [`onpair_batch_offsets`]). The caller has validated
/// that the codes are u8 or u16; the sweep reads them at their native width.
async fn stage_codes(
    codes: PrimitiveArray,
    dict: &StagedDict,
    status: &mut CudaSlice<u32>,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<StagedCodes> {
    let num_tokens = codes.len();
    let code_width = u32::try_from(codes.ptype().byte_width())?;
    let PrimitiveDataParts {
        buffer: codes_buffer,
        ..
    } = codes.into_data_parts();
    let codes_dev = ctx.ensure_on_device(codes_buffer).await?;

    let num_batches = num_tokens.div_ceil(TOKENS_PER_BATCH);
    let launch_config = batch_launch_config(num_batches)?;
    let chunk_offsets = onpair_batch_offsets(
        &codes_dev,
        code_width,
        &dict.lens,
        dict.dict_size,
        num_tokens,
        num_batches,
        status,
        ctx,
    )?;

    Ok(StagedCodes {
        codes: codes_dev,
        chunk_offsets,
        num_tokens,
        launch_config,
    })
}

/// Launch config for the warp-per-batch kernels: one warp per 128-token batch.
fn batch_launch_config(num_batches: usize) -> VortexResult<LaunchConfig> {
    let grid_dim = u32::try_from(num_batches.div_ceil(WARPS_PER_BLOCK))?;
    Ok(LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
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

/// Decompress a child array to a canonical primitive on `ctx`'s stream.
async fn decode_primitive_child(
    child: ArrayRef,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    Ok(child.execute_cuda(ctx).await?.into_primitive())
}

/// Cold path: the window has no codes, so the rows must decode to zero bytes.
async fn ensure_zero_lengths(lengths: PrimitiveArray) -> VortexResult<()> {
    let lengths = Canonical::Primitive(lengths)
        .into_host()
        .await?
        .into_primitive();
    let total = sum_lengths(&lengths)?;
    vortex_ensure!(
        total == 0,
        "OnPair records {total} decoded bytes but has no codes"
    );
    Ok(())
}

/// Checked host sum of the per-row decoded lengths. A negative length
/// sign-extends and surfaces as overflow here or as a mismatch against the
/// GPU-computed window size.
fn sum_lengths(lengths: &PrimitiveArray) -> VortexResult<u64> {
    match_each_integer_ptype!(lengths.ptype(), |P| {
        let mut acc = 0u64;
        for &length in lengths.as_slice::<P>() {
            acc = acc
                .checked_add(AsPrimitive::<u64>::as_(length))
                .ok_or_else(|| vortex_err!("OnPair decoded size overflow"))?;
        }
        Ok(acc)
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
    use vortex::dtype::Nullability;
    use vortex::error::VortexExpect;
    use vortex_array::VortexSessionExecute;
    use vortex_onpair::DEFAULT_CONFIG;
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
        let onpair = onpair_compress(&varbin, DEFAULT_CONFIG, ctx.execution_ctx())?;
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
        let onpair = onpair_compress(&varbin, DEFAULT_CONFIG, cuda_ctx.execution_ctx())?;

        let gpu_result = OnPairExecutor
            .execute(onpair.clone(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed");
        assert_device_resident(&gpu_result);

        let host_result = gpu_result.into_host().await?.into_array();
        assert_arrays_eq!(onpair, host_result, &mut ctx);
        Ok(())
    }

    /// A slice deep into a large array: both code-window boundaries land
    /// mid-batch in non-zero batches, exercising the on-device window-bounds
    /// resolution (whole-batch prefix plus partial-batch reduction) and the
    /// zero-copy window slice of the full decoded heap.
    #[crate::test]
    async fn test_cuda_onpair_decompression_sliced_large() -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create execution context");

        let strings: Vec<String> = (0..40_000)
            .map(|i| format!("https://www.example.com/path/{i}/segment?q={}", i % 97))
            .collect();
        let varbin = VarBinArray::from_iter(
            strings.iter().map(|s| Some(s.as_str())),
            DType::Utf8(Nullability::NonNullable),
        )
        .into_array();
        let onpair = onpair_compress(&varbin, DEFAULT_CONFIG, cuda_ctx.execution_ctx())?;
        let sliced = onpair.slice(19_997..20_101)?;

        let gpu_result = OnPairExecutor
            .execute(sliced.clone(), &mut cuda_ctx)
            .await?;
        assert_device_resident(&gpu_result);
        let host_result = gpu_result.into_host().await?.into_array();
        assert_arrays_eq!(sliced, host_result, &mut ctx);
        Ok(())
    }

    /// Codes narrowed to u8 dispatch the u8 kernel instantiations end to end.
    /// A trained dictionary always holds the 256 single-byte tokens sorted
    /// among its merges, so real merge codes never fit u8; the u8-addressable
    /// case is the minimal alphabet-only dictionary, where token id `b` is
    /// exactly the byte `b` and every row is coded byte per byte.
    #[crate::test]
    async fn test_cuda_onpair_decompression_u8_codes() -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create execution context");

        let strings: Vec<&[u8]> = [
            &b"tokenized token stream"[..],
            b"tokenized",
            b"token stream",
            b"stream of tokens",
        ]
        .into_iter()
        .cycle()
        .take(800)
        .collect();

        // The alphabet-only compact dictionary: the 256 single-byte tokens
        // (sorted by construction) plus the trailing read padding.
        let mut dict_bytes: Vec<u8> = (0..=u8::MAX).collect();
        dict_bytes.resize(255 + MAX_TOKEN_SIZE, 0);
        let dict_offsets: Vec<u32> = (0..=256).collect();

        let codes: Vec<u8> = strings.concat();
        let mut codes_offsets = vec![0u32];
        let mut lengths = Vec::with_capacity(strings.len());
        let mut acc = 0u32;
        for s in &strings {
            let len = u32::try_from(s.len())?;
            lengths.push(len);
            acc += len;
            codes_offsets.push(acc);
        }

        let onpair = OnPair::try_new(
            DType::Utf8(Nullability::NonNullable),
            BufferHandle::new_host(Buffer::from(dict_bytes).into_byte_buffer()),
            Buffer::from(dict_offsets).into_array(),
            Buffer::from(codes).into_array(),
            Buffer::from(codes_offsets).into_array(),
            Buffer::from(lengths).into_array(),
            Validity::NonNullable,
        )?;

        let expected = VarBinArray::from_iter(
            strings.iter().map(|s| Some(*s)),
            DType::Utf8(Nullability::NonNullable),
        )
        .into_array();

        let gpu_result = OnPairExecutor
            .execute(onpair.into_array(), &mut cuda_ctx)
            .await?;
        assert_device_resident(&gpu_result);
        let host_result = gpu_result.into_host().await?.into_array();
        assert_arrays_eq!(expected, host_result, &mut ctx);
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
