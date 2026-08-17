// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA executor for FSST decompression.

use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::DevicePtr;
use cudarc::driver::DeviceRepr;
use cudarc::driver::PushKernelArg;
use tracing::instrument;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::arrays::Bool;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::arrays::bool::BoolDataParts;
use vortex::array::arrays::primitive::PrimitiveDataParts;
use vortex::array::arrays::varbin::VarBinArraySlotsExt;
use vortex::array::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use vortex::array::arrays::varbinview::build_views::build_views;
use vortex::array::buffer::BufferHandle;
use vortex::array::buffer::DeviceBuffer;
use vortex::array::match_each_integer_ptype;
use vortex::array::match_each_unsigned_integer_ptype;
use vortex::array::validity::Validity;
use vortex::buffer::Alignment;
use vortex::buffer::Buffer;
use vortex::dtype::DType;
use vortex::dtype::NativePType;
use vortex::dtype::PType;
use vortex::encodings::fsst::FSST;
use vortex::encodings::fsst::FSSTArray;
use vortex::encodings::fsst::FSSTArrayExt;
use vortex::encodings::fsst::FSSTArraySlotsExt;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_err;

use crate::CanonicalCudaExt;
use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::arrow::I32Offsets;
use crate::arrow::i32_offsets_from_lengths;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;
use crate::executor::execute_validity_cuda;

/// Device-resident offset-based result of FSST decompression.
pub(crate) struct FSSTVarBin {
    pub(crate) dtype: DType,
    pub(crate) len: usize,
    pub(crate) offsets: BufferHandle,
    pub(crate) values: BufferHandle,
    pub(crate) validity: Validity,
}

/// Returns validity backing bytes and the bit offset of the first row for the FSST kernels.
///
/// `BitBuffer` slices normalize whole-byte offsets but can retain a sub-byte offset. Passing that
/// offset to CUDA lets the kernels address sliced validity directly without repacking it on host.
async fn cuda_validity(
    validity: &Validity,
    len: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<(u64, BufferHandle)> {
    match execute_validity_cuda(validity.clone(), len, ctx).await? {
        Validity::NonNullable | Validity::AllValid => Ok((
            0,
            BufferHandle::new_host(Buffer::from(vec![u8::MAX; len.div_ceil(8)]).into_byte_buffer()),
        )),
        Validity::AllInvalid => Ok((
            0,
            BufferHandle::new_host(Buffer::from(vec![0u8; len.div_ceil(8)]).into_byte_buffer()),
        )),
        Validity::Array(array) => {
            let bool_array = array.try_downcast::<Bool>().map_err(|array| {
                vortex_err!("CUDA validity execution produced {}", array.dtype())
            })?;
            let BoolDataParts { bits, meta } = bool_array.into_data().into_parts(len);
            let bit_offset = meta.offset();
            let byte_len = (bit_offset + len).div_ceil(8);
            Ok((u64::try_from(bit_offset)?, bits.slice(0..byte_len)))
        }
    }
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
            return Ok(Canonical::empty(&dtype));
        }

        if validity.definitely_all_null() {
            let views = ctx.copy_to_device(vec![0i128; fsst.len()])?.await?;
            return Ok(Canonical::VarBinView(unsafe {
                VarBinViewArray::new_handle_unchecked(views, Arc::from([]), dtype, validity)
            }));
        }

        if can_build_i32_offsets(&fsst) {
            decode_fsst_varbinview(fsst, ctx).await
        } else {
            decode_fsst_host_varbinview(fsst, ctx).await
        }
    }
}

/// Decode FSST directly into a device-resident canonical `VarBinView` array.
async fn decode_fsst_varbinview(
    fsst: FSSTArray,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical> {
    let dtype = fsst.dtype().clone();
    let validity = fsst.codes().validity()?;
    let len = fsst.len();
    let lens = fsst
        .uncompressed_lengths()
        .clone()
        .execute_cuda(ctx)
        .await?
        .into_primitive();
    let codes_offsets = fsst
        .codes()
        .offsets()
        .clone()
        .execute_cuda(ctx)
        .await?
        .into_primitive();
    let I32Offsets {
        buffer: output_offsets,
        total: total_size,
    } = i32_offsets_from_lengths(lens, ctx).await?;

    if total_size == 0 {
        let views = ctx.copy_to_device(vec![0i128; len])?.await?;
        return Ok(Canonical::VarBinView(unsafe {
            VarBinViewArray::new_handle_unchecked(views, Arc::from([]), dtype, validity)
        }));
    }

    match_each_unsigned_integer_ptype!(codes_offsets.ptype().to_unsigned(), |U| {
        decode_fsst_varbinview_typed::<U>(fsst, codes_offsets, output_offsets, total_size, ctx)
            .await
    })
}

async fn decode_fsst_varbinview_typed<U>(
    fsst: FSSTArray,
    codes_offsets: PrimitiveArray,
    output_offsets: BufferHandle,
    total_size: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical>
where
    U: NativePType + DeviceRepr + Send + Sync + 'static,
{
    let dtype = fsst.dtype().clone();
    let validity = fsst.codes().validity()?;
    let num_strings = fsst.len();
    let num_strings_u64 = u64::try_from(num_strings)?;
    let symbols_u64 = fsst
        .symbols()
        .iter()
        .map(|symbol| symbol.to_u64())
        .collect::<Vec<_>>();
    let symbol_lengths = fsst.padded_symbol_lengths().slice(0..fsst.n_symbols());
    let codes_bytes_handle = fsst.codes_bytes_handle().clone();
    let PrimitiveDataParts {
        buffer: codes_offsets_buffer,
        ..
    } = codes_offsets.into_data_parts();
    let (validity_bit_offset, validity_bits) = cuda_validity(&validity, num_strings, ctx).await?;

    let symbols = ctx.stream().copy_to_device_sync(&symbols_u64)?;
    let symbol_lengths = ctx.stream().copy_to_device_sync(symbol_lengths.as_ref())?;
    let validity_device = ctx.ensure_on_device_sync(validity_bits)?;
    let (codes_bytes, codes_offsets) = futures::try_join!(
        ctx.ensure_on_device(codes_bytes_handle),
        ctx.ensure_on_device(codes_offsets_buffer),
    )?;

    let mut output = ctx.device_alloc::<u8>(total_size)?;
    let mut views = ctx.device_alloc::<i128>(num_strings)?;
    let codes_bytes_view = codes_bytes.cuda_view::<u8>()?;
    let codes_offsets_view = codes_offsets.cuda_view::<U>()?;
    let symbols_view = symbols.cuda_view::<u64>()?;
    let symbol_lengths_view = symbol_lengths.cuda_view::<u8>()?;
    let output_offsets_view = output_offsets.cuda_view::<i32>()?;
    let validity_view = validity_device.cuda_view::<u8>()?;
    let ptype = U::PTYPE.to_string();
    let cuda_function = ctx.load_function_with_suffixes("fsst", &["varbinview", &ptype])?;

    ctx.launch_kernel(&cuda_function, num_strings, |args| {
        args.arg(&codes_bytes_view)
            .arg(&codes_offsets_view)
            .arg(&symbols_view)
            .arg(&symbol_lengths_view)
            .arg(&output_offsets_view)
            .arg(&validity_view)
            .arg(&validity_bit_offset)
            .arg(&mut output)
            .arg(&mut views)
            .arg(&num_strings_u64);
    })?;

    let views = BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(views)));
    let values = BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(output)));
    Ok(Canonical::VarBinView(unsafe {
        VarBinViewArray::new_handle_unchecked(views, Arc::from([values]), dtype, validity)
    }))
}

/// Decode FSST directly into Arrow-compatible i32 offsets and contiguous values on device.
pub(crate) async fn decode_fsst_varbin(
    fsst: FSSTArray,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<FSSTVarBin> {
    let dtype = fsst.dtype().clone();
    let validity = fsst.codes().validity()?;
    let len = fsst.len();
    let lens = fsst
        .uncompressed_lengths()
        .clone()
        .execute_cuda(ctx)
        .await?
        .into_primitive();
    let codes_offsets = fsst
        .codes()
        .offsets()
        .clone()
        .execute_cuda(ctx)
        .await?
        .into_primitive();
    let I32Offsets {
        buffer: output_offsets,
        total: total_size,
    } = i32_offsets_from_lengths(lens, ctx).await?;

    if total_size == 0 {
        let allocation = CudaDeviceBuffer::new(ctx.device_alloc::<u8>(1)?);
        let values = BufferHandle::new_device(allocation.slice(0..0));
        return Ok(FSSTVarBin {
            dtype,
            len,
            offsets: output_offsets,
            values,
            validity,
        });
    }

    match_each_unsigned_integer_ptype!(codes_offsets.ptype().to_unsigned(), |U| {
        decode_fsst_varbin_typed::<U>(fsst, codes_offsets, output_offsets, total_size, ctx).await
    })
}

async fn decode_fsst_varbin_typed<U>(
    fsst: FSSTArray,
    codes_offsets: PrimitiveArray,
    output_offsets: BufferHandle,
    total_size: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<FSSTVarBin>
where
    U: NativePType + DeviceRepr + Send + Sync + 'static,
{
    let dtype = fsst.dtype().clone();
    let validity = fsst.codes().validity()?;
    let len = fsst.len();
    let len_u64 = len as u64;
    let symbols_u64 = fsst
        .symbols()
        .iter()
        .map(|s| s.to_u64())
        .collect::<Vec<_>>();
    let symbol_lengths = fsst.padded_symbol_lengths().slice(0..fsst.n_symbols());
    let codes_bytes_handle = fsst.codes_bytes_handle().clone();
    let PrimitiveDataParts {
        buffer: codes_offsets_buffer,
        ..
    } = codes_offsets.into_data_parts();
    let (validity_bit_offset, validity_bits) = cuda_validity(&validity, len, ctx).await?;

    let symbols = ctx.stream().copy_to_device_sync(&symbols_u64)?;
    let symbol_lengths = ctx.stream().copy_to_device_sync(symbol_lengths.as_ref())?;
    let validity_device = ctx.ensure_on_device_sync(validity_bits)?;
    let (codes_bytes, codes_offsets) = futures::try_join!(
        ctx.ensure_on_device(codes_bytes_handle),
        ctx.ensure_on_device(codes_offsets_buffer),
    )?;

    // The kernel gates store widths on `out_pos % N` relative to the base, so the base must
    // satisfy the widest store (u128 → 16).
    let mut output = ctx.device_alloc::<u8>(total_size)?;
    let (output_base_ptr, _) = output.device_ptr(ctx.stream());
    assert_eq!(
        output_base_ptr % 16,
        0,
        "output base not 16-aligned: {output_base_ptr:#x}",
    );

    let codes_bytes_view = codes_bytes.cuda_view::<u8>()?;
    let codes_offsets_view = codes_offsets.cuda_view::<U>()?;
    let symbols_view = symbols.cuda_view::<u64>()?;
    let symbol_lengths_view = symbol_lengths.cuda_view::<u8>()?;
    let output_offsets_view = output_offsets.cuda_view::<i32>()?;
    let validity_view = validity_device.cuda_view::<u8>()?;
    let ptype = U::PTYPE.to_string();
    let cuda_function = ctx.load_function_with_suffixes("fsst", &["varbin", &ptype])?;

    ctx.launch_kernel(&cuda_function, len, |args| {
        args.arg(&codes_bytes_view)
            .arg(&codes_offsets_view)
            .arg(&symbols_view)
            .arg(&symbol_lengths_view)
            .arg(&output_offsets_view)
            .arg(&validity_view)
            .arg(&validity_bit_offset)
            .arg(&mut output)
            .arg(&len_u64);
    })?;

    Ok(FSSTVarBin {
        dtype,
        len,
        offsets: output_offsets,
        values: BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(output))),
        validity,
    })
}

fn can_build_i32_offsets(fsst: &FSSTArray) -> bool {
    let max_length = match fsst.uncompressed_lengths().dtype().as_ptype() {
        PType::U8 => u8::MAX as usize,
        PType::U16 => u16::MAX as usize,
        PType::U32 => u32::MAX as usize,
        PType::U64 => usize::MAX,
        _ => return false,
    };
    fsst.len()
        .checked_mul(max_length)
        .is_some_and(|max_total| max_total <= i32::MAX as usize)
}

async fn decode_fsst_host_varbinview(
    fsst: FSSTArray,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical> {
    let lens = fsst
        .uncompressed_lengths()
        .clone()
        .execute_cuda(ctx)
        .await?
        .into_host()
        .await?
        .into_primitive();
    let codes_offsets = fsst
        .codes()
        .offsets()
        .clone()
        .execute_cuda(ctx)
        .await?
        .into_primitive();

    let output_offsets: Vec<u64> = match_each_integer_ptype!(lens.ptype(), |P| {
        let mut out = Vec::with_capacity(lens.len() + 1);
        let mut acc: u64 = 0;
        out.push(0u64);
        #[allow(clippy::unnecessary_cast)]
        for &length in lens.as_slice::<P>() {
            acc += length as u64;
            out.push(acc);
        }
        out
    });

    match_each_unsigned_integer_ptype!(codes_offsets.ptype().to_unsigned(), |U| {
        decode_fsst::<U>(fsst, codes_offsets, lens, output_offsets, ctx).await
    })
}

async fn decode_fsst<U>(
    fsst: FSSTArray,
    codes_offsets: PrimitiveArray,
    lens: PrimitiveArray,
    output_offsets: Vec<u64>,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical>
where
    U: NativePType + DeviceRepr + Send + Sync + 'static,
{
    let dtype = fsst.dtype().clone();
    let validity = fsst.codes().validity()?;
    let num_strings = fsst.len();
    let num_strings_u64 = num_strings as u64;
    let total_size = usize::try_from(
        *output_offsets
            .last()
            .vortex_expect("output_offsets has at least one entry"),
    )
    .vortex_expect("total_size fits in usize");

    if total_size == 0 {
        let views = ctx.copy_to_device(vec![0i128; num_strings])?.await?;
        return Ok(Canonical::VarBinView(unsafe {
            VarBinViewArray::new_handle_unchecked(views, Arc::from([]), dtype, validity)
        }));
    }

    let symbols_u64 = fsst
        .symbols()
        .iter()
        .map(|s| s.to_u64())
        .collect::<Vec<_>>();
    let symbol_lengths = fsst.padded_symbol_lengths().slice(0..fsst.n_symbols());
    let codes_bytes_handle = fsst.codes_bytes_handle().clone();
    let PrimitiveDataParts {
        buffer: codes_offsets_buffer,
        ..
    } = codes_offsets.into_data_parts();

    let (validity_bit_offset, validity_bits) = cuda_validity(&validity, num_strings, ctx).await?;

    let (symbols, symbol_lengths, output_offsets, validity_device, codes_bytes, codes_offsets) = futures::try_join!(
        ctx.copy_to_device(symbols_u64)?,
        ctx.copy_to_device(symbol_lengths)?,
        ctx.copy_to_device(output_offsets)?,
        ctx.ensure_on_device(validity_bits),
        ctx.ensure_on_device(codes_bytes_handle),
        ctx.ensure_on_device(codes_offsets_buffer),
    )?;

    // The kernel checks store alignment relative to the base via
    // `out_pos % N`, so the base must satisfy the widest store (u128 → 16).
    let mut device_output = ctx.device_alloc::<u8>(total_size)?;
    let mut device_views = (total_size <= MAX_BUFFER_LEN)
        .then(|| ctx.device_alloc::<i128>(num_strings))
        .transpose()?;
    let (output_base_ptr, _) = device_output.device_ptr(ctx.stream());
    assert_eq!(
        output_base_ptr % 16,
        0,
        "device_output base not 16-aligned: {output_base_ptr:#x}",
    );

    let codes_bytes_view = codes_bytes.cuda_view::<u8>()?;
    let codes_offsets_view = codes_offsets.cuda_view::<U>()?;
    let symbols_view = symbols.cuda_view::<u64>()?;
    let symbol_lengths_view = symbol_lengths.cuda_view::<u8>()?;
    let output_offsets_view = output_offsets.cuda_view::<u64>()?;
    let validity_view = validity_device.cuda_view::<u8>()?;

    let cuda_function = ctx.load_function("fsst", &[U::PTYPE])?;
    let null_views = 0u64;
    ctx.launch_kernel(&cuda_function, num_strings, |args| {
        args.arg(&codes_bytes_view)
            .arg(&codes_offsets_view)
            .arg(&symbols_view)
            .arg(&symbol_lengths_view)
            .arg(&output_offsets_view)
            .arg(&validity_view)
            .arg(&validity_bit_offset)
            .arg(&mut device_output);
        if let Some(device_views) = device_views.as_mut() {
            args.arg(device_views);
        } else {
            args.arg(&null_views);
        }
        args.arg(&num_strings_u64);
    })?;

    // Fast path: the decoded heap fits in one BinaryView backing buffer, so the kernel wrote
    // views directly and both buffers can remain on-device. Larger heaps use the host rollover
    // path below to split decoded bytes across multiple backing buffers.
    if let Some(device_views) = device_views {
        let views = BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(device_views)));
        let bytes = BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(device_output)));
        return Ok(Canonical::VarBinView(unsafe {
            VarBinViewArray::new_handle_unchecked(views, Arc::from([bytes]), dtype, validity)
        }));
    }

    // BinaryView offsets are u32. Retain the host rollover path for decoded heaps
    // that need multiple backing buffers; ordinary batches stay entirely on-device.
    let host_bytes = CudaDeviceBuffer::new(device_output)
        .copy_to_host(Alignment::new(1))?
        .await?;
    let host_bytes = host_bytes.slice(0..total_size);

    let (buffers, views) = match_each_integer_ptype!(lens.ptype(), |P| {
        build_views(0, MAX_BUFFER_LEN, host_bytes, lens.as_slice::<P>())
    });

    Ok(Canonical::VarBinView(unsafe {
        VarBinViewArray::new_unchecked(views, Arc::from(buffers), dtype, validity)
    }))
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
    use vortex::dtype::DType;
    use vortex::dtype::Nullability;
    use vortex::encodings::fsst::fsst_compress;
    use vortex::encodings::fsst::fsst_train_compressor;
    use vortex::error::VortexExpect;
    use vortex_array::VortexSessionExecute;

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

    #[rstest]
    #[case::binary_non_null(
        vec![Some(&b"the quick brown fox"[..]),
             Some(&b"jumps over the lazy dog"[..]),
             Some(&b"hello world"[..]),
             Some(&b"vortex fsst test string"[..])],
        DType::Binary(Nullability::NonNullable),
    )]
    #[case::utf8_non_null(
        vec![Some(&b"the quick brown fox"[..]),
             Some(&b"jumps over the lazy dog"[..]),
             Some(&b"hello world"[..]),
             Some(&b"vortex fsst test string"[..])],
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
    #[case::binary_all_nulls(
        vec![None, None, None, None, None],
        DType::Binary(Nullability::Nullable),
    )]
    #[crate::test]
    async fn test_cuda_fsst_decompression_roundtrip(
        #[case] strings: Vec<Option<&'static [u8]>>,
        #[case] dtype: DType,
    ) -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create execution context");

        let varbin = VarBinArray::from_iter(strings, dtype.clone()).into_array();
        let compressor = fsst_train_compressor(&varbin, cuda_ctx.execution_ctx())?;
        let fsst_array =
            fsst_compress(&varbin, &compressor, cuda_ctx.execution_ctx())?.into_array();

        let gpu_result = FSSTExecutor
            .execute(fsst_array.clone(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed");
        assert_eq!(gpu_result.dtype(), &dtype);
        assert_device_resident(&gpu_result);

        let host_result = gpu_result.into_host().await?.into_array();
        assert_arrays_eq!(fsst_array, host_result, &mut ctx);
        Ok(())
    }

    /// Verifies that the view kernel applies a sliced validity bitmap's nonzero bit offset.
    #[crate::test]
    async fn test_cuda_fsst_decompression_sliced_validity() -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create execution context");
        let values = [
            Some(&b"before"[..]),
            None,
            Some(&b"gamma"[..]),
            None,
            Some(&b"after"[..]),
        ];
        let varbin =
            VarBinArray::from_iter(values, DType::Utf8(Nullability::Nullable)).into_array();
        let compressor = fsst_train_compressor(&varbin, cuda_ctx.execution_ctx())?;
        let fsst = fsst_compress(&varbin, &compressor, cuda_ctx.execution_ctx())?.into_array();
        let sliced = fsst.slice(1..4)?;

        let gpu_result = FSSTExecutor.execute(sliced.clone(), &mut cuda_ctx).await?;
        assert_device_resident(&gpu_result);
        let host_result = gpu_result.into_host().await?.into_array();
        assert_arrays_eq!(sliced, host_result, &mut ctx);
        Ok(())
    }

    #[crate::test]
    async fn test_cuda_fsst_direct_varbin_output() -> VortexResult<()> {
        let mut cuda_ctx = cuda_ctx_with_varbin_layout(VarBinExportLayout::VarBin)?;
        let values: [&[u8]; 3] = [
            b"",
            b"short",
            b"this value is stored directly in the values buffer",
        ];
        let varbin = VarBinArray::from_iter(
            values.into_iter().map(Some),
            DType::Utf8(Nullability::NonNullable),
        )
        .into_array();
        let compressor = fsst_train_compressor(&varbin, cuda_ctx.execution_ctx())?;
        let fsst = fsst_compress(&varbin, &compressor, cuda_ctx.execution_ctx())?;

        let output = decode_fsst_varbin(fsst, &mut cuda_ctx).await?;
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
    async fn test_cuda_fsst_arrow_export_uses_dtype_layout(
        #[case] dtype: DType,
        #[case] layout: VarBinExportLayout,
        #[case] expected_data_type: DataType,
        #[case] expected_n_buffers: i64,
    ) -> VortexResult<()> {
        let mut cuda_ctx = cuda_ctx_with_varbin_layout(layout)?;
        let values = [
            Some(&b"short"[..]),
            Some(&b"this value is stored out of line"[..]),
        ];
        let varbin = VarBinArray::from_iter(values, dtype).into_array();
        let compressor = fsst_train_compressor(&varbin, cuda_ctx.execution_ctx())?;
        let fsst_array =
            fsst_compress(&varbin, &compressor, cuda_ctx.execution_ctx())?.into_array();

        let mut exported = fsst_array
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

    /// Exercises the multi-block grid-stride path on a larger dataset.
    #[crate::test]
    async fn test_cuda_fsst_decompression_roundtrip_large() -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        use vortex_fsst::test_utils::make_fsst_clickbench_urls;

        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create execution context");

        let fsst_array = make_fsst_clickbench_urls(100_000, cuda_ctx.execution_ctx()).into_array();

        let gpu_result = FSSTExecutor
            .execute(fsst_array.clone(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed");
        assert_device_resident(&gpu_result);

        let host_result = gpu_result.into_host().await?.into_array();
        assert_arrays_eq!(fsst_array, host_result, &mut ctx);
        Ok(())
    }
}
