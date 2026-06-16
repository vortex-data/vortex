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
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::arrays::primitive::PrimitiveDataParts;
use vortex::array::arrays::varbin::VarBinArrayExt;
use vortex::array::arrays::varbinview::BinaryView;
use vortex::array::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use vortex::array::arrays::varbinview::build_views::build_views;
use vortex::array::buffer::DeviceBuffer;
use vortex::array::match_each_integer_ptype;
use vortex::array::match_each_unsigned_integer_ptype;
use vortex::buffer::Alignment;
use vortex::buffer::Buffer;
use vortex::dtype::NativePType;
use vortex::encodings::fsst::FSST;
use vortex::encodings::fsst::FSSTArray;
use vortex::encodings::fsst::FSSTArrayExt;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_err;

use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;

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

        if fsst.is_empty() || validity.definitely_all_null() {
            let empty = unsafe {
                VarBinViewArray::new_unchecked(
                    Buffer::<BinaryView>::zeroed(fsst.len()),
                    Arc::from([]),
                    dtype,
                    validity,
                )
            };
            return Ok(Canonical::VarBinView(empty));
        }

        let lens = fsst
            .uncompressed_lengths()
            .clone()
            .execute::<PrimitiveArray>(ctx.execution_ctx())?;
        let codes_offsets = fsst
            .codes()
            .offsets()
            .clone()
            .execute::<PrimitiveArray>(ctx.execution_ctx())?;

        // Prefix-sum lens to per-string u64 output offsets so the kernel
        // knows where to write each decoded string.
        let output_offsets: Vec<u64> = match_each_integer_ptype!(lens.ptype(), |P| {
            let mut out = Vec::with_capacity(lens.len() + 1);
            let mut acc: u64 = 0;
            out.push(0u64);
            #[allow(clippy::unnecessary_cast)]
            for &l in lens.as_slice::<P>() {
                acc += l as u64;
                out.push(acc);
            }
            out
        });

        // Dispatch on the unsigned width; signed and unsigned offsets of the
        // same width share an identical byte representation.
        match_each_unsigned_integer_ptype!(codes_offsets.ptype().to_unsigned(), |U| {
            decode_fsst::<U>(fsst, codes_offsets, lens, output_offsets, ctx).await
        })
    }
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

    let symbols_u64: Vec<u64> = fsst.symbols().iter().map(|s| s.to_u64()).collect();
    let symbol_lengths = fsst.symbol_lengths().clone();
    let codes_bytes_handle = fsst.codes_bytes_handle().clone();
    let PrimitiveDataParts {
        buffer: codes_offsets_buffer,
        ..
    } = codes_offsets.into_data_parts();

    let (.., validity_bits) = validity
        .clone()
        .execute_mask(num_strings, ctx.execution_ctx())?
        .into_bit_buffer()
        .sliced()
        .into_inner();

    let (symbols, symbol_lengths, output_offsets, validity_device, codes_bytes, codes_offsets) = futures::try_join!(
        ctx.copy_to_device(symbols_u64)?,
        ctx.copy_to_device(symbol_lengths)?,
        ctx.copy_to_device(output_offsets)?,
        ctx.copy_to_device(validity_bits.to_vec())?,
        ctx.ensure_on_device(codes_bytes_handle),
        ctx.ensure_on_device(codes_offsets_buffer),
    )?;

    // The kernel checks store alignment relative to the base via
    // `out_pos % N`, so the base must satisfy the widest store (u128 → 16).
    let device_output = ctx.device_alloc::<u8>(total_size)?;
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
    ctx.launch_kernel(&cuda_function, num_strings, |args| {
        args.arg(&codes_bytes_view)
            .arg(&codes_offsets_view)
            .arg(&symbols_view)
            .arg(&symbol_lengths_view)
            .arg(&output_offsets_view)
            .arg(&validity_view)
            .arg(&device_output)
            .arg(&num_strings_u64);
    })?;

    let host_bytes = CudaDeviceBuffer::new(device_output)
        .copy_to_host(Alignment::new(1))?
        .await?;
    let host_bytes = host_bytes.slice(0..total_size);

    let (buffers, views) = match_each_integer_ptype!(lens.ptype(), |P| {
        build_views(
            0,
            MAX_BUFFER_LEN,
            host_bytes.into_mut(),
            lens.as_slice::<P>(),
        )
    });

    Ok(Canonical::VarBinView(unsafe {
        VarBinViewArray::new_unchecked(views, Arc::from(buffers), dtype, validity)
    }))
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

    /// Exercises the multi-block grid-stride path on a larger dataset.
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
