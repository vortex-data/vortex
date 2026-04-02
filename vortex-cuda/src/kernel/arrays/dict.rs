// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::DeviceRepr;
use cudarc::driver::PushKernelArg;
use tracing::instrument;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::arrays::DecimalArray;
use vortex::array::arrays::Dict;
use vortex::array::arrays::DictArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::arrays::decimal::DecimalArrayParts;
use vortex::array::arrays::dict::DictArrayParts;
use vortex::array::arrays::primitive::PrimitiveArrayParts;
use vortex::array::arrays::varbinview::VarBinViewArrayParts;
use vortex::array::buffer::BufferHandle;
use vortex::array::match_each_decimal_value_type;
use vortex::array::match_each_integer_ptype;
use vortex::array::match_each_native_simd_ptype;
use vortex::dtype::DType;
use vortex::dtype::NativeDecimalType;
use vortex::dtype::NativePType;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;

use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;

/// CUDA executor for dictionary-encoded arrays.
#[derive(Debug)]
pub(crate) struct DictExecutor;

#[async_trait]
impl CudaExecute for DictExecutor {
    #[instrument(level = "trace", skip_all, fields(executor = ?self))]
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let dict_array = array
            .try_into::<Dict>()
            .ok()
            .vortex_expect("Array is not a Dict array");

        let values_dtype = dict_array.values().dtype().clone();
        match &values_dtype {
            DType::Decimal(..) => execute_dict_decimal(dict_array, ctx).await,
            DType::Primitive(..) => execute_dict_prim(dict_array, ctx).await,
            DType::Utf8(..) | DType::Binary(..) => execute_dict_varbinview(dict_array, ctx).await,
            dt => vortex_bail!("unsupported decompress for DType={dt}"),
        }
    }
}

#[expect(clippy::cognitive_complexity)]
async fn execute_dict_prim(dict: DictArray, ctx: &mut CudaExecutionCtx) -> VortexResult<Canonical> {
    let DictArrayParts { values, codes, .. } = dict.into_data().into_parts();

    // Execute both children to get them as primitives on the device
    let values_canonical = values.execute_cuda(ctx).await?;
    let codes_canonical = codes.execute_cuda(ctx).await?;

    let values_prim = values_canonical.into_primitive();
    let codes_prim = codes_canonical.into_primitive();

    let values_ptype = values_prim.ptype();
    let codes_ptype = codes_prim.ptype();

    // Dispatch based on both value type and code type
    match_each_native_simd_ptype!(values_ptype, |V| {
        match_each_integer_ptype!(codes_ptype, |I| {
            execute_dict_prim_typed::<V, I>(values_prim, codes_prim, ctx).await
        })
    })
}

async fn execute_dict_prim_typed<V: DeviceRepr + NativePType, I: DeviceRepr + NativePType>(
    values: PrimitiveArray,
    codes: PrimitiveArray,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical> {
    assert!(!codes.is_empty());
    let codes_len = codes.len();

    let PrimitiveArrayParts {
        ptype: value_ptype,
        buffer: values_buffer,
        validity: values_validity,
        ..
    } = values.into_data().into_parts();
    let output_validity = values_validity.take(&codes.clone().into_array())?;
    let PrimitiveArrayParts {
        buffer: codes_buffer,
        ..
    } = codes.into_data().into_parts();

    // Get device buffers for values and codes
    let values_device = ctx.ensure_on_device(values_buffer).await?;
    let codes_device = ctx.ensure_on_device(codes_buffer).await?;

    // Allocate output buffer on device
    let output_slice = ctx.device_alloc::<V>(codes_len)?;
    let output_device = CudaDeviceBuffer::new(output_slice);

    // Get views for kernel launch
    let values_view = values_device.cuda_view::<V>()?;
    let codes_view = codes_device.cuda_view::<I>()?;
    let output_view = output_device.as_view::<V>();

    let codes_len_u64 = codes_len as u64;

    let kernel_function = ctx.load_function("dict", &[value_ptype, I::PTYPE])?;
    ctx.launch_kernel(&kernel_function, codes_len, |args| {
        args.arg(&codes_view)
            .arg(&codes_len_u64)
            .arg(&values_view)
            .arg(&output_view);
    })?;

    Ok(Canonical::Primitive(PrimitiveArray::from_buffer_handle(
        BufferHandle::new_device(Arc::new(output_device)),
        value_ptype,
        output_validity,
    )))
}

/// Execute dict array decompression for decimal types (i128/i256).
///
/// These types don't have a `PType` so we need to handle them separately.
#[expect(clippy::cognitive_complexity)]
async fn execute_dict_decimal(
    dict: DictArray,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical> {
    let DictArrayParts {
        values,
        codes,
        dtype,
        ..
    } = dict.into_data().into_parts();

    // Execute codes to get them as primitives on the device
    let codes_prim = codes.execute_cuda(ctx).await?.into_primitive();
    let codes_ptype = codes_prim.ptype();

    // For decimal values, execute recursively to handle any nested encodings
    let values_decimal = values.execute_cuda(ctx).await?.into_decimal();
    let decimal_type = values_decimal.values_type();

    match_each_decimal_value_type!(decimal_type, |V| {
        match_each_integer_ptype!(codes_ptype, |C| {
            execute_dict_decimal_typed::<V, C>(values_decimal, codes_prim, dtype, ctx).await
        })
    })
}

/// Type-parameterized decimal dict execution for a specific code type.
async fn execute_dict_decimal_typed<
    V: DeviceRepr + NativeDecimalType,
    C: DeviceRepr + NativePType,
>(
    values: DecimalArray,
    codes: PrimitiveArray,
    output_dtype: DType,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical> {
    assert!(!codes.is_empty());
    let codes_len = codes.len();
    let codes_len_u64 = codes_len as u64;

    let DecimalArrayParts {
        values: values_buffer,
        validity: values_validity,
        ..
    } = values.into_data().into_parts();
    let output_validity = values_validity.take(&codes.clone().into_array())?;

    let PrimitiveArrayParts {
        buffer: codes_buffer,
        ..
    } = codes.into_data().into_parts();

    // Copy buffers to device if needed
    let values_device = ctx.ensure_on_device(values_buffer).await?;
    let codes_device = ctx.ensure_on_device(codes_buffer).await?;

    // Allocate output buffer on device (codes_len * value_byte_width bytes)
    let output_slice = ctx.device_alloc::<V>(codes_len)?;
    let output_device = CudaDeviceBuffer::new(output_slice);

    // Get views for kernel launch
    let values_view = values_device.cuda_view::<V>()?;
    let codes_view = codes_device.cuda_view::<C>()?;
    let output_view = output_device.as_view::<V>();

    // Load kernel function using string suffixes
    let cuda_function = ctx.load_function_with_suffixes(
        "dict",
        &[&V::DECIMAL_TYPE.to_string(), &C::PTYPE.to_string()],
    )?;

    ctx.launch_kernel(&cuda_function, codes_len, |args| {
        args.arg(&codes_view)
            .arg(&codes_len_u64)
            .arg(&values_view)
            .arg(&output_view);
    })?;

    Ok(Canonical::Decimal(DecimalArray::new_handle(
        BufferHandle::new_device(Arc::new(output_device)),
        V::DECIMAL_TYPE,
        output_dtype.into_decimal_opt().vortex_expect("is decimal"),
        output_validity,
    )))
}

/// Dictionary array decompression for string (UTF-8/Binary) values.
///
/// Reinterprets the dictionary's `BinaryView` buffer as `i128` values and gathers
/// them by code index. Both inlined (≤ 12 bytes) and outlined (> 12 bytes) views
/// are supported. For outlined views, the output shares the values' data buffers.
async fn execute_dict_varbinview(
    dict: DictArray,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical> {
    let DictArrayParts {
        values,
        codes,
        dtype,
        ..
    } = dict.into_data().into_parts();

    let codes_prim = codes.execute_cuda(ctx).await?.into_primitive();
    let codes_ptype = codes_prim.ptype();
    let codes_len = codes_prim.len();
    let values_vbv = values.execute_cuda(ctx).await?.into_varbinview();

    let VarBinViewArrayParts {
        views: values_views_handle,
        buffers: values_data_buffers,
        validity: values_validity,
        ..
    } = values_vbv.into_data().into_parts();
    let output_validity = values_validity.take(&codes_prim.clone().into_array())?;

    let PrimitiveArrayParts {
        buffer: codes_buffer,
        ..
    } = codes_prim.into_data().into_parts();

    // Move buffers to device if needed.
    let values_device = ctx.ensure_on_device(values_views_handle).await?;
    let codes_device = ctx.ensure_on_device(codes_buffer).await?;

    // Allocate output: one i128 per code.
    let output_slice = ctx.device_alloc::<i128>(codes_len)?;
    let output_device = CudaDeviceBuffer::new(output_slice);

    // Dispatch by code type, reusing the existing dict_i128_<code_type> kernel.
    // BinaryView is repr(C, align(16)) and 16 bytes — identical layout to i128.
    match_each_integer_ptype!(codes_ptype, |C| {
        let values_view = values_device.cuda_view::<i128>()?;
        let codes_view = codes_device.cuda_view::<C>()?;
        let output_view = output_device.as_view::<i128>();

        let codes_ptype_str = C::PTYPE.to_string();
        let cuda_function = ctx.load_function_with_suffixes("dict", &["i128", &codes_ptype_str])?;

        let codes_len_u64 = codes_len as u64;

        ctx.launch_kernel(&cuda_function, codes_len, |args| {
            args.arg(&codes_view);
            args.arg(&codes_len_u64);
            args.arg(&values_view);
            args.arg(&output_view);
        })?;
    });

    // Output views gathered by the kernel share the values' data buffers.
    // Outlined views reference into these buffers via buffer_index + offset,
    // and inlined views are self-contained within the 16-byte view.
    Ok(Canonical::VarBinView(unsafe {
        VarBinViewArray::new_handle_unchecked(
            BufferHandle::new_device(Arc::new(output_device)),
            values_data_buffers,
            dtype,
            output_validity,
        )
    }))
}

#[cfg(test)]
mod tests {
    use vortex::array::IntoArray;
    use vortex::array::arrays::DecimalArray;
    use vortex::array::arrays::DictArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::VarBinViewArray;
    use vortex::array::assert_arrays_eq;
    use vortex::array::validity::Validity::NonNullable;
    use vortex::buffer::Buffer;
    use vortex::dtype::DecimalDType;
    use vortex::dtype::i256;
    use vortex::error::VortexExpect;
    use vortex::session::VortexSession;

    use super::*;
    use crate::CanonicalCudaExt;
    use crate::session::CudaSession;

    /// Copy a CUDA primitive array result to host memory.
    fn cuda_primitive_to_host(prim: PrimitiveArray) -> VortexResult<PrimitiveArray> {
        Ok(PrimitiveArray::from_byte_buffer(
            prim.buffer_handle().try_to_host_sync()?,
            prim.ptype(),
            prim.validity(),
        ))
    }

    #[crate::test]
    async fn test_cuda_dict_u32_values_u8_codes() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Dictionary values: [100, 200, 300, 400]
        let values = PrimitiveArray::new(Buffer::from(vec![100u32, 200, 300, 400]), NonNullable);

        // Codes: indices into the values array
        let codes: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3, 2, 2, 1, 0];
        let codes_array = PrimitiveArray::new(Buffer::from(codes), NonNullable);

        let dict_array = DictArray::try_new(codes_array.into_array(), values.into_array())
            .vortex_expect("failed to create Dict array");

        // Get baseline from CPU canonicalization
        let baseline = dict_array.to_canonical()?;

        // Execute on CUDA
        let cuda_result = DictExecutor
            .execute(dict_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_primitive();

        let cuda_result = cuda_primitive_to_host(cuda_result)?;

        // Compare CUDA result with baseline
        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[crate::test]
    async fn test_cuda_dict_u64_values_u16_codes() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Dictionary values: large u64 values
        let values = PrimitiveArray::new(
            Buffer::from(vec![1000000u64, 2000000, 3000000, 4000000, 5000000]),
            NonNullable,
        );

        // Codes: indices into the values array (using u16)
        let codes: Vec<u16> = vec![4, 3, 2, 1, 0, 0, 1, 2, 3, 4];
        let codes_array = PrimitiveArray::new(Buffer::from(codes), NonNullable);

        let dict_array = DictArray::try_new(codes_array.into_array(), values.into_array())
            .vortex_expect("failed to create Dict array");

        // Get baseline from CPU canonicalization
        let baseline = dict_array.to_canonical()?;

        // Execute on CUDA
        let cuda_result = DictExecutor
            .execute(dict_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_primitive();

        let cuda_result = cuda_primitive_to_host(cuda_result)?;

        // Compare CUDA result with baseline
        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[crate::test]
    async fn test_cuda_dict_i32_values_u32_codes() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Dictionary values: signed integers including negatives
        let values = PrimitiveArray::new(Buffer::from(vec![-100i32, 0, 100, 200]), NonNullable);

        // Codes using u32
        let codes: Vec<u32> = vec![0, 1, 2, 3, 3, 2, 1, 0];
        let codes_array = PrimitiveArray::new(Buffer::from(codes), NonNullable);

        let dict_array = DictArray::try_new(codes_array.into_array(), values.into_array())
            .vortex_expect("failed to create Dict array");

        // Get baseline from CPU canonicalization
        let baseline = dict_array.to_canonical()?;

        // Execute on CUDA
        let cuda_result = DictExecutor
            .execute(dict_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_primitive();
        let cuda_result = cuda_primitive_to_host(cuda_result)?;

        // Compare CUDA result with baseline
        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[crate::test]
    async fn test_cuda_dict_large_array() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Dictionary with 256 values
        let values: Vec<u32> = (0..256).map(|i| i * 1000).collect();
        let values_array = PrimitiveArray::new(Buffer::from(values), NonNullable);

        let codes: Vec<u16> = (0..2050).map(|i| (i % 256) as u16).collect();
        let codes_array = PrimitiveArray::new(Buffer::from(codes), NonNullable);

        let dict_array = DictArray::try_new(codes_array.into_array(), values_array.into_array())
            .vortex_expect("failed to create Dict array");

        // Get baseline from CPU canonicalization
        let baseline = dict_array.to_canonical()?;

        // Execute on CUDA
        let cuda_result = DictExecutor
            .execute(dict_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_primitive();

        let cuda_result = cuda_primitive_to_host(cuda_result)?;

        // Compare CUDA result with baseline
        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[crate::test]
    async fn test_cuda_dict_values_with_validity() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Dictionary values with nulls: [100, null, 300, 400]
        let values =
            PrimitiveArray::from_option_iter(vec![Some(100u32), None, Some(300), Some(400)]);

        // Codes: indices into the values array (code 1 points to null)
        let codes: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let codes_array = PrimitiveArray::new(Buffer::from(codes), NonNullable);

        let dict_array = DictArray::try_new(codes_array.into_array(), values.into_array())
            .vortex_expect("failed to create Dict array");

        // Get baseline from CPU canonicalization
        let baseline = dict_array.to_canonical()?;

        let cuda_result = DictExecutor
            .execute(dict_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_primitive();

        let cuda_result = cuda_primitive_to_host(cuda_result)?;

        // Compare CUDA result with baseline
        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[crate::test]
    async fn test_cuda_dict_codes_with_validity() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Dictionary values: [100, 200, 300, 400]
        let values = PrimitiveArray::new(Buffer::from(vec![100u32, 200, 300, 400]), NonNullable);

        // Codes with nulls: [0, null, 2, null, 0, 1]
        let codes = PrimitiveArray::from_option_iter(vec![
            Some(0u8),
            None,
            Some(2),
            None,
            Some(0),
            Some(1),
        ]);

        let dict_array = DictArray::try_new(codes.into_array(), values.into_array())
            .vortex_expect("failed to create Dict array");

        // Get baseline from CPU canonicalization
        let baseline = dict_array.to_canonical()?;

        // Execute on CUDA
        let cuda_result = DictExecutor
            .execute(dict_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_primitive();
        let cuda_result = cuda_primitive_to_host(cuda_result)?;

        // Compare CUDA result with baseline
        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[crate::test]
    async fn test_cuda_dict_both_with_validity() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Dictionary values with nulls: [100, null, 300, 400]
        let values =
            PrimitiveArray::from_option_iter(vec![Some(100u32), None, Some(300), Some(400)]);

        // Codes with nulls: [0, null, 1, 2, null, 3]
        // Position 0: code=0 -> value=100 (valid)
        // Position 1: code=null -> output=null
        // Position 2: code=1 -> value=null -> output=null
        // Position 3: code=2 -> value=300 (valid)
        // Position 4: code=null -> output=null
        // Position 5: code=3 -> value=400 (valid)
        let codes = PrimitiveArray::from_option_iter(vec![
            Some(0u8),
            None,
            Some(1),
            Some(2),
            None,
            Some(3),
        ]);

        let dict_array = DictArray::try_new(codes.into_array(), values.into_array())
            .vortex_expect("failed to create Dict array");

        // Get baseline from CPU canonicalization
        let baseline = dict_array.to_canonical()?;

        // Execute on CUDA
        let cuda_result = DictExecutor
            .execute(dict_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_primitive();
        let cuda_result = cuda_primitive_to_host(cuda_result)?;

        // Compare CUDA result with baseline
        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[crate::test]
    async fn test_cuda_dict_i64_values_with_validity() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Dictionary values with nulls (i64)
        let values = PrimitiveArray::from_option_iter(vec![
            Some(-1000i64),
            None,
            Some(2000),
            None,
            Some(4000),
        ]);

        // Codes with nulls (u16)
        let codes = PrimitiveArray::from_option_iter(vec![
            Some(0u16),
            Some(1),
            None,
            Some(2),
            Some(3),
            None,
            Some(4),
            Some(0),
        ]);

        let dict_array = DictArray::try_new(codes.into_array(), values.into_array())
            .vortex_expect("failed to create Dict array");

        // Get baseline from CPU canonicalization
        let baseline = dict_array.to_canonical()?;

        // Execute on CUDA
        let cuda_result = DictExecutor
            .execute(dict_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_primitive();
        let cuda_result = cuda_primitive_to_host(cuda_result)?;

        // Compare CUDA result with baseline
        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[crate::test]
    async fn test_cuda_dict_all_valid_matches_baseline() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Non-nullable values
        let values = PrimitiveArray::new(Buffer::from(vec![10u32, 20, 30, 40, 50]), NonNullable);

        // Non-nullable codes
        let codes = PrimitiveArray::new(
            Buffer::from(vec![0u8, 1, 2, 3, 4, 4, 3, 2, 1, 0]),
            NonNullable,
        );

        let dict_array = DictArray::try_new(codes.into_array(), values.into_array())
            .vortex_expect("failed to create Dict array");

        // Get baseline from CPU canonicalization
        let baseline = dict_array.to_canonical()?;

        // Execute on CUDA
        let cuda_result = DictExecutor
            .execute(dict_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_primitive();
        let cuda_result = cuda_primitive_to_host(cuda_result)?;

        // Compare CUDA result with baseline
        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    /// Helper to copy CUDA decimal array result to host memory.
    fn cuda_decimal_to_host(decimal: DecimalArray) -> VortexResult<DecimalArray> {
        Ok(DecimalArray::new_handle(
            BufferHandle::new_host(decimal.buffer_handle().try_to_host_sync()?),
            decimal.values_type(),
            decimal.decimal_dtype(),
            decimal.validity(),
        ))
    }

    #[crate::test]
    async fn test_cuda_dict_decimal_i8_values() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Precision 2 uses i8 backing type
        let decimal_dtype = DecimalDType::new(2, 1);
        let values = DecimalArray::from_iter([10i8, 20, 30, 40], decimal_dtype);

        let codes: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3, 2, 2, 1, 0];
        let codes_array = PrimitiveArray::new(Buffer::from(codes), NonNullable);

        let dict_array = DictArray::try_new(codes_array.into_array(), values.into_array())
            .vortex_expect("failed to create Dict array");

        let baseline = dict_array.to_canonical()?;

        let cuda_result = DictExecutor
            .execute(dict_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_decimal();
        let cuda_result = cuda_decimal_to_host(cuda_result)?;

        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[crate::test]
    async fn test_cuda_dict_decimal_i16_values() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Precision 4 uses i16 backing type
        let decimal_dtype = DecimalDType::new(4, 2);
        let values = DecimalArray::from_iter([1000i16, 2000, 3000, 4000], decimal_dtype);

        let codes: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3, 2, 2, 1, 0];
        let codes_array = PrimitiveArray::new(Buffer::from(codes), NonNullable);

        let dict_array = DictArray::try_new(codes_array.into_array(), values.into_array())
            .vortex_expect("failed to create Dict array");

        let baseline = dict_array.to_canonical()?;

        let cuda_result = DictExecutor
            .execute(dict_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_decimal();
        let cuda_result = cuda_decimal_to_host(cuda_result)?;

        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[crate::test]
    async fn test_cuda_dict_decimal_i32_values() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Precision 9 uses i32 backing type
        let decimal_dtype = DecimalDType::new(9, 4);
        let values = DecimalArray::from_iter([100000i32, 200000, 300000, 400000], decimal_dtype);

        let codes: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3, 2, 2, 1, 0];
        let codes_array = PrimitiveArray::new(Buffer::from(codes), NonNullable);

        let dict_array = DictArray::try_new(codes_array.into_array(), values.into_array())
            .vortex_expect("failed to create Dict array");

        let baseline = dict_array.to_canonical()?;

        let cuda_result = DictExecutor
            .execute(dict_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_decimal();
        let cuda_result = cuda_decimal_to_host(cuda_result)?;

        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[crate::test]
    async fn test_cuda_dict_decimal_i64_values() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Precision 18 uses i64 backing type
        let decimal_dtype = DecimalDType::new(18, 6);
        let values = DecimalArray::from_iter(
            [1000000000i64, 2000000000, 3000000000, 4000000000],
            decimal_dtype,
        );

        let codes: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3, 2, 2, 1, 0];
        let codes_array = PrimitiveArray::new(Buffer::from(codes), NonNullable);

        let dict_array = DictArray::try_new(codes_array.into_array(), values.into_array())
            .vortex_expect("failed to create Dict array");

        let baseline = dict_array.to_canonical()?;

        let cuda_result = DictExecutor
            .execute(dict_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_decimal();
        let cuda_result = cuda_decimal_to_host(cuda_result)?;

        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[crate::test]
    async fn test_cuda_dict_decimal_i128_values() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Precision 38 uses i128 backing type
        let decimal_dtype = DecimalDType::new(38, 10);
        let values = DecimalArray::from_iter(
            [
                10000000000000000000i128,
                20000000000000000000,
                30000000000000000000,
                40000000000000000000,
            ],
            decimal_dtype,
        );

        let codes: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3, 2, 2, 1, 0];
        let codes_array = PrimitiveArray::new(Buffer::from(codes), NonNullable);

        let dict_array = DictArray::try_new(codes_array.into_array(), values.into_array())
            .vortex_expect("failed to create Dict array");

        let baseline = dict_array.to_canonical()?;

        let cuda_result = DictExecutor
            .execute(dict_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_decimal();
        let cuda_result = cuda_decimal_to_host(cuda_result)?;

        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    /// Helper to copy a CUDA VarBinViewArray result to host memory.
    async fn cuda_varbinview_to_host(vbv: VarBinViewArray) -> VortexResult<VarBinViewArray> {
        Ok(Canonical::VarBinView(vbv)
            .into_host()
            .await?
            .into_varbinview())
    }

    #[crate::test]
    async fn test_cuda_dict_string_values_u8_codes() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let values = VarBinViewArray::from_iter_str(["cat", "dog", "bird", "fish"]);
        let codes: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3, 2, 2, 1, 0];
        let codes_array = PrimitiveArray::new(Buffer::from(codes), NonNullable);

        let dict_array = DictArray::try_new(codes_array.into_array(), values.into_array())
            .vortex_expect("failed to create Dict array");

        let baseline = dict_array.to_canonical()?;

        let cuda_result = DictExecutor
            .execute(dict_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_varbinview();
        let cuda_result = cuda_varbinview_to_host(cuda_result).await?;

        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[crate::test]
    async fn test_cuda_dict_string_values_u16_codes() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let values = VarBinViewArray::from_iter_str(["alpha", "beta", "gamma", "delta", "epsilon"]);
        let codes: Vec<u16> = vec![4, 3, 2, 1, 0, 0, 1, 2, 3, 4];
        let codes_array = PrimitiveArray::new(Buffer::from(codes), NonNullable);

        let dict_array = DictArray::try_new(codes_array.into_array(), values.into_array())
            .vortex_expect("failed to create Dict array");

        let baseline = dict_array.to_canonical()?;

        let cuda_result = DictExecutor
            .execute(dict_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_varbinview();
        let cuda_result = cuda_varbinview_to_host(cuda_result).await?;

        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[crate::test]
    async fn test_cuda_dict_string_max_inlined_12_bytes() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Exactly 12 bytes — the maximum inlined BinaryView size
        let values =
            VarBinViewArray::from_iter_str(["abcdefghijkl", "123456789012", "xxxxyyyyzzzz"]);
        let codes: Vec<u8> = vec![0, 1, 2, 2, 1, 0, 0, 2];
        let codes_array = PrimitiveArray::new(Buffer::from(codes), NonNullable);

        let dict_array = DictArray::try_new(codes_array.into_array(), values.into_array())
            .vortex_expect("failed to create Dict array");

        let baseline = dict_array.to_canonical()?;

        let cuda_result = DictExecutor
            .execute(dict_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_varbinview();
        let cuda_result = cuda_varbinview_to_host(cuda_result).await?;

        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[crate::test]
    async fn test_cuda_dict_string_outlined_views() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // 13+ bytes — outlined BinaryViews that reference data buffers
        let values = VarBinViewArray::from_iter_str([
            "short",
            "this_is_a_longer_string_that_is_outlined",
            "another_outlined_string_value",
        ]);
        let codes: Vec<u8> = vec![0, 1, 2, 1, 0, 2, 1];
        let codes_array = PrimitiveArray::new(Buffer::from(codes), NonNullable);

        let dict_array = DictArray::try_new(codes_array.into_array(), values.into_array())
            .vortex_expect("failed to create Dict array");

        let baseline = dict_array.to_canonical()?;

        let cuda_result = DictExecutor
            .execute(dict_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_varbinview();
        let cuda_result = cuda_varbinview_to_host(cuda_result).await?;

        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[crate::test]
    async fn test_cuda_dict_string_empty_strings() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let values = VarBinViewArray::from_iter_str(["", "a", ""]);
        let codes: Vec<u8> = vec![0, 1, 2, 0, 1, 2];
        let codes_array = PrimitiveArray::new(Buffer::from(codes), NonNullable);

        let dict_array = DictArray::try_new(codes_array.into_array(), values.into_array())
            .vortex_expect("failed to create Dict array");

        let baseline = dict_array.to_canonical()?;

        let cuda_result = DictExecutor
            .execute(dict_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_varbinview();
        let cuda_result = cuda_varbinview_to_host(cuda_result).await?;

        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[crate::test]
    async fn test_cuda_dict_string_values_with_validity() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let values = VarBinViewArray::from_iter_nullable_str([Some("hello"), None, Some("world")]);

        let codes: Vec<u8> = vec![0, 1, 2, 0, 1, 2];
        let codes_array = PrimitiveArray::new(Buffer::from(codes), NonNullable);

        let dict_array = DictArray::try_new(codes_array.into_array(), values.into_array())
            .vortex_expect("failed to create Dict array");

        let baseline = dict_array.to_canonical()?;

        let cuda_result = DictExecutor
            .execute(dict_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_varbinview();
        let cuda_result = cuda_varbinview_to_host(cuda_result).await?;

        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[crate::test]
    async fn test_cuda_dict_string_outlined_with_validity() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Mix of inlined, outlined, and null dictionary values
        let values = VarBinViewArray::from_iter_nullable_str([
            Some("short"),
            None,
            Some("a_very_long_outlined_string_value_here"),
            Some("another_long_outlined_string"),
        ]);

        // Codes referencing all value indices including the null
        let codes: Vec<u8> = vec![0, 1, 2, 3, 0, 2, 1, 3];
        let codes_array = PrimitiveArray::new(Buffer::from(codes), NonNullable);

        let dict_array = DictArray::try_new(codes_array.into_array(), values.into_array())
            .vortex_expect("failed to create Dict array");

        let baseline = dict_array.to_canonical()?;

        let cuda_result = DictExecutor
            .execute(dict_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_varbinview();
        let cuda_result = cuda_varbinview_to_host(cuda_result).await?;

        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[crate::test]
    async fn test_cuda_dict_decimal_i256_values() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Precision 76 uses i256 backing type
        let decimal_dtype = DecimalDType::new(76, 20);
        let values = DecimalArray::from_iter(
            [
                i256::from_i128(10000000000000000000i128),
                i256::from_i128(20000000000000000000i128),
                i256::from_i128(30000000000000000000i128),
                i256::from_i128(40000000000000000000i128),
            ],
            decimal_dtype,
        );

        let codes: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3, 2, 2, 1, 0];
        let codes_array = PrimitiveArray::new(Buffer::from(codes), NonNullable);

        let dict_array = DictArray::try_new(codes_array.into_array(), values.into_array())
            .vortex_expect("failed to create Dict array");

        let baseline = dict_array.to_canonical()?;

        let cuda_result = DictExecutor
            .execute(dict_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_decimal();
        let cuda_result = cuda_decimal_to_host(cuda_result)?;

        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }
}
