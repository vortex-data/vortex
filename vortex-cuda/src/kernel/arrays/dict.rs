// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::DeviceRepr;
use cudarc::driver::PushKernelArg;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::DecimalArrayParts;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::DictArrayParts;
use vortex_array::arrays::DictVTable;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::PrimitiveArrayParts;
use vortex_array::buffer::BufferHandle;
use vortex_dtype::DType;
use vortex_dtype::DecimalType;
use vortex_dtype::NativeDecimalType;
use vortex_dtype::NativePType;
use vortex_dtype::match_each_decimal_value_type;
use vortex_dtype::match_each_native_simd_ptype;
use vortex_dtype::match_each_unsigned_integer_ptype;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;
use crate::launch_cuda_kernel_impl;

/// CUDA executor for dictionary-encoded arrays.
#[derive(Debug)]
pub struct DictExecutor;

#[async_trait]
impl CudaExecute for DictExecutor {
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let dict_array = array
            .try_into::<DictVTable>()
            .ok()
            .vortex_expect("Array is not a Dict array");

        let values_dtype = dict_array.values().dtype().clone();
        match &values_dtype {
            DType::Decimal(..) => execute_dict_decimal(dict_array, ctx).await,
            DType::Primitive(..) => execute_dict_prim(dict_array, ctx).await,
            dt => vortex_bail!("unsupported decompress for DType={dt}"),
        }
    }
}

#[expect(clippy::cognitive_complexity)]
async fn execute_dict_prim(dict: DictArray, ctx: &mut CudaExecutionCtx) -> VortexResult<Canonical> {
    let DictArrayParts { values, codes, .. } = dict.into_parts();

    // Execute both children to get them as primitives on the device
    let values_canonical = values.execute_cuda(ctx).await?;
    let codes_canonical = codes.execute_cuda(ctx).await?;

    let values_prim = values_canonical.into_primitive();
    let codes_prim = codes_canonical.into_primitive();

    let values_ptype = values_prim.ptype();
    let codes_ptype = codes_prim.ptype();

    // Dispatch based on both value type and code type
    match_each_native_simd_ptype!(values_ptype, |V| {
        match_each_unsigned_integer_ptype!(codes_ptype, |I| {
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
    } = values.into_parts();
    let output_validity = values_validity.take(codes.as_ref())?;
    let PrimitiveArrayParts {
        buffer: codes_buffer,
        ..
    } = codes.into_parts();

    // Get device buffers for values and codes
    let values_device = if values_buffer.is_on_device() {
        values_buffer
    } else {
        ctx.move_to_device::<V>(values_buffer)?.await?
    };

    let codes_device = if codes_buffer.is_on_device() {
        codes_buffer
    } else {
        ctx.move_to_device::<I>(codes_buffer)?.await?
    };

    // Allocate output buffer on device
    let output_slice = ctx.device_alloc::<V>(codes_len)?;
    let output_device = CudaDeviceBuffer::new(output_slice);

    // Get views for kernel launch
    let values_view = values_device.cuda_view::<V>()?;
    let codes_view = codes_device.cuda_view::<I>()?;
    let output_view = output_device.as_view();

    let codes_len_u64 = codes_len as u64;
    // Launch the dict kernel
    let _cuda_events = crate::launch_cuda_kernel!(
        execution_ctx: ctx,
        module: "dict",
        ptypes: &[value_ptype.to_string().as_str(), I::PTYPE.to_string().as_str()],
        launch_args: [codes_view, codes_len_u64, values_view, output_view],
        event_recording: cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING,
        array_len: codes_len
    );

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
    } = dict.into_parts();

    // Execute codes to get them as primitives on the device
    let codes_prim = codes.execute_cuda(ctx).await?.into_primitive();
    let codes_ptype = codes_prim.ptype();

    // For decimal values, execute recursively to handle any nested encodings
    let values_decimal = values.execute_cuda(ctx).await?.into_decimal();
    let decimal_type = values_decimal.values_type();

    match_each_decimal_value_type!(decimal_type, |V| {
        match_each_unsigned_integer_ptype!(codes_ptype, |C| {
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
    if codes_len == 0 {
        vortex_bail!("Cannot execute dict on empty codes array");
    }

    let DecimalArrayParts {
        values: values_buffer,
        validity: values_validity,
        ..
    } = values.into_parts();
    let output_validity = values_validity.take(codes.as_ref())?;

    let PrimitiveArrayParts {
        buffer: codes_buffer,
        ..
    } = codes.into_parts();

    // Determine value type suffix for kernel name
    let value_suffix = match V::DECIMAL_TYPE {
        DecimalType::I8 => "i8",
        DecimalType::I16 => "i16",
        DecimalType::I32 => "i32",
        DecimalType::I64 => "i64",
        DecimalType::I128 => "i128",
        DecimalType::I256 => "i256",
    };

    // Copy buffers to device if needed
    // Note: We use u8 for the buffer type since we're treating these as raw bytes
    let values_device = if values_buffer.is_on_device() {
        values_buffer
    } else {
        ctx.move_to_device::<V>(values_buffer)?.await?
    };

    let codes_device = if codes_buffer.is_on_device() {
        codes_buffer
    } else {
        ctx.move_to_device::<C>(codes_buffer)?.await?
    };

    // Allocate output buffer on device (codes_len * value_byte_width bytes)
    let output_slice = ctx.device_alloc::<V>(codes_len)?;
    let output_device = CudaDeviceBuffer::new(output_slice);

    // Get views for kernel launch
    let values_view = values_device.cuda_view::<V>()?;
    let codes_view = codes_device.cuda_view::<C>()?;
    let output_view = output_device.as_view();

    // Load kernel function using string suffixes
    let cuda_function = ctx.load_function("dict", &[value_suffix, &C::PTYPE.to_string()])?;
    let mut launch_builder = ctx.launch_builder(&cuda_function);

    launch_builder.arg(&codes_view);
    launch_builder.arg(&codes_len);
    launch_builder.arg(&values_view);
    launch_builder.arg(&output_view);

    let _cuda_events = launch_cuda_kernel_impl(
        &mut launch_builder,
        cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING,
        codes_len,
    )?;

    Ok(Canonical::Decimal(DecimalArray::new_handle(
        BufferHandle::new_device(Arc::new(output_device)),
        V::DECIMAL_TYPE,
        output_dtype.into_decimal_opt().vortex_expect("is decimal"),
        output_validity,
    )))
}

#[cfg(test)]
#[cfg(cuda_available)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity::NonNullable;
    use vortex_buffer::Buffer;
    use vortex_dtype::DecimalDType;
    use vortex_dtype::i256;
    use vortex_error::VortexExpect;
    use vortex_session::VortexSession;

    use super::*;
    use crate::session::CudaSession;

    /// Copy a CUDA primitive array result to host memory.
    fn cuda_primitive_to_host(prim: PrimitiveArray) -> VortexResult<PrimitiveArray> {
        Ok(PrimitiveArray::from_byte_buffer(
            prim.buffer_handle().try_to_host_sync()?,
            prim.ptype(),
            prim.validity()?,
        ))
    }

    #[tokio::test]
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
            .execute(dict_array.to_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_primitive();
        cuda_ctx.synchronize_stream()?;
        let cuda_result = cuda_primitive_to_host(cuda_result)?;

        // Compare CUDA result with baseline
        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[tokio::test]
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
            .execute(dict_array.to_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_primitive();
        cuda_ctx.synchronize_stream()?;
        let cuda_result = cuda_primitive_to_host(cuda_result)?;

        // Compare CUDA result with baseline
        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[tokio::test]
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
            .execute(dict_array.to_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_primitive();
        cuda_ctx.synchronize_stream()?;
        let cuda_result = cuda_primitive_to_host(cuda_result)?;

        // Compare CUDA result with baseline
        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[tokio::test]
    async fn test_cuda_dict_large_array() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Dictionary with 256 values
        let values: Vec<u32> = (0..256).map(|i| i * 1000).collect();
        let values_array = PrimitiveArray::new(Buffer::from(values), NonNullable);

        let codes: Vec<u16> = (0..5000).map(|i| (i % 256) as u16).collect();
        let codes_array = PrimitiveArray::new(Buffer::from(codes), NonNullable);

        let dict_array = DictArray::try_new(codes_array.into_array(), values_array.into_array())
            .vortex_expect("failed to create Dict array");

        // Get baseline from CPU canonicalization
        let baseline = dict_array.to_canonical()?;

        // Execute on CUDA
        let cuda_result = DictExecutor
            .execute(dict_array.to_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_primitive();
        cuda_ctx.synchronize_stream()?;
        let cuda_result = cuda_primitive_to_host(cuda_result)?;

        // Compare CUDA result with baseline
        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[test]
    fn test_cuda_dict_values_with_validity() -> VortexResult<()> {
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

        let cuda_result = futures::executor::block_on(async {
            // Execute on CUDA
            DictExecutor
                .execute(dict_array.into_array(), &mut cuda_ctx)
                .await
                .vortex_expect("GPU decompression failed")
                .into_primitive()
        });

        cuda_ctx.synchronize_stream()?;
        let cuda_result = cuda_primitive_to_host(cuda_result)?;

        // Compare CUDA result with baseline
        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[tokio::test]
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
            .execute(dict_array.to_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_primitive();
        cuda_ctx.synchronize_stream()?;
        let cuda_result = cuda_primitive_to_host(cuda_result)?;

        // Compare CUDA result with baseline
        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[tokio::test]
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
            .execute(dict_array.to_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_primitive();
        cuda_ctx.synchronize_stream()?;
        let cuda_result = cuda_primitive_to_host(cuda_result)?;

        // Compare CUDA result with baseline
        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[tokio::test]
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
            .execute(dict_array.to_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_primitive();
        cuda_ctx.synchronize_stream()?;
        let cuda_result = cuda_primitive_to_host(cuda_result)?;

        // Compare CUDA result with baseline
        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[tokio::test]
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
            .execute(dict_array.to_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_primitive();
        cuda_ctx.synchronize_stream()?;
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
            decimal.validity()?,
        ))
    }

    #[tokio::test]
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
            .execute(dict_array.to_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_decimal();
        cuda_ctx.synchronize_stream()?;
        let cuda_result = cuda_decimal_to_host(cuda_result)?;

        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[tokio::test]
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
            .execute(dict_array.to_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_decimal();
        cuda_ctx.synchronize_stream()?;
        let cuda_result = cuda_decimal_to_host(cuda_result)?;

        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[tokio::test]
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
            .execute(dict_array.to_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_decimal();
        cuda_ctx.synchronize_stream()?;
        let cuda_result = cuda_decimal_to_host(cuda_result)?;

        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[tokio::test]
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
            .execute(dict_array.to_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_decimal();
        cuda_ctx.synchronize_stream()?;
        let cuda_result = cuda_decimal_to_host(cuda_result)?;

        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[tokio::test]
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
            .execute(dict_array.to_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_decimal();
        cuda_ctx.synchronize_stream()?;
        let cuda_result = cuda_decimal_to_host(cuda_result)?;

        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[tokio::test]
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
            .execute(dict_array.to_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_decimal();
        cuda_ctx.synchronize_stream()?;
        let cuda_result = cuda_decimal_to_host(cuda_result)?;

        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }
}
