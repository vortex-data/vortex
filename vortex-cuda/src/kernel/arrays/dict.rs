// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use cudarc::driver::DeviceRepr;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::DictArrayParts;
use vortex_array::arrays::DictVTable;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::PrimitiveArrayParts;
use vortex_dtype::NativePType;
use vortex_dtype::match_each_native_simd_ptype;
use vortex_dtype::match_each_unsigned_integer_ptype;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;

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

        execute_dict(dict_array, ctx).await
    }
}

#[expect(clippy::cognitive_complexity)]
async fn execute_dict(dict: DictArray, ctx: &mut CudaExecutionCtx) -> VortexResult<Canonical> {
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
            execute_dict_typed::<V, I>(values_prim, codes_prim, ctx).await
        })
    })
}

async fn execute_dict_typed<V: DeviceRepr + NativePType, I: DeviceRepr + NativePType>(
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
        ctx.copy_buffer_to_device_async::<V>(values_buffer)?.await?
    };

    let codes_device = if codes_buffer.is_on_device() {
        codes_buffer
    } else {
        ctx.copy_buffer_to_device_async::<I>(codes_buffer)?.await?
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
        ptypes: &[value_ptype, I::PTYPE],
        launch_args: [codes_view, codes_len_u64, values_view, output_view],
        event_recording: cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING,
        array_len: codes_len
    );

    Ok(Canonical::Primitive(PrimitiveArray::from_buffer_handle(
        vortex_array::buffer::BufferHandle::new_device(std::sync::Arc::new(output_device)),
        value_ptype,
        output_validity,
    )))
}

#[cfg(test)]
#[cfg(cuda_available)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity::NonNullable;
    use vortex_buffer::Buffer;
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
        let mut cuda_ctx = CudaSession::create_execution_ctx(VortexSession::empty())
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
        let mut cuda_ctx = CudaSession::create_execution_ctx(VortexSession::empty())
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
        let mut cuda_ctx = CudaSession::create_execution_ctx(VortexSession::empty())
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
        let mut cuda_ctx = CudaSession::create_execution_ctx(VortexSession::empty())
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
        let mut cuda_ctx = CudaSession::create_execution_ctx(VortexSession::empty())
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

        let cuda_result = cuda_primitive_to_host(cuda_result)?;
        cuda_ctx.synchronize_stream()?;

        // Compare CUDA result with baseline
        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }

    #[tokio::test]
    async fn test_cuda_dict_codes_with_validity() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(VortexSession::empty())
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
        let mut cuda_ctx = CudaSession::create_execution_ctx(VortexSession::empty())
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
        let mut cuda_ctx = CudaSession::create_execution_ctx(VortexSession::empty())
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
        let mut cuda_ctx = CudaSession::create_execution_ctx(VortexSession::empty())
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
}
