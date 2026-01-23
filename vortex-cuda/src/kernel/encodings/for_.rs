// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_fastlanes::FoRArray;
use vortex_fastlanes::FoRVTable;

use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;
use crate::impl_for_scalar_decoder;
use crate::kernel::scalar::execute_scalar_decoder;

// Generate decoder implementations for all supported primitive types
impl_for_scalar_decoder! {
    u8 => FoRDecoderU8,
    u16 => FoRDecoderU16,
    u32 => FoRDecoderU32,
    u64 => FoRDecoderU64,
    i8 => FoRDecoderI8,
    i16 => FoRDecoderI16,
    i32 => FoRDecoderI32,
    i64 => FoRDecoderI64,
}

/// CUDA executor for frame-of-reference decoding.
///
/// This executor dispatches to type-specific decoders based on the array's
/// primitive type. Each decoder implements [`ScalarGpuDecoder`] and uses
/// the common [`execute_scalar_decoder`] function.
#[derive(Debug)]
pub struct FoRExecutor;

impl FoRExecutor {
    fn try_specialize(array: ArrayRef) -> Option<FoRArray> {
        array.try_into::<FoRVTable>().ok()
    }
}

#[async_trait]
impl CudaExecute for FoRExecutor {
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let for_array =
            Self::try_specialize(array).ok_or_else(|| vortex_err!("Expected FoRArray"))?;

        match for_array.ptype() {
            PType::U8 => execute_scalar_decoder::<FoRDecoderU8>(for_array, ctx).await,
            PType::U16 => execute_scalar_decoder::<FoRDecoderU16>(for_array, ctx).await,
            PType::U32 => execute_scalar_decoder::<FoRDecoderU32>(for_array, ctx).await,
            PType::U64 => execute_scalar_decoder::<FoRDecoderU64>(for_array, ctx).await,
            PType::I8 => execute_scalar_decoder::<FoRDecoderI8>(for_array, ctx).await,
            PType::I16 => execute_scalar_decoder::<FoRDecoderI16>(for_array, ctx).await,
            PType::I32 => execute_scalar_decoder::<FoRDecoderI32>(for_array, ctx).await,
            PType::I64 => execute_scalar_decoder::<FoRDecoderI64>(for_array, ctx).await,
            other => Err(vortex_err!(
                "Unsupported ptype for FoR GPU decode: {}",
                other
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity::NonNullable;
    use vortex_buffer::Buffer;
    use vortex_error::VortexExpect;
    use vortex_fastlanes::FoRArray;
    use vortex_session::VortexSession;

    use super::*;
    use crate::has_nvcc;
    use crate::session::CudaSession;

    #[tokio::test]
    async fn test_cuda_for_decompression_u8() {
        if !has_nvcc() {
            return;
        }

        let mut cuda_ctx = CudaSession::create_execution_ctx(VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Create u8 offset values that cycle through 0-245, creating 5000 elements
        #[allow(clippy::cast_possible_truncation)]
        let input_data: Vec<u8> = (0..5000).map(|i| (i % 246) as u8).collect();

        let for_array = FoRArray::try_new(
            PrimitiveArray::new(Buffer::from(input_data.clone()), NonNullable).into_array(),
            10u8.into(),
        )
        .vortex_expect("failed to create FoR array");

        // Decompress on the GPU.
        let result = FoRExecutor
            .execute(for_array.to_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed");

        let result_buf =
            Buffer::<u8>::from_byte_buffer(result.as_primitive().buffer_handle().to_host().await);

        assert_eq!(result_buf.len(), input_data.len());
        assert_eq!(
            result_buf,
            input_data.iter().map(|&val| val + 10).collect::<Vec<u8>>()
        );
    }

    #[tokio::test]
    async fn test_cuda_for_decompression_u16() {
        if !has_nvcc() {
            return;
        }

        let mut cuda_ctx = CudaSession::create_execution_ctx(VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Create u16 offset values that cycle through 0-5000, creating 5000 elements
        let input_data: Vec<u16> = (0..5000).map(|i| (i % 5000) as u16).collect();

        let for_array = FoRArray::try_new(
            PrimitiveArray::new(Buffer::from(input_data.clone()), NonNullable).into_array(),
            1000u16.into(),
        )
        .vortex_expect("failed to create FoR array");

        // Decompress on the GPU.
        let result = FoRExecutor
            .execute(for_array.to_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed");

        let result_buf =
            Buffer::<u16>::from_byte_buffer(result.as_primitive().buffer_handle().to_host().await);

        assert_eq!(result_buf.len(), input_data.len());
        assert_eq!(
            result_buf,
            input_data
                .iter()
                .map(|&val| val + 1000)
                .collect::<Vec<u16>>()
        );
    }

    #[tokio::test]
    async fn test_cuda_for_decompression_u32() {
        if !has_nvcc() {
            return;
        }

        let mut cuda_ctx = CudaSession::create_execution_ctx(VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Create u32 offset values that cycle through 0-5000, creating 5000 elements
        let input_data: Vec<u32> = (0..5000).map(|i| (i % 5000) as u32).collect();

        let for_array = FoRArray::try_new(
            PrimitiveArray::new(Buffer::from(input_data.clone()), NonNullable).into_array(),
            100000u32.into(),
        )
        .vortex_expect("failed to create FoR array");

        // Decompress on the GPU.
        let result = FoRExecutor
            .execute(for_array.to_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed");

        let result_buf =
            Buffer::<u32>::from_byte_buffer(result.as_primitive().buffer_handle().to_host().await);

        assert_eq!(result_buf.len(), input_data.len());
        assert_eq!(
            result_buf,
            input_data
                .iter()
                .map(|&val| val + 100000)
                .collect::<Vec<u32>>()
        );
    }

    #[tokio::test]
    async fn test_cuda_for_decompression_u64() {
        if !has_nvcc() {
            return;
        }

        let mut cuda_ctx = CudaSession::create_execution_ctx(VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Create u64 offset values that cycle through 0-5000, creating 5000 elements
        let input_data: Vec<u64> = (0..5000).map(|i| (i % 5000) as u64).collect();

        let for_array = FoRArray::try_new(
            PrimitiveArray::new(Buffer::from(input_data.clone()), NonNullable).into_array(),
            1000000u64.into(),
        )
        .vortex_expect("failed to create FoR array");

        // Decompress on the GPU.
        let result = FoRExecutor
            .execute(for_array.to_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed");

        let result_buf =
            Buffer::<u64>::from_byte_buffer(result.as_primitive().buffer_handle().to_host().await);

        assert_eq!(result_buf.len(), input_data.len());
        assert_eq!(
            result_buf,
            input_data
                .iter()
                .map(|&val| val + 1000000u64)
                .collect::<Vec<u64>>()
        );
    }
}
