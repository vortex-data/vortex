// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_zigzag::ZigZagArray;
use vortex_zigzag::ZigZagVTable;

use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;
use crate::impl_zigzag_scalar_decoder;
use crate::kernel::scalar::execute_scalar_decoder;

// Generate decoder implementations for all supported unsigned -> signed pairs
impl_zigzag_scalar_decoder! {
    u8, i8 => ZigZagDecoderU8,
    u16, i16 => ZigZagDecoderU16,
    u32, i32 => ZigZagDecoderU32,
    u64, i64 => ZigZagDecoderU64,
}

/// CUDA executor for ZigZag decoding.
///
/// This executor dispatches to type-specific decoders based on the array's
/// encoded primitive type. Each decoder implements [`ScalarGpuDecoder`] and uses
/// the common [`execute_scalar_decoder`] function.
#[derive(Debug)]
pub struct ZigZagExecutor;

impl ZigZagExecutor {
    fn try_specialize(array: ArrayRef) -> Option<ZigZagArray> {
        array.try_into::<ZigZagVTable>().ok()
    }
}

#[async_trait]
impl CudaExecute for ZigZagExecutor {
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let zigzag_array =
            Self::try_specialize(array).ok_or_else(|| vortex_err!("Expected ZigZagArray"))?;

        // Dispatch based on the encoded (unsigned) type
        let encoded_ptype = zigzag_array.encoded().dtype().as_ptype();

        match encoded_ptype {
            PType::U8 => execute_scalar_decoder::<ZigZagDecoderU8>(zigzag_array, ctx).await,
            PType::U16 => execute_scalar_decoder::<ZigZagDecoderU16>(zigzag_array, ctx).await,
            PType::U32 => execute_scalar_decoder::<ZigZagDecoderU32>(zigzag_array, ctx).await,
            PType::U64 => execute_scalar_decoder::<ZigZagDecoderU64>(zigzag_array, ctx).await,
            other => Err(vortex_err!(
                "Unsupported encoded ptype for ZigZag GPU decode: {}",
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
    use vortex_session::VortexSession;
    use vortex_zigzag::ZigZagArray;

    use super::*;
    use crate::has_nvcc;
    use crate::session::CudaSession;

    #[tokio::test]
    async fn test_cuda_zigzag_decompression_u32() {
        if !has_nvcc() {
            return;
        }

        let mut cuda_ctx = CudaSession::create_execution_ctx(VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // ZigZag encoding: 0->0, 1->-1, 2->1, 3->-2, 4->2, ...
        // So encoded [0, 2, 4, 1, 3] should decode to [0, 1, 2, -1, -2]
        let encoded_data: Vec<u32> = vec![0, 2, 4, 1, 3];
        let expected: Vec<i32> = vec![0, 1, 2, -1, -2];

        let zigzag_array = ZigZagArray::try_new(
            PrimitiveArray::new(Buffer::from(encoded_data), NonNullable).into_array(),
        )
        .vortex_expect("failed to create ZigZag array");

        // Decode on CPU for comparison
        let cpu_result = zigzag_array
            .to_array()
            .to_canonical()
            .vortex_expect("CPU canonicalize failed");
        let cpu_slice = cpu_result.as_primitive().as_slice::<i32>();
        assert_eq!(cpu_slice, expected.as_slice(), "CPU decode mismatch");

        // Decode on GPU
        let gpu_result = ZigZagExecutor
            .execute(zigzag_array.to_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed");

        // Copy GPU result back to host for comparison
        let gpu_host = Buffer::<i32>::from_byte_buffer(
            gpu_result.into_primitive().buffer_handle().to_host().await,
        );
        assert_eq!(
            gpu_host.as_slice(),
            expected.as_slice(),
            "GPU decode mismatch"
        );
    }
}
