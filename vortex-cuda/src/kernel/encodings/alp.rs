// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use vortex_alp::ALPArray;
use vortex_alp::ALPVTable;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;
use crate::impl_alp_scalar_decoder;
use crate::kernel::scalar::execute_scalar_decoder;

// Generate decoder implementations for ALP float types
impl_alp_scalar_decoder! {
    i32, f32 => ALPDecoderF32, ALPMetadataF32,
    i64, f64 => ALPDecoderF64, ALPMetadataF64,
}

/// CUDA executor for ALP (Adaptive Lossless floating-Point) decoding.
///
/// This executor dispatches to type-specific decoders based on the array's
/// float type. Each decoder implements [`ScalarGpuDecoder`] and uses
/// the common [`execute_scalar_decoder`] function.
#[derive(Debug)]
pub struct ALPExecutor;

impl ALPExecutor {
    fn try_specialize(array: ArrayRef) -> Option<ALPArray> {
        array.try_into::<ALPVTable>().ok()
    }
}

#[async_trait]
impl CudaExecute for ALPExecutor {
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let alp_array =
            Self::try_specialize(array).ok_or_else(|| vortex_err!("Expected ALPArray"))?;

        // Dispatch based on the output (float) type
        match alp_array.ptype() {
            PType::F32 => execute_scalar_decoder::<ALPDecoderF32>(alp_array, ctx).await,
            PType::F64 => execute_scalar_decoder::<ALPDecoderF64>(alp_array, ctx).await,
            other => Err(vortex_err!(
                "Unsupported ptype for ALP GPU decode: {}",
                other
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_alp::ALPArray;
    use vortex_alp::Exponents;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity::NonNullable;
    use vortex_buffer::Buffer;
    use vortex_error::VortexExpect;
    use vortex_session::VortexSession;

    use super::*;
    use crate::has_nvcc;
    use crate::session::CudaSession;

    #[tokio::test]
    async fn test_cuda_alp_decompression_f32() {
        if !has_nvcc() {
            return;
        }

        let mut cuda_ctx = CudaSession::create_execution_ctx(VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Create encoded values (what ALP would produce)
        // For f32 with exponents (e=0, f=2): decoded = encoded * F10[2] * IF10[0]
        //                                            = encoded * 0.01 * 1.0
        // So encoded value of 100 -> decoded 1.0
        let encoded_data: Vec<i32> = vec![100, 200, 300, 400, 500];
        let exponents = Exponents { e: 0, f: 2 }; // divide by 100

        let alp_array = ALPArray::try_new(
            PrimitiveArray::new(Buffer::from(encoded_data.clone()), NonNullable).into_array(),
            exponents,
            None,
        )
        .vortex_expect("failed to create ALP array");

        let result = ALPExecutor
            .execute(alp_array.to_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed");

        let result_buf =
            Buffer::<f32>::from_byte_buffer(result.as_primitive().buffer_handle().to_host().await);

        assert_eq!(result_buf.len(), encoded_data.len());

        // Check decoded values: encoded * F10[2] * IF10[0] = encoded * 0.01 * 1.0
        let expected: Vec<f32> = encoded_data.iter().map(|&v| v as f32 * 0.01).collect();
        for (i, (&got, &want)) in result_buf
            .as_slice()
            .iter()
            .zip(expected.iter())
            .enumerate()
        {
            assert!(
                (got - want).abs() < 1e-6,
                "Mismatch at {}: got {}, want {}",
                i,
                got,
                want
            );
        }
    }
}
