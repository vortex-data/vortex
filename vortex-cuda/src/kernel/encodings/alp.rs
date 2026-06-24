// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

use async_trait::async_trait;
use cudarc::driver::DeviceRepr;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use tracing::instrument;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::primitive::PrimitiveDataParts;
use vortex::array::buffer::BufferHandle;
use vortex::array::buffer::DeviceBufferExt;
use vortex::dtype::NativePType;
use vortex::encodings::alp::ALP;
use vortex::encodings::alp::ALPArray;
use vortex::encodings::alp::ALPArrayExt;
use vortex::encodings::alp::ALPArraySlotsExt;
use vortex::encodings::alp::ALPFloat;
use vortex::encodings::alp::match_each_alp_float_ptype;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;

use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;
use crate::kernel::patches::build_gpu_patches;
use crate::kernel::patches::types::load_device_patches;

/// CUDA decoder for ALP (Adaptive Lossless floating-Point) decompression.
#[derive(Debug)]
pub(crate) struct ALPExecutor;

#[async_trait]
impl CudaExecute for ALPExecutor {
    #[instrument(level = "trace", skip_all, fields(executor = ?self))]
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let array = array
            .try_downcast::<ALP>()
            .map_err(|_| vortex_err!("Expected ALPArray"))?;

        match_each_alp_float_ptype!(array.dtype().as_ptype(), |A| {
            decode_alp::<A>(array, ctx).await
        })
    }
}

/// Threads per block. 32 threads × 32 elements = 1024 element chunks for both
/// (f32, i32) and (f64, i64). f64 uses 8 KB of shared memory per block.
const ALP_THREADS_PER_BLOCK: u32 = 32;

#[instrument(skip_all)]
async fn decode_alp<A>(array: ALPArray, ctx: &mut CudaExecutionCtx) -> VortexResult<Canonical>
where
    A: ALPFloat + NativePType + DeviceRepr + Send + Sync + 'static,
    A::ALPInt: NativePType + DeviceRepr + Send + Sync + 'static,
{
    let array_len = array.encoded().len();
    vortex_ensure!(array_len > 0, "ALP array must not be empty");

    // Get the exponent factors from the lookup tables.
    let exponents = array.exponents();
    let f: A = A::F10[exponents.f as usize];
    let e: A = A::IF10[exponents.e as usize];

    // Execute child and copy to device.
    let canonical = array.encoded().clone().execute_cuda(ctx).await?;
    let primitive = canonical.into_primitive();
    let PrimitiveDataParts {
        buffer, validity, ..
    } = primitive.into_data_parts();

    let device_input = ctx.ensure_on_device(buffer).await?;
    let input_view = device_input.cuda_view::<A::ALPInt>()?;

    // Allocate output rounded up to a full chunk: the fused kernel writes a
    // whole 1024-element chunk per block, and we slice off any padding below.
    let output_slice = ctx.device_alloc::<A>(array_len.next_multiple_of(1024))?;
    let output_buf = CudaDeviceBuffer::new(output_slice);
    let output_view = output_buf.as_view::<A>();

    // Patch validity does not need to be scattered: the ALP encoder strips null
    // positions from the exception list, so patches only exist at valid
    // positions. load_device_patches additionally rejects patches without
    // chunk_offsets (required by the fused kernel's PatchesCursor).
    let device_patches = if let Some(patches) = array.patches() {
        Some(load_device_patches(&patches, ctx).await?)
    } else {
        None
    };
    let patches_arg = build_gpu_patches(device_patches.as_ref())?;

    // Load the kernel: alp_{enc}_{float}_32t
    let enc_suffix = A::ALPInt::PTYPE.to_string();
    let float_suffix = A::PTYPE.to_string();
    let cuda_function = ctx
        .load_function_with_suffixes("alp", &[enc_suffix.as_str(), float_suffix.as_str(), "32t"])?;

    let num_blocks = u32::try_from(array_len.div_ceil(1024))?;
    let config = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (ALP_THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let array_len_u64 = array_len as u64;
    ctx.launch_kernel_config(&cuda_function, config, array_len, |args| {
        args.arg(&input_view)
            .arg(&output_view)
            .arg(&f)
            .arg(&e)
            .arg(&array_len_u64)
            .arg(&patches_arg);
    })?;

    // Synchronize so the device patches buffers remain alive for the kernel.
    ctx.synchronize_stream()?;
    drop(device_patches);

    let output_handle = BufferHandle::new_device(output_buf.slice_typed::<A>(0..array_len));
    Ok(Canonical::Primitive(PrimitiveArray::from_buffer_handle(
        output_handle,
        A::PTYPE,
        validity,
    )))
}

#[cfg(test)]
mod tests {
    use std::f32;
    use std::f64;

    use vortex::array::IntoArray;
    use vortex::array::array_session;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::assert_arrays_eq;
    use vortex::array::patches::Patches;
    use vortex::array::validity::Validity;
    use vortex::buffer::Buffer;
    use vortex::buffer::buffer;
    use vortex::encodings::alp::ALP;
    use vortex::encodings::alp::Exponents;
    use vortex::encodings::alp::alp_encode;
    use vortex::error::VortexExpect;
    use vortex_array::VortexSessionExecute;

    use super::*;
    use crate::CanonicalCudaExt;
    use crate::executor::CudaArrayExt;
    use crate::session::CudaSession;

    /// Irrational values ALP cannot encode losslessly, guaranteed to land
    /// in the exception list on round-trip through `alp_encode`.
    const UNENCODABLE: f64 = f64::consts::PI;
    const UNENCODABLE_F32: f32 = f32::consts::PI;

    /// Small manually-constructed ALP array with patches. Exercises the
    /// custom-construction path (as opposed to going through `alp_encode`).
    /// Patches must carry `chunk_offsets` — the fused kernel requires them.
    #[crate::test]
    async fn test_cuda_alp_decompression_f32() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create execution context");

        // For f32 with exponents (e=0, f=2): decoded = encoded * F10[2] * IF10[0]
        //                                            = encoded * 100.0 * 1.0
        // Encoded value of 100 -> decoded 10000.0.
        let encoded_data: Vec<i32> = vec![100, 200, 300, 400, 500];
        let exponents = Exponents { e: 0, f: 2 };

        // One chunk holds all 5 elements. chunk_offsets[0] = 0: chunk 0's
        // patches begin at patch index 0.
        let patches = Patches::new(
            5,
            0,
            PrimitiveArray::new(buffer![0u32, 4u32], Validity::NonNullable).into_array(),
            PrimitiveArray::new(buffer![0.0f32, 999f32], Validity::NonNullable).into_array(),
            Some(PrimitiveArray::new(buffer![0u32], Validity::NonNullable).into_array()),
        )?;

        let alp_array = ALP::try_new(
            PrimitiveArray::new(Buffer::from(encoded_data.clone()), Validity::NonNullable)
                .into_array(),
            exponents,
            Some(patches),
        )?;

        let gpu_result = ALPExecutor
            .execute(alp_array.clone().into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(alp_array, gpu_result, &mut ctx);

        Ok(())
    }

    /// ALP with nullable encoded data and patches — the encoder strips null
    /// positions from the exception list, so patch validity doesn't need
    /// scattering. This test verifies that the encoded child's validity is
    /// preserved through the standalone ALP GPU executor.
    #[crate::test]
    async fn test_cuda_alp_nullable_with_patches() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create execution context");

        // Values that will produce ALP exceptions at non-null positions.
        // Nulls at positions 1 and 3; the exception at position 4 can't be
        // encoded losslessly by ALP.
        let values: Vec<Option<f32>> = vec![
            Some(1.0),
            None,
            Some(2.0),
            None,
            Some(UNENCODABLE_F32),
            Some(3.0),
            Some(4.0),
            Some(5.0),
        ];
        let prim = PrimitiveArray::from_option_iter(values);
        let alp_array = alp_encode(
            prim.as_view(),
            None,
            &mut array_session().create_execution_ctx(),
        )?;

        let gpu_result = alp_array
            .clone()
            .into_array()
            .execute_cuda(&mut cuda_ctx)
            .await?
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(alp_array, gpu_result, &mut ctx);
        Ok(())
    }

    /// ALP with all-valid nullable data — the dtype is nullable but no
    /// elements are actually null.
    #[crate::test]
    async fn test_cuda_alp_all_valid_nullable() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create execution context");

        let values = PrimitiveArray::new(
            Buffer::from(vec![1.0f32, 2.0, 3.0, 4.0, 5.0]),
            Validity::AllValid,
        );
        let alp_array = alp_encode(
            values.as_view(),
            None,
            &mut array_session().create_execution_ctx(),
        )?;

        let gpu_result = alp_array
            .clone()
            .into_array()
            .execute_cuda(&mut cuda_ctx)
            .await?
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(alp_array, gpu_result, &mut ctx);
        Ok(())
    }

    /// Multi-chunk ALP (> 1024 elements) with patches in chunks 0 and 2 but
    /// none in chunk 1. Exercises the `PatchesCursor` branch where a
    /// non-trailing chunk has `chunk_offsets[c] == chunk_offsets[c+1]`
    /// (zero patches) via the offset math rather than the NULL sentinel.
    #[crate::test]
    async fn test_cuda_alp_multi_chunk_sparse_patches() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create execution context");

        // 3072 values (3 chunks). Inject exceptions (values ALP can't encode
        // losslessly) only in chunks 0 and 2; chunk 1 stays exception-free so
        // its cursor slice is empty despite patches existing in the array.
        let values: Buffer<f32> = (0u32..3072)
            .map(|i| {
                if matches!(i, 0 | 100 | 1023 | 3071) {
                    UNENCODABLE_F32
                } else {
                    i as f32
                }
            })
            .collect();
        let prim = PrimitiveArray::new(values, Validity::NonNullable);
        let alp_array = alp_encode(
            prim.as_view(),
            None,
            &mut array_session().create_execution_ctx(),
        )?;
        assert!(
            alp_array.patches().is_some(),
            "expected patches from ALP exceptions"
        );

        let gpu_result = alp_array
            .clone()
            .into_array()
            .execute_cuda(&mut cuda_ctx)
            .await?
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(alp_array, gpu_result, &mut ctx);
        Ok(())
    }

    /// Multi-chunk f64 decode with patches distributed across chunks. The f64
    /// path (i64 → double) is otherwise only covered by the partial-tail case,
    /// so this guards the fast-path for the (i64, f64) kernel variant.
    #[crate::test]
    async fn test_cuda_alp_f64_multi_chunk_with_patches() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create execution context");

        // 3072 values (3 chunks). Sprinkle exceptions into each chunk.
        let values: Buffer<f64> = (0u32..3072)
            .map(|i| {
                if matches!(i, 0 | 500 | 1024 | 1500 | 2048 | 3071) {
                    UNENCODABLE
                } else {
                    i as f64
                }
            })
            .collect();
        let prim = PrimitiveArray::new(values, Validity::NonNullable);
        let alp_array = alp_encode(
            prim.as_view(),
            None,
            &mut array_session().create_execution_ctx(),
        )?;
        assert!(
            alp_array.patches().is_some(),
            "expected patches from ALP exceptions"
        );

        let gpu_result = alp_array
            .clone()
            .into_array()
            .execute_cuda(&mut cuda_ctx)
            .await?
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(alp_array, gpu_result, &mut ctx);
        Ok(())
    }

    /// Single chunk with more patches than threads per block (32). Forces
    /// `PatchesCursor` to split patches across multiple threads, exercising
    /// the per-thread ceil-division and clamping math that no other test hits
    /// (existing tests have ≤ 6 patches per chunk).
    #[crate::test]
    async fn test_cuda_alp_dense_patches_single_chunk() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create execution context");

        // Build a 1024-element ALP array manually with exactly 40 patches
        // all in the single chunk. 40 > 32 forces cursor division (each of
        // the first 20 threads handles 2 patches; remaining threads idle).
        const LEN: i32 = 1024;
        const NUM_PATCHES: u32 = 40;

        let exponents = Exponents { e: 0, f: 2 };
        let encoded: Buffer<i32> = (0i32..LEN).map(|i| i * 100).collect();

        let patch_indices: Buffer<u32> = (0u32..NUM_PATCHES).collect();
        let patch_values: Buffer<f32> = (0..NUM_PATCHES).map(|i| i as f32 * 0.125 + 0.5).collect();

        let patches = Patches::new(
            LEN as usize,
            0,
            PrimitiveArray::new(patch_indices, Validity::NonNullable).into_array(),
            PrimitiveArray::new(patch_values, Validity::NonNullable).into_array(),
            Some(PrimitiveArray::new(buffer![0u32], Validity::NonNullable).into_array()),
        )?;

        let alp_array = ALP::try_new(
            PrimitiveArray::new(encoded, Validity::NonNullable).into_array(),
            exponents,
            Some(patches),
        )?;

        let gpu_result = ALPExecutor
            .execute(alp_array.clone().into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(alp_array, gpu_result, &mut ctx);
        Ok(())
    }

    /// Tail-chunk bounds check: an array whose length is not a multiple of
    /// 1024 forces the kernel's tail-block path to bounds-check its decode
    /// loop. Includes a patch in the tail.
    #[crate::test]
    async fn test_cuda_alp_partial_tail_chunk() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create execution context");

        let values: Buffer<f64> = (0u32..1500)
            .map(|i| if i == 1400 { UNENCODABLE } else { i as f64 })
            .collect();
        let prim = PrimitiveArray::new(values, Validity::NonNullable);
        let alp_array = alp_encode(
            prim.as_view(),
            None,
            &mut array_session().create_execution_ctx(),
        )?;
        assert!(
            alp_array.patches().is_some(),
            "expected patches from ALP exceptions"
        );

        let gpu_result = alp_array
            .clone()
            .into_array()
            .execute_cuda(&mut cuda_ctx)
            .await?
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(alp_array, gpu_result, &mut ctx);
        Ok(())
    }
}
