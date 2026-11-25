// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA executor for FastLanes delta.
//!
//! Delta stores its values in the FastLanes transposed layout: a 1024-element chunk is
//! `LANES` independent columns, each carrying its own running total seeded from that lane's
//! base. A chunk therefore decodes as `LANES` independent scans, not one scan of 1024
//! elements, so the whole array is data-parallel across both chunks and lanes.
//!
//! This mirrors the CPU decoder in `vortex-fastlanes`: `undelta` into the transposed layout,
//! then untranspose into natural order, then apply the array's logical slice.

use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use tracing::instrument;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::primitive::PrimitiveDataParts;
use vortex::array::buffer::BufferHandle;
use vortex::array::match_each_unsigned_integer_ptype;
use vortex::dtype::NativePType;
use vortex::dtype::PType;
use vortex::encodings::fastlanes::Delta;
use vortex::encodings::fastlanes::DeltaArray;
use vortex::encodings::fastlanes::DeltaArrayExt;
use vortex::encodings::fastlanes::DeltaArraySlotsExt;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_ensure_eq;
use vortex::error::vortex_err;
use vortex_cuda_driver::DeviceRepr;
use vortex_cuda_driver::LaunchConfig;
use vortex_cuda_driver::PushKernelArg;

use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::dynamic_dispatch::DispatchPlan;
use crate::dynamic_dispatch::FusedPlan;
use crate::executor::CudaArrayExt;
use crate::executor::CudaDispatchMode;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;
use crate::executor::execute_validity_cuda;

/// Elements per FastLanes chunk. Must match `FL_CHUNK` in `kernels/src/fastlanes_common.cuh`.
const FL_CHUNK: usize = 1024;
/// Threads per block: covers the widest lane count (128, for 8-bit values) in a single pass.
const BLOCK_THREADS: u32 = 128;

/// CUDA decoder for FastLanes delta.
#[derive(Debug)]
pub(crate) struct DeltaExecutor;

#[async_trait]
impl CudaExecute for DeltaExecutor {
    #[instrument(level = "trace", skip_all, fields(executor = ?self))]
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let delta = array
            .try_downcast::<Delta>()
            .map_err(|_| vortex_err!("Expected DeltaArray"))?;
        decode_delta(delta, ctx).await
    }
}

#[instrument(skip_all)]
async fn decode_delta(array: DeltaArray, ctx: &mut CudaExecutionCtx) -> VortexResult<Canonical> {
    let dtype = array.dtype().clone();
    let len = array.len();
    if len == 0 {
        return Ok(Canonical::empty(&dtype));
    }

    // The vtable already narrows validity to the logical slice.
    let validity = execute_validity_cuda(array.validity()?, len, ctx).await?;

    let deltas = array.deltas().clone();
    // Signed values decode through their unsigned counterpart: the kernel's wrapping add
    // inverts the wrapping subtract applied at compress time regardless of signedness. The
    // buffer is untyped, so only the kernel and the device view need the unsigned type.
    let ptype = PType::try_from(deltas.dtype())?;
    let deltas_len = deltas.len();
    let offset = array.offset();
    vortex_ensure_eq!(
        deltas_len % FL_CHUNK,
        0,
        "Delta deltas child must be padded to a multiple of {FL_CHUNK}, got {deltas_len}"
    );
    vortex_ensure!(
        offset + len <= deltas_len,
        "Delta slice {offset}..{} exceeds its {deltas_len} decoded values",
        offset + len
    );
    let num_chunks = deltas_len / FL_CHUNK;
    let lanes = FL_CHUNK / (ptype.byte_width() * 8);
    let required_bases = num_chunks * lanes;

    let bases = array
        .bases()
        .clone()
        .execute_cuda(ctx)
        .await?
        .into_primitive();
    vortex_ensure!(
        bases.len() >= required_bases,
        "Delta needs {required_bases} bases for {num_chunks} chunks, got {}",
        bases.len()
    );
    let PrimitiveDataParts {
        buffer: bases_buffer,
        ..
    } = bases.into_data_parts();
    let bases_device = ctx.ensure_on_device(bases_buffer)?;

    let decoded: BufferHandle = match fused_deltas_plan(&deltas, ctx)? {
        Some(plan) => {
            let plan = plan.materialize(ctx).await?;
            match_each_unsigned_integer_ptype!(ptype.to_unsigned(), |U| {
                plan.execute_delta::<U>(&bases_device, deltas_len, ctx)?
            })
        }
        None => {
            let deltas = deltas.execute_cuda(ctx).await?.into_primitive();
            let PrimitiveDataParts {
                buffer: deltas_buffer,
                ..
            } = deltas.into_data_parts();
            let deltas_device = ctx.ensure_on_device(deltas_buffer)?;
            match_each_unsigned_integer_ptype!(ptype.to_unsigned(), |U| {
                launch_delta::<U>(&deltas_device, &bases_device, num_chunks, ctx)?
            })
        }
    };

    // Chunks are decoded whole; the logical slice is applied to the result.
    let width = ptype.byte_width();
    let sliced = decoded.slice(offset * width..(offset + len) * width);

    Ok(Canonical::Primitive(PrimitiveArray::from_buffer_handle(
        sliced, ptype, validity,
    )))
}

/// A fused plan for the deltas, which the delta variant of the dynamic dispatch kernel decodes
/// and undeltas in shared memory without writing the deltas out. `None` when they don't fuse.
fn fused_deltas_plan(deltas: &ArrayRef, ctx: &CudaExecutionCtx) -> VortexResult<Option<FusedPlan>> {
    if ctx.dispatch_mode() == CudaDispatchMode::StandaloneOnly {
        return Ok(None);
    }
    // Plan even deltas that a standalone kernel covers: that kernel would write them out.
    Ok(
        match DispatchPlan::new(deltas, CudaDispatchMode::DynDispatchOnly)? {
            DispatchPlan::Fused(plan) if plan.supports_delta_output() => Some(plan),
            _ => None,
        },
    )
}

/// Undelta `num_chunks` chunks of materialized deltas with the standalone kernel.
fn launch_delta<U>(
    deltas: &BufferHandle,
    bases: &BufferHandle,
    num_chunks: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<BufferHandle>
where
    U: NativePType + DeviceRepr,
{
    let deltas_view = deltas.cuda_view::<U>()?;
    let bases_view = bases.cuda_view::<U>()?;
    let deltas_len = num_chunks * FL_CHUNK;
    let mut output = ctx.device_alloc::<U>(deltas_len)?;
    let function = ctx.load_function("delta", &[U::PTYPE])?;
    let config = LaunchConfig {
        grid_dim: (u32::try_from(num_chunks)?, 1, 1),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let num_chunks_u64 = num_chunks as u64;
    ctx.launch_kernel_config(&function, config, deltas_len, |args| {
        args.arg(&deltas_view)
            .arg(&bases_view)
            .arg(&mut output)
            .arg(&num_chunks_u64);
    })?;
    Ok(BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(
        output,
    )?)))
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex::array::IntoArray;
    use vortex::array::assert_arrays_eq;
    use vortex::array::optimizer::ArrayOptimizer;
    use vortex::array::validity::Validity;
    use vortex::buffer::Buffer;
    use vortex::encodings::fastlanes;
    use vortex::encodings::fastlanes::BitPacked;
    use vortex::error::VortexExpect;
    use vortex::session::VortexSession;
    use vortex_array::VortexSessionExecute;

    use super::*;
    use crate::CanonicalCudaExt;
    use crate::session::CudaSession;

    // The fastlanes rules reduce a slice of a Delta array to a Delta array.
    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        fastlanes::initialize(&session);
        session
    });

    /// Decodes on the GPU and compares against the independent uncompressed fixture, once
    /// through the fused dynamic dispatch path and once through the standalone delta kernel.
    async fn assert_gpu_matches_expected(
        delta: DeltaArray,
        expected: ArrayRef,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        for mode in [CudaDispatchMode::Auto, CudaDispatchMode::StandaloneOnly] {
            let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
                .vortex_expect("failed to create execution context")
                .with_dispatch_mode(mode);

            let gpu = DeltaExecutor
                .execute(delta.clone().into_array(), &mut cuda_ctx)
                .await
                .vortex_expect("GPU decompression failed")
                .into_host(cuda_ctx.execution_ctx())
                .await?
                .into_array();

            assert_arrays_eq!(gpu, expected.clone(), &mut ctx);
        }
        Ok(())
    }

    /// Bit-packed deltas, the shape the compressor writes, are unpacked in the fused kernel's
    /// shared memory and undeltaed there. The 32- and 64-bit cases split each lane into slices.
    #[rstest]
    #[case::u32(PrimitiveArray::new(
        Buffer::from_iter((0u32..5000).map(|i| i * 3)),
        Validity::NonNullable,
    ))]
    #[case::u64(PrimitiveArray::new(
        Buffer::from_iter((0u64..5000).map(|i| (1 << 40) + i * 3)),
        Validity::NonNullable,
    ))]
    #[crate::test]
    async fn test_cuda_delta_bitpacked_deltas(
        #[case] primitive: PrimitiveArray,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let delta = Delta::try_from_primitive_array(&primitive, &mut ctx)?;
        let deltas = BitPacked::encode(delta.deltas(), 16, &mut ctx)?.into_array();
        let delta = Delta::try_new(
            delta.bases().clone(),
            deltas,
            Validity::NonNullable,
            0,
            primitive.len(),
        )?;
        assert_gpu_matches_expected(delta, primitive.into_array()).await
    }

    /// Every element width is worth covering: lane count is `1024 / bit-width`, so each width
    /// splits the 1024-element chunk differently — 128 lanes of 8 rows for `u8` through 16
    /// lanes of 64 rows for `u64` — and each has its own kernel instantiation. Every case
    /// spans several chunks.
    #[crate::test]
    async fn test_cuda_delta_u8() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = PrimitiveArray::new(
            Buffer::from_iter((0u8..=255).cycle().take(3000)),
            Validity::NonNullable,
        );

        let delta = Delta::try_from_primitive_array(&primitive, &mut ctx)?;
        assert_gpu_matches_expected(delta, primitive.into_array()).await
    }

    #[crate::test]
    async fn test_cuda_delta_u16() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = PrimitiveArray::new(Buffer::from_iter(0u16..3000), Validity::NonNullable);

        let delta = Delta::try_from_primitive_array(&primitive, &mut ctx)?;
        assert_gpu_matches_expected(delta, primitive.into_array()).await
    }

    #[crate::test]
    async fn test_cuda_delta_u32() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = PrimitiveArray::new(
            Buffer::from_iter((0u32..3000).map(|i| i * 7)),
            Validity::NonNullable,
        );

        let delta = Delta::try_from_primitive_array(&primitive, &mut ctx)?;
        assert_gpu_matches_expected(delta, primitive.into_array()).await
    }

    #[crate::test]
    async fn test_cuda_delta_u64() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = PrimitiveArray::new(
            Buffer::from_iter((0u64..3000).map(|i| i * 1_000_003)),
            Validity::NonNullable,
        );

        let delta = Delta::try_from_primitive_array(&primitive, &mut ctx)?;
        assert_gpu_matches_expected(delta, primitive.into_array()).await
    }

    /// Deltas across negative values wrap at compress time, so signed input must decode
    /// through the unsigned kernel unchanged.
    #[crate::test]
    async fn test_cuda_delta_signed_values() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = PrimitiveArray::new(
            Buffer::from_iter((0..3000i32).map(|i| i - 1500)),
            Validity::NonNullable,
        );

        let delta = Delta::try_from_primitive_array(&primitive, &mut ctx)?;
        assert_gpu_matches_expected(delta, primitive.into_array()).await
    }

    /// A sliced Delta keeps whole chunks and carries a nonzero offset, which the decode
    /// applies only after the chunks are decoded.
    #[rstest]
    #[case::within_first_chunk(5, 100)]
    #[case::across_chunks(1000, 1500)]
    #[crate::test]
    async fn test_cuda_delta_sliced(#[case] start: usize, #[case] end: usize) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = PrimitiveArray::new(
            Buffer::from_iter((0..3000u32).map(|i| i * 3)),
            Validity::NonNullable,
        );

        let delta = Delta::try_from_primitive_array(&primitive, &mut ctx)?
            .into_array()
            .slice(start..end)?
            .optimize_ctx(&SESSION)?;
        let delta = delta
            .try_downcast::<Delta>()
            .ok()
            .vortex_expect("slice fixture must exercise the Delta GPU decoder");
        assert_gpu_matches_expected(delta, primitive.slice(start..end)?).await
    }

    /// Nulls ride alongside the values, and the validity is narrowed to the logical slice.
    #[crate::test]
    async fn test_cuda_delta_nullable() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = PrimitiveArray::from_option_iter(
            (0..3000u32).map(|value| (value % 7 != 0).then_some(value)),
        );

        let delta = Delta::try_from_primitive_array(&primitive, &mut ctx)?;
        assert_gpu_matches_expected(delta, primitive.into_array()).await
    }
}
