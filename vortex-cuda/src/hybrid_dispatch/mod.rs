// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Hybrid dispatch: fuses dynamic-dispatch plans with single-kernel fallbacks.
//!
//! When an array is executed on the GPU, we fuse as much of its encoding
//! tree as possible into a single kernel launch via [`DispatchPlan`].
//! Unfusable subtrees are identified automatically by [`DispatchPlan::new`],
//! executed by their own kernels, and their outputs fed back into the fused
//! plan as `LOAD` sources via `FusedPlan::materialize_with_subtrees`.
//!
//! ```text
//!   Dict                       <-- fusable
//!   ├── codes: FoR(BP)         <-- fusable
//!   └── values: Zstd(FoR(BP)) <-- Zstd is NOT fusable (unfusable node)
//!                 └── FoR(BP)  <-- fusable (fuses inside the subtree)
//! ```
//!
//! Strategies tried in order:
//!
//! 1. Fully fused — no unfusable nodes, entire tree compiles into one
//!    [`DispatchPlan`] → `MaterializedPlan` → kernel launch.
//!
//! 2. Partial fusion — `pending_subtrees` from the `PartiallyFused`
//!    variant are executed first (sequentially, same stream), their device buffers
//!    become `LOAD` ops in a fused plan via `FusedPlan::materialize_with_subtrees`.
//!    Each subtree re-enters [`try_gpu_dispatch`] and may itself fuse.
//!
//! 3. Fallback — root is not fusable. Delegate to its registered
//!    `CudaExecute` kernel; its children re-enter [`try_gpu_dispatch`].
//!
//! All three compose recursively to arbitrary depth.
//!
//! Zone-map pruning is handled by ZonedReader before chunks reach the plan
//! builder. Filtering within a chunk is done after decompression, not as push-down.
//!
//! ```text
//!   ZonedReader (zone-map pruning, skips whole chunks)
//!     └── CudaFlatReader (per chunk)
//!           └── try_gpu_dispatch
//!                 └── FilterExecutor (CUB DeviceSelect on full output)
//! ```

use tracing::trace;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::dtype::PType;
use vortex::error::VortexResult;
use vortex::error::vortex_err;

use crate::dynamic_dispatch::plan_builder::DispatchPlan;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecutionCtx;

/// Try to execute `array` on the GPU, attempting three strategies in order:
///
/// 1. Fully fused — [`DispatchPlan::new`] + `FusedPlan::materialize`.
/// 2. Partially fused — pending subtrees executed first, then
///    `FusedPlan::materialize_with_subtrees`.
/// 3. Fallback — root encoding's `CudaExecute` kernel; children
///    re-enter this function recursively.
///
/// Returns `Ok(Canonical)` on success. Returns `Err` when the array
/// cannot be handled (non-primitive output dtype, no registered kernel).
#[allow(clippy::cognitive_complexity)]
pub async fn try_gpu_dispatch(
    array: &ArrayRef,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical> {
    trace!(encoding = %array.encoding_id(), dtype = ?array.dtype(), len = array.len(), "try GPU dispatch");

    match DispatchPlan::new(array)? {
        DispatchPlan::Fused(plan) => {
            let output_ptype = PType::try_from(array.dtype())?;
            let materialized = plan.materialize(ctx)?;
            let num_stages = materialized.dispatch_plan.num_stages();
            trace!(encoding = %array.encoding_id(), num_stages, "fully-fused dispatch");
            materialized.execute(output_ptype, array.len(), ctx)
        }
        DispatchPlan::PartiallyFused {
            plan,
            pending_subtrees,
        } => {
            let output_ptype = PType::try_from(array.dtype())?;
            let mut subtree_buffers = Vec::with_capacity(pending_subtrees.len());

            // TODO(0ax1): execute subtrees concurrently using separate CUDA streams.
            for subtree in &pending_subtrees {
                let canonical = subtree.clone().execute_cuda(ctx).await?;
                subtree_buffers.push(canonical.into_primitive().into_data().into_parts().buffer);
            }

            let num_subtrees = subtree_buffers.len();
            let materialized = plan.materialize_with_subtrees(subtree_buffers, ctx)?;
            let num_stages = materialized.dispatch_plan.num_stages();
            trace!(encoding = %array.encoding_id(), num_stages, num_subtrees, "partially-fused dispatch");
            materialized.execute(output_ptype, array.len(), ctx)
        }
        DispatchPlan::Unfused => {
            // Unfused kernel dispatch fallback.
            ctx.cuda_session()
                .kernel(&array.encoding_id())
                .ok_or_else(|| {
                    vortex_err!("No CUDA kernel for encoding {:?}", array.encoding_id())
                })?
                .execute(array.clone(), ctx)
                .await
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::assert_arrays_eq;
    use vortex::array::validity::Validity::NonNullable;
    use vortex::buffer::Buffer;
    use vortex::encodings::fastlanes::BitPacked;
    use vortex::encodings::fastlanes::FoR;
    use vortex::error::VortexExpect;
    use vortex::error::VortexResult;
    use vortex::mask::Mask;
    use vortex::session::VortexSession;

    use crate::CanonicalCudaExt;
    use crate::executor::CudaArrayExt;
    use crate::session::CudaSession;

    /// FoR(BitPacked) u32 — entire tree compiles into a single fused plan.
    #[crate::test]
    async fn test_fused() -> VortexResult<()> {
        let mut ctx =
            CudaSession::create_execution_ctx(&VortexSession::empty()).vortex_expect("ctx");
        let values: Vec<u32> = (0..2048).map(|i| (i % 128) as u32).collect();
        let bp = BitPacked::encode(
            &PrimitiveArray::new(Buffer::from(values), NonNullable).into_array(),
            7,
        )
        .vortex_expect("bp");
        let arr = FoR::try_new(bp.into_array(), 1000u32.into()).vortex_expect("for");

        let cpu = arr.to_canonical()?.into_array();
        let gpu = arr
            .into_array()
            .execute_cuda(&mut ctx)
            .await?
            .into_host()
            .await?
            .into_array();
        assert_arrays_eq!(cpu, gpu);
        Ok(())
    }

    /// ALP(FoR(BP)) f32 — fully fused, output is f32 but kernel runs as u32.
    /// Exercises the unsigned type reinterpretation in CudaDispatchPlan::execute.
    #[crate::test]
    async fn test_fused_f32() -> VortexResult<()> {
        use vortex::encodings::alp::ALP;
        use vortex::encodings::alp::Exponents;

        let mut ctx =
            CudaSession::create_execution_ctx(&VortexSession::empty()).vortex_expect("ctx");
        let encoded: Vec<i32> = (0i32..2048).map(|i| i % 500).collect();
        let bp = BitPacked::encode(
            &PrimitiveArray::new(Buffer::from(encoded), NonNullable).into_array(),
            9,
        )
        .vortex_expect("bp");
        let alp = ALP::try_new(
            FoR::try_new(bp.into_array(), 0i32.into())
                .vortex_expect("for")
                .into_array(),
            Exponents { e: 0, f: 2 },
            None,
        )?;

        let cpu = alp.to_canonical()?.into_array();
        let gpu = alp
            .into_array()
            .execute_cuda(&mut ctx)
            .await?
            .into_host()
            .await?
            .into_array();
        assert_arrays_eq!(cpu, gpu);
        Ok(())
    }

    /// ALP with patches — plan builder rejects it, falls back to ALPExecutor.
    #[crate::test]
    async fn test_fallback() -> VortexResult<()> {
        use vortex::array::patches::Patches;
        use vortex::array::validity::Validity::NonNullable as NN;
        use vortex::buffer::buffer;
        use vortex::encodings::alp::ALP;
        use vortex::encodings::alp::Exponents;

        let mut ctx =
            CudaSession::create_execution_ctx(&VortexSession::empty()).vortex_expect("ctx");
        let encoded = PrimitiveArray::new(
            Buffer::from((0i32..2048).map(|i| i % 500).collect::<Vec<_>>()),
            NonNullable,
        )
        .into_array();
        let patches = Patches::new(
            2048,
            0,
            PrimitiveArray::new(buffer![0u32, 1024u32], NN).into_array(),
            PrimitiveArray::new(buffer![99.9f32, 88.8f32], NN).into_array(),
            None,
        )
        .unwrap();
        let arr = ALP::try_new(encoded, Exponents { e: 0, f: 2 }, Some(patches))?;

        let cpu = arr.to_canonical()?.into_array();
        let gpu = arr
            .into_array()
            .execute_cuda(&mut ctx)
            .await?
            .into_host()
            .await?
            .into_array();
        assert_arrays_eq!(cpu, gpu);
        Ok(())
    }

    /// Dict(values=ZstdBuffers(FoR(BP)), codes=FoR(BP)) — ZstdBuffers is
    /// executed separately, then Dict+FoR+BP fuses with its output as a LOAD.
    /// 3 launches: nvcomp + fused FoR+BP + fused LOAD+FoR+BP+DICT.
    #[cfg(feature = "unstable_encodings")]
    #[crate::test]
    async fn test_partial_fusion() -> VortexResult<()> {
        use vortex::array::arrays::DictArray;
        use vortex::array::session::ArraySessionExt;
        use vortex::encodings::fastlanes;
        use vortex::encodings::zstd::ZstdBuffers;
        use vortex::encodings::zstd::ZstdBuffersData;

        let session = VortexSession::empty();
        fastlanes::initialize(&session);
        session.arrays().register(ZstdBuffers);
        let mut ctx = CudaSession::create_execution_ctx(&session).vortex_expect("ctx");

        let num_values: u32 = 64;
        let len: u32 = 2048;

        // values = ZstdBuffers(FoR(BitPacked))
        let vals = PrimitiveArray::new(
            Buffer::from((0..num_values).collect::<Vec<_>>()),
            NonNullable,
        )
        .into_array();
        let vals = FoR::try_new(
            BitPacked::encode(&vals, 6).vortex_expect("bp").into_array(),
            0u32.into(),
        )
        .vortex_expect("for");
        let vals = ZstdBuffersData::compress(&vals.into_array(), 3).vortex_expect("zstd");

        // codes = FoR(BitPacked)
        let codes = PrimitiveArray::new(
            Buffer::from((0..len).map(|i| i % num_values).collect::<Vec<_>>()),
            NonNullable,
        )
        .into_array();
        let codes = FoR::try_new(
            BitPacked::encode(&codes, 6)
                .vortex_expect("bp")
                .into_array(),
            0u32.into(),
        )
        .vortex_expect("for");

        let dict = DictArray::try_new(codes.into_array(), vals.into_array()).vortex_expect("dict");

        let cpu = PrimitiveArray::new(
            Buffer::from((0..len).map(|i| i % num_values).collect::<Vec<_>>()),
            NonNullable,
        )
        .into_array();
        let gpu = dict
            .into_array()
            .execute_cuda(&mut ctx)
            .await?
            .into_host()
            .await?
            .into_array();
        assert_arrays_eq!(cpu, gpu);
        Ok(())
    }

    /// Filter(FoR(BP), mask) — FoR+BP fuses via dyn dispatch, then CUB filters the result.
    #[crate::test]
    async fn test_filter_fused_child() -> VortexResult<()> {
        let mut ctx =
            CudaSession::create_execution_ctx(&VortexSession::empty()).vortex_expect("ctx");

        let len = 2048u32;
        let data: Vec<u32> = (0..len).map(|i| i % 128).collect();
        let bp = BitPacked::encode(
            &PrimitiveArray::new(Buffer::from(data.clone()), NonNullable).into_array(),
            7,
        )
        .vortex_expect("bp");
        let for_arr = FoR::try_new(bp.into_array(), 100u32.into()).vortex_expect("for");

        // Keep every other element.
        let mask = Mask::from_iter((0..len as usize).map(|i| i % 2 == 0));
        let filtered = for_arr.into_array().filter(mask)?;

        let expected: Vec<u32> = data.iter().step_by(2).map(|v| v + 100).collect();
        let cpu = PrimitiveArray::new(Buffer::from(expected), NonNullable).into_array();
        let gpu = filtered
            .execute_cuda(&mut ctx)
            .await?
            .into_host()
            .await?
            .into_array();
        assert_arrays_eq!(cpu, gpu);
        Ok(())
    }
}
