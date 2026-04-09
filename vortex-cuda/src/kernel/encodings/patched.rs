// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

use async_trait::async_trait;
use cudarc::driver::DeviceRepr;
use cudarc::driver::PushKernelArg;
use tracing::instrument;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::primitive::PrimitiveDataParts;
use vortex::array::match_each_integer_ptype;
use vortex::dtype::NativePType;
use vortex::encodings::fastlanes::BitPacked;
use vortex::encodings::fastlanes::BitPackedArrayExt;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex_array::arrays::PatchedArray;
use vortex_array::arrays::patched::Patched;
use vortex_array::arrays::patched::PatchedArrayExt;
use vortex_array::arrays::patched::PatchedArraySlotsExt;
use vortex_array::buffer::BufferHandle;
use vortex_array::match_each_native_simd_ptype;

use crate::CudaBufferExt;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;
use crate::kernel::encodings::bitpacked::decode_bitpacked;
use crate::kernel::patches::types::DevicePatches;

/// CUDA decoder for Patched arrays.
///
/// When the inner child is BitPacked, fuses patching with bit-unpacking to avoid
/// an additional kernel dispatch.
#[derive(Debug)]
pub(crate) struct PatchedExecutor;

impl PatchedExecutor {
    fn try_specialize(array: ArrayRef) -> Option<PatchedArray> {
        array.try_downcast::<Patched>().ok()
    }
}

#[async_trait]
impl CudaExecute for PatchedExecutor {
    #[instrument(level = "trace", skip_all, fields(executor = ?self))]
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let array =
            Self::try_specialize(array).ok_or_else(|| vortex_err!("Expected PatchedArray"))?;

        // Check if the inner child is BitPacked - if so, we can fuse patching with unpacking
        if let Some(bitpacked) = array.inner().as_opt::<BitPacked>() {
            // The inner BitPacked should not have its own interior patches since they've
            // been externalized into the Patched wrapper
            if bitpacked.patches().is_some() {
                return Err(vortex_err!(
                    "Patched(BitPacked) should not have interior patches in BitPacked child"
                ));
            }

            // Execute the components
            let lane_offsets = array
                .lane_offsets()
                .clone()
                .execute_cuda(ctx)
                .await?
                .into_primitive()
                .into_data_parts()
                .buffer;

            let patch_indices = array
                .patch_indices()
                .clone()
                .execute_cuda(ctx)
                .await?
                .into_primitive()
                .into_data_parts()
                .buffer;

            let patch_values = array
                .patch_values()
                .clone()
                .execute_cuda(ctx)
                .await?
                .into_primitive()
                .into_data_parts()
                .buffer;

            match_each_integer_ptype!(bitpacked.ptype(bitpacked.dtype()), |P| {
                return decode_bitpacked::<P>(
                    bitpacked.into_owned(),
                    P::default(),
                    Some(DevicePatches {
                        lane_offsets: ctx.ensure_on_device(lane_offsets).await?,
                        indices: ctx.ensure_on_device(patch_indices).await?,
                        values: ctx.ensure_on_device(patch_values).await?,
                    }),
                    ctx,
                )
                .await;
            })
        }

        // Fallback: execute inner on GPU, then apply patches using GPU kernel
        let n_lanes = array.n_lanes();
        let offset = array.offset();
        let len = array.as_ref().len();

        // Execute inner on GPU to get the base values
        let inner_canonical = array.inner().clone().execute_cuda(ctx).await?;
        let inner_primitive = inner_canonical.into_primitive();
        let validity = inner_primitive.validity()?.clone();
        let ptype = inner_primitive.ptype();

        // Get the inner buffer on device
        let PrimitiveDataParts { buffer, .. } = inner_primitive.into_data_parts();
        let d_output = ctx.ensure_on_device(buffer).await?;

        // Execute patch components on GPU
        let lane_offsets = array.lane_offsets().clone().execute_cuda(ctx).await?;
        let lane_offsets_prim = lane_offsets.into_primitive();

        // one thread per lane, i.e. lane_offsets.len() - 1
        let n_threads = lane_offsets_prim.len().saturating_sub(1);

        let PrimitiveDataParts {
            buffer: lane_offsets_buffer,
            ..
        } = lane_offsets_prim.into_data_parts();
        let d_lane_offsets = ctx.ensure_on_device(lane_offsets_buffer).await?;

        let patch_indices = array.patch_indices().clone().execute_cuda(ctx).await?;
        let patch_indices_prim = patch_indices.into_primitive();
        let PrimitiveDataParts {
            buffer: patch_indices_buffer,
            ..
        } = patch_indices_prim.into_data_parts();
        let d_patch_indices = ctx.ensure_on_device(patch_indices_buffer).await?;

        let patch_values = array.patch_values().clone().execute_cuda(ctx).await?;
        let patch_values_prim = patch_values.into_primitive();
        let PrimitiveDataParts {
            buffer: patch_values_buffer,
            ..
        } = patch_values_prim.into_data_parts();
        let d_patch_values = ctx.ensure_on_device(patch_values_buffer).await?;

        // Apply patches on GPU using thread-per-lane model
        match_each_native_simd_ptype!(ptype, |V| {
            let patched_buffer = execute_patched::<V>(
                d_output,
                d_lane_offsets,
                d_patch_indices,
                d_patch_values,
                n_threads,
                n_lanes,
                offset,
                len,
                ctx,
            )?;

            Ok(Canonical::Primitive(PrimitiveArray::from_buffer_handle(
                patched_buffer,
                ptype,
                validity,
            )))
        })
    }
}

/// Apply patches to an output buffer using the Patched array GPU kernel.
///
/// Uses a thread-per-lane model where each thread handles one (chunk, lane) slot
/// and applies all patches in that slot.
///
/// `n_threads` is the number of threads to execute the kernel with, which should
/// be equal to `lane_offsets.len() - 1`, i.e. one per lane.
#[instrument(skip_all)]
#[allow(clippy::too_many_arguments)]
fn execute_patched<T: NativePType + DeviceRepr>(
    output: BufferHandle,
    lane_offsets: BufferHandle,
    patch_indices: BufferHandle,
    patch_values: BufferHandle,
    n_threads: usize,
    n_lanes: usize,
    offset: usize,
    len: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<BufferHandle> {
    if n_threads == 0 {
        // No lanes to process
        return Ok(output);
    }

    let d_output_view = output.cuda_view::<T>()?;
    let d_lane_offsets_view = lane_offsets.cuda_view::<u32>()?;
    let d_patch_indices_view = patch_indices.cuda_view::<u16>()?;
    let d_patch_values_view = patch_values.cuda_view::<T>()?;

    let n_lanes_u32 = u32::try_from(n_lanes)?;
    let total_lane_slots_u32 = u32::try_from(n_threads)?;
    let offset_u64 = offset as u64;
    let len_u64 = len as u64;

    let kernel_func = ctx.load_function("patched", &[T::PTYPE])?;

    // Launch with one thread per lane slot
    ctx.launch_kernel(&kernel_func, n_threads, |args| {
        args.arg(&d_output_view)
            .arg(&d_lane_offsets_view)
            .arg(&d_patch_indices_view)
            .arg(&d_patch_values_view)
            .arg(&n_lanes_u32)
            .arg(&total_lane_slots_u32)
            .arg(&offset_u64)
            .arg(&len_u64);
    })?;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::assert_arrays_eq;
    use vortex::array::validity::Validity::NonNullable;
    use vortex::buffer::Buffer;
    use vortex::encodings::fastlanes::BitPacked;
    use vortex::error::VortexExpect;
    use vortex::error::VortexResult;
    use vortex::session::VortexSession;
    use vortex_array::ExecutionCtx;
    use vortex_array::arrays::Patched;
    use vortex_array::patches::Patches;

    use super::*;
    use crate::CanonicalCudaExt;
    use crate::session::CudaSession;

    #[crate::test]
    fn test_patched_bitpacked() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Create a primitive array with values that all fit in 6 bits (0-63)
        // We'll add patches for some positions manually
        let mut values: Vec<u16> = (0u16..64).cycle().take(2048).collect();
        // Set the patch positions to filler values (0)
        values[100] = 0;
        values[500] = 0;
        values[1000] = 0;
        values[1500] = 0;

        let array = PrimitiveArray::new(Buffer::from(values), NonNullable);

        // Encode with 6 bits - all values fit, so no internal patches
        let bp_array = BitPacked::encode(&array.into_array(), 6)?;
        assert!(bp_array.patches().is_none());

        // Create patches for the positions we zeroed out
        let patches = Patches::new(
            2048,
            0,
            PrimitiveArray::from_iter([100u32, 500, 1000, 1500]).into_array(),
            PrimitiveArray::from_iter([1000u16, 2000, 3000, 4000]).into_array(),
            None,
        )?;

        let session = VortexSession::empty();
        let mut exec_ctx = ExecutionCtx::new(session);
        let patched_array =
            Patched::from_array_and_patches(bp_array.into_array(), &patches, &mut exec_ctx)?;

        let cpu_result = patched_array.to_canonical()?.into_array();

        let gpu_result = block_on(async {
            PatchedExecutor
                .execute(patched_array.into_array(), &mut cuda_ctx)
                .await
                .vortex_expect("GPU decompression failed")
                .into_host()
                .await
                .map(|a| a.into_array())
        })?;

        assert_arrays_eq!(cpu_result, gpu_result);

        Ok(())
    }

    #[crate::test]
    fn test_patched_primitive_fallback() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Create a primitive array
        let values = PrimitiveArray::new((0u16..1024).collect::<Buffer<_>>(), NonNullable);

        // Create patches for some values
        let patches = Patches::new(
            1024,
            0,
            PrimitiveArray::from_iter([100u32, 200, 300, 400]).into_array(),
            PrimitiveArray::from_iter([9999u16, 8888, 7777, 6666]).into_array(),
            None,
        )?;

        let session = VortexSession::empty();
        let mut exec_ctx = ExecutionCtx::new(session);
        let patched_array =
            Patched::from_array_and_patches(values.into_array(), &patches, &mut exec_ctx)?;

        let cpu_result = patched_array.to_canonical()?.into_array();

        // This should use the GPU kernel fallback since inner is Primitive, not BitPacked
        let gpu_result = block_on(async {
            PatchedExecutor
                .execute(patched_array.into_array(), &mut cuda_ctx)
                .await
                .vortex_expect("GPU decompression failed")
                .into_host()
                .await
                .map(|a| a.into_array())
        })?;

        assert_arrays_eq!(cpu_result, gpu_result);

        Ok(())
    }

    #[crate::test]
    fn test_patched_bitpacked_sliced() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Create a primitive array with values that all fit in 6 bits (0-63)
        let mut values: Vec<u16> = (0u16..64).cycle().take(2048).collect();
        // Set the patch positions to filler values (0)
        values[100] = 0;
        values[500] = 0;
        values[1000] = 0;
        values[1500] = 0;

        let array = PrimitiveArray::new(Buffer::from(values), NonNullable);

        // Encode with 6 bits - all values fit, so no internal patches
        let bp_array = BitPacked::encode(&array.into_array(), 6)?;
        assert!(bp_array.patches().is_none());

        // Create patches for the positions we zeroed out
        let patches = Patches::new(
            2048,
            0,
            PrimitiveArray::from_iter([100u32, 500, 1000, 1500]).into_array(),
            PrimitiveArray::from_iter([1000u16, 2000, 3000, 4000]).into_array(),
            None,
        )?;

        let session = VortexSession::empty();
        let mut exec_ctx = ExecutionCtx::new(session);
        let patched_array =
            Patched::from_array_and_patches(bp_array.into_array(), &patches, &mut exec_ctx)?;

        // Slice starting after the first patch but before the second
        let sliced = patched_array.into_array().slice(200..1800)?;

        let cpu_result = sliced.to_canonical()?.into_array();

        let gpu_result = block_on(async {
            PatchedExecutor
                .execute(sliced, &mut cuda_ctx)
                .await
                .vortex_expect("GPU decompression failed")
                .into_host()
                .await
                .map(|a| a.into_array())
        })?;

        assert_arrays_eq!(cpu_result, gpu_result);

        Ok(())
    }

    #[crate::test]
    fn test_patched_bitpacked_multi_chunk() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Create a large array spanning multiple FL chunks (each chunk is 1024 elements)
        // 5 chunks = 5120 elements
        let mut values: Vec<u32> = (0u32..256).cycle().take(5120).collect();
        // Set patch positions to filler values - one in each chunk
        values[100] = 0; // chunk 0
        values[1200] = 0; // chunk 1
        values[2300] = 0; // chunk 2
        values[3400] = 0; // chunk 3
        values[4500] = 0; // chunk 4

        let array = PrimitiveArray::new(Buffer::from(values), NonNullable);

        // Encode with 8 bits - all values fit (0-255), so no internal patches
        let bp_array = BitPacked::encode(&array.into_array(), 8)?;
        assert!(bp_array.patches().is_none());

        // Create patches across multiple chunks
        let patches = Patches::new(
            5120,
            0,
            PrimitiveArray::from_iter([100u32, 1200, 2300, 3400, 4500]).into_array(),
            PrimitiveArray::from_iter([10000u32, 20000, 30000, 40000, 50000]).into_array(),
            None,
        )?;

        let session = VortexSession::empty();
        let mut exec_ctx = ExecutionCtx::new(session);
        let patched_array =
            Patched::from_array_and_patches(bp_array.into_array(), &patches, &mut exec_ctx)?;

        let cpu_result = patched_array.to_canonical()?.into_array();

        let gpu_result = block_on(async {
            PatchedExecutor
                .execute(patched_array.into_array(), &mut cuda_ctx)
                .await
                .vortex_expect("GPU decompression failed")
                .into_host()
                .await
                .map(|a| a.into_array())
        })?;

        assert_arrays_eq!(cpu_result, gpu_result);

        Ok(())
    }

    #[crate::test]
    fn test_patched_bitpacked_multi_chunk_sliced() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Create a large array spanning multiple FL chunks
        let mut values: Vec<u32> = (0u32..256).cycle().take(4096).collect();
        // Set patch positions to filler values - one in each chunk
        values[100] = 0; // chunk 0
        values[1200] = 0; // chunk 1
        values[2300] = 0; // chunk 2
        values[3400] = 0; // chunk 3

        let array = PrimitiveArray::new(Buffer::from(values), NonNullable);

        // Encode with 8 bits
        let bp_array = BitPacked::encode(&array.into_array(), 8)?;
        assert!(bp_array.patches().is_none());

        // Create patches across multiple chunks
        let patches = Patches::new(
            4096,
            0,
            PrimitiveArray::from_iter([100u32, 1200, 2300, 3400]).into_array(),
            PrimitiveArray::from_iter([10000u32, 20000, 30000, 40000]).into_array(),
            None,
        )?;

        let session = VortexSession::empty();
        let mut exec_ctx = ExecutionCtx::new(session);
        let patched_array =
            Patched::from_array_and_patches(bp_array.into_array(), &patches, &mut exec_ctx)?;

        // Slice across chunk boundaries (from middle of chunk 1 to middle of chunk 3)
        let sliced = patched_array.into_array().slice(1500..3000)?;

        let cpu_result = sliced.to_canonical()?.into_array();

        let gpu_result = block_on(async {
            PatchedExecutor
                .execute(sliced, &mut cuda_ctx)
                .await
                .vortex_expect("GPU decompression failed")
                .into_host()
                .await
                .map(|a| a.into_array())
        })?;

        assert_arrays_eq!(cpu_result, gpu_result);

        Ok(())
    }
}
