// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! GPU patches loading for fused exception patching during bit-unpacking.

use std::mem::size_of;

use num_traits::ToPrimitive;
use vortex::array::buffer::BufferHandle;
use vortex::buffer::Alignment;
use vortex::buffer::Buffer;
use vortex::buffer::BufferMut;
use vortex::buffer::ByteBufferMut;
use vortex::dtype::PType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::patches::Patches;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::CudaBufferExt;
use crate::CudaExecutionCtx;
use crate::executor::CudaArrayExt;
use crate::kernel::patches::gpu::ChunkOffsetType;
use crate::kernel::patches::gpu::ChunkOffsetType_CO_U8;
use crate::kernel::patches::gpu::ChunkOffsetType_CO_U16;
use crate::kernel::patches::gpu::ChunkOffsetType_CO_U32;
use crate::kernel::patches::gpu::ChunkOffsetType_CO_U64;
use crate::kernel::patches::gpu::GPUPatches;

/// A set of device-resident patches.
pub struct DevicePatches {
    pub(crate) chunk_offsets: BufferHandle,
    pub(crate) chunk_offset_ptype: PType,
    pub(crate) indices: BufferHandle,
    pub(crate) values: BufferHandle,
    pub(crate) offset: usize,
    pub(crate) offset_within_chunk: usize,
    pub(crate) num_patches: usize,
    pub(crate) n_chunks: usize,
}

/// Load patches for GPU use (async).
///
/// # Errors
///
/// If the patches do not have `chunk_offsets`. They have been written by
/// default in new Vortex files since 0.54.0.
pub(crate) async fn load_patches(
    patches: &Patches,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<DevicePatches> {
    let offset = patches.offset();
    let offset_within_chunk = patches.offset_within_chunk().unwrap_or_default();
    // Get or compute chunk_offsets
    let Some(co) = patches.chunk_offsets() else {
        vortex_bail!("cannot execute_cuda for patched BitPacked array without chunk_offsets")
    };

    let (chunk_offsets, chunk_offset_ptype, n_chunks) = {
        let co_canonical = co.clone().execute_cuda(ctx).await?.into_primitive();
        let ptype = co_canonical.ptype();
        let len = co_canonical.len();
        (co_canonical.buffer_handle().clone(), ptype, len)
    };

    // Load indices - must be converted to u32 for GPU use
    let indices = patches
        .indices()
        .clone()
        .execute_cuda(ctx)
        .await?
        .into_primitive();
    let indices_ptype = indices.ptype();
    #[expect(clippy::expect_used)]
    let indices = if indices_ptype == PType::U32 {
        indices.buffer_handle().clone()
    } else {
        // Convert indices to u32
        let indices_buf = indices.buffer_handle().to_host().await;
        let indices_u32 = match_each_unsigned_integer_ptype!(indices_ptype, |I| {
            let src: Buffer<I> = Buffer::from_byte_buffer(indices_buf);
            let mut dst: BufferMut<u32> = BufferMut::with_capacity(src.len());
            for &idx in src.as_slice() {
                // Indices are limited to u32 range for GPU
                dst.push(idx.to_u32().expect("index should fit in u32"));
            }
            dst.freeze()
        });
        BufferHandle::new_host(indices_u32.into_byte_buffer())
    };

    // Load values
    let values = patches
        .values()
        .clone()
        .execute_cuda(ctx)
        .await?
        .into_primitive();

    // Ensure all on device
    let chunk_offsets = ctx.ensure_on_device(chunk_offsets).await?;
    let indices = ctx.ensure_on_device(indices).await?;
    let values = ctx.ensure_on_device(values.buffer_handle().clone()).await?;

    let num_patches = patches.num_patches();

    Ok(DevicePatches {
        chunk_offsets,
        chunk_offset_ptype,
        indices,
        values,
        offset,
        offset_within_chunk,
        num_patches,
        n_chunks,
    })
}

/// Convert a PType to the corresponding `ChunkOffsetType` for GPU patches.
pub(crate) fn ptype_to_chunk_offset_type(ptype: PType) -> VortexResult<ChunkOffsetType> {
    match ptype {
        PType::U8 => Ok(ChunkOffsetType_CO_U8),
        PType::U16 => Ok(ChunkOffsetType_CO_U16),
        PType::U32 => Ok(ChunkOffsetType_CO_U32),
        PType::U64 => Ok(ChunkOffsetType_CO_U64),
        _ => vortex_bail!("Invalid PType for chunk_offsets: {:?}", ptype),
    }
}

/// Build a [`GPUPatches`] struct from [`DevicePatches`], serialize it to
/// bytes, and upload to the device. Returns the device pointer and a buffer
/// handle that must be kept alive for the kernel launch.
fn build_gpu_patches(
    dp: &DevicePatches,
    ctx: &CudaExecutionCtx,
) -> VortexResult<(BufferHandle, u64)> {
    // Zero-initialize to avoid uninitialized padding bytes (e.g. between
    // chunk_offset_type and indices) which would be UB when serialized.
    let mut gpu_patches: GPUPatches = unsafe { std::mem::zeroed() };
    gpu_patches.chunk_offsets = dp.chunk_offsets.cuda_device_ptr()? as _;
    gpu_patches.chunk_offset_type = ptype_to_chunk_offset_type(dp.chunk_offset_ptype)?;
    gpu_patches.indices = dp.indices.cuda_device_ptr()? as _;
    gpu_patches.values = dp.values.cuda_device_ptr()? as _;
    #[expect(clippy::cast_possible_truncation)]
    {
        gpu_patches.offset = dp.offset as u32;
        gpu_patches.offset_within_chunk = dp.offset_within_chunk as u32;
        gpu_patches.num_patches = dp.num_patches as u32;
        // n_chunks must match the chunk_offsets array length, not array_len / 1024.
        // When patches are sliced, chunk_offsets is sliced to only include chunks
        // overlapping the slice range — matching the CPU's patch_chunk which uses
        // chunk_offsets_slice.len().
        gpu_patches.n_chunks = dp.n_chunks as u32;
    }

    let bytes = unsafe {
        std::slice::from_raw_parts(
            std::ptr::from_ref(&gpu_patches).cast::<u8>(),
            size_of::<GPUPatches>(),
        )
    };
    let mut buf =
        ByteBufferMut::with_capacity_aligned(size_of::<GPUPatches>(), Alignment::of::<u64>());
    buf.extend_from_slice(bytes);
    let gpu_buf = ctx.ensure_on_device_sync(BufferHandle::new_host(buf.freeze()))?;
    let ptr = gpu_buf.cuda_device_ptr()?;
    Ok((gpu_buf, ptr))
}

/// Sync wrapper: load patches via [`load_patches`] (blocking), then build and
/// upload a [`GPUPatches`] struct. Returns the device pointer and all buffer
/// handles that must be kept alive for the kernel launch.
pub(crate) fn load_patches_sync(
    patches: &Patches,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<(u64, Vec<BufferHandle>)> {
    let device_patches = futures::executor::block_on(load_patches(patches, ctx))?;
    let (gpu_buf, ptr) = build_gpu_patches(&device_patches, ctx)?;

    let DevicePatches {
        chunk_offsets,
        indices,
        values,
        ..
    } = device_patches;
    Ok((ptr, vec![chunk_offsets, indices, values, gpu_buf]))
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::patches::Patches;
    use vortex_error::VortexResult;

    #[test]
    fn test_patches_with_chunk_offsets() -> VortexResult<()> {
        // Test creating patches with pre-computed chunk_offsets
        let indices = PrimitiveArray::from_iter([0u32, 500, 1024, 2000]);
        let values = PrimitiveArray::from_iter([10u32, 20, 30, 40]);
        let chunk_offsets = PrimitiveArray::from_iter([0u32, 2, 3, 4]);

        let patches = Patches::new(
            3072,
            0,
            indices.into_array(),
            values.into_array(),
            Some(chunk_offsets.into_array()),
        )?;

        assert!(patches.chunk_offsets().is_some());
        assert_eq!(patches.chunk_offset_at(0)?, 0);
        assert_eq!(patches.chunk_offset_at(1)?, 2);
        assert_eq!(patches.chunk_offset_at(2)?, 3);

        Ok(())
    }
}
