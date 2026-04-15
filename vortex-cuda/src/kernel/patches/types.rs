// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! GPU patches loading for fused exception patching during bit-unpacking.

use num_traits::ToPrimitive;
use vortex::array::buffer::BufferHandle;
use vortex::buffer::Buffer;
use vortex::buffer::BufferMut;
use vortex::dtype::PType;
use vortex::dtype::UnsignedPType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::patches::Patches;
use vortex_error::VortexResult;

use crate::CudaExecutionCtx;
use crate::executor::CudaArrayExt;

/// A set of device-resident patches.
pub struct DevicePatches {
    pub(crate) chunk_offsets: BufferHandle,
    pub(crate) chunk_offset_ptype: PType,
    pub(crate) indices: BufferHandle,
    pub(crate) values: BufferHandle,
    pub(crate) offset: usize,
}

/// Load patches for GPU use.
///
/// If chunk_offsets is not present in the patches, computes it by scanning indices.
/// Indices are kept as-is (u32); the kernel computes within-chunk offsets at runtime.
pub async fn load_patches(
    patches: &Patches,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<DevicePatches> {
    let offset = patches.offset();
    let array_len = patches.array_len();

    // Get or compute chunk_offsets
    let (chunk_offsets, chunk_offset_ptype) = if let Some(co) = patches.chunk_offsets() {
        let co_canonical = co.clone().execute_cuda(ctx).await?.into_primitive();
        let ptype = co_canonical.ptype();
        (co_canonical.buffer_handle().clone(), ptype)
    } else {
        // Build chunk_offsets by scanning indices
        let chunk_offsets = build_chunk_offsets(patches, array_len, ctx).await?;
        (
            BufferHandle::new_host(chunk_offsets.freeze().into_byte_buffer()),
            PType::U32,
        )
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

    Ok(DevicePatches {
        chunk_offsets,
        chunk_offset_ptype,
        indices,
        values,
        offset,
    })
}

/// Build chunk_offsets by scanning indices when not provided.
async fn build_chunk_offsets(
    patches: &Patches,
    array_len: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<BufferMut<u32>> {
    let indices = patches
        .indices()
        .clone()
        .execute_cuda(ctx)
        .await?
        .into_primitive();

    let offset = patches.offset();
    let n_chunks = array_len.div_ceil(1024);

    let indices_ptype = indices.ptype();
    let indices_buf = indices.buffer_handle().to_host().await;

    match_each_unsigned_integer_ptype!(indices_ptype, |I| {
        let indices_slice: Buffer<I> = Buffer::from_byte_buffer(indices_buf);
        Ok(compute_chunk_offsets(
            indices_slice.as_slice(),
            offset,
            n_chunks,
        ))
    })
}

#[expect(clippy::cast_possible_truncation, clippy::expect_used)]
fn compute_chunk_offsets<I: UnsignedPType + ToPrimitive>(
    indices: &[I],
    offset: usize,
    n_chunks: usize,
) -> BufferMut<u32> {
    let mut chunk_offsets: BufferMut<u32> = BufferMut::zeroed(n_chunks + 1);

    // For each patch, determine which chunk it belongs to
    for (i, &idx) in indices.iter().enumerate() {
        let absolute_idx: usize = idx.to_usize().expect("index should fit in usize");
        let chunk = (absolute_idx - offset) / 1024;
        // Update offsets for all chunks after this patch's chunk
        // Since indices are sorted, we can just set the end offset for subsequent chunks
        for c in (chunk + 1)..chunk_offsets.len() {
            if chunk_offsets[c] < (i + 1) as u32 {
                chunk_offsets[c] = (i + 1) as u32;
            }
        }
    }

    chunk_offsets
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::patches::Patches;
    use vortex_error::VortexResult;

    use super::compute_chunk_offsets;

    #[test]
    fn test_compute_chunk_offsets_single_chunk() {
        // All patches in chunk 0
        let indices: &[u32] = &[0, 100, 500, 1000];
        let offsets = compute_chunk_offsets(indices, 0, 2);
        assert_eq!(offsets.as_slice(), &[0, 4, 4]);
    }

    #[test]
    fn test_compute_chunk_offsets_multiple_chunks() {
        // Patches spread across chunks
        let indices: &[u32] = &[0, 500, 1024, 1500, 2048, 3072];
        let offsets = compute_chunk_offsets(indices, 0, 4);
        // Chunk 0: indices 0, 500 (positions 0..2)
        // Chunk 1: indices 1024, 1500 (positions 2..4)
        // Chunk 2: indices 2048 (positions 4..5)
        // Chunk 3: indices 3072 (positions 5..6)
        assert_eq!(offsets.as_slice(), &[0, 2, 4, 5, 6]);
    }

    #[test]
    fn test_compute_chunk_offsets_with_offset() {
        // Patches with array offset
        let indices: &[u32] = &[1024, 1500, 2048];
        let offsets = compute_chunk_offsets(indices, 1024, 2);
        // After subtracting offset 1024:
        // Chunk 0: indices 0, 476 (positions 0..2)
        // Chunk 1: index 1024 (positions 2..3)
        assert_eq!(offsets.as_slice(), &[0, 2, 3]);
    }

    #[test]
    fn test_compute_chunk_offsets_empty_chunks() {
        // Patches skip some chunks
        let indices: &[u32] = &[0, 3072];
        let offsets = compute_chunk_offsets(indices, 0, 4);
        // Chunk 0: index 0 (positions 0..1)
        // Chunk 1: empty (positions 1..1)
        // Chunk 2: empty (positions 1..1)
        // Chunk 3: index 3072 (positions 1..2)
        assert_eq!(offsets.as_slice(), &[0, 1, 1, 1, 2]);
    }

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
