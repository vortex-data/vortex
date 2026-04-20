// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! GPU patches loading for fused exception patching during bit-unpacking.

use num_traits::ToPrimitive;
use vortex::array::buffer::BufferHandle;
use vortex::buffer::Buffer;
use vortex::buffer::BufferMut;
use vortex::dtype::PType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::patches::Patches;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::CudaExecutionCtx;
use crate::executor::CudaArrayExt;

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

/// Load patches for GPU use.
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
    let array_len = patches.array_len();

    // Get or compute chunk_offsets
    let Some(co) = patches.chunk_offsets() else {
        vortex_bail!("cannot execute_cuda for patched BitPacked array without chunk_offsets")
    };

    let (chunk_offsets, chunk_offset_ptype) = {
        let co_canonical = co.clone().execute_cuda(ctx).await?.into_primitive();
        let ptype = co_canonical.ptype();
        (co_canonical.buffer_handle().clone(), ptype)
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
    let n_chunks = array_len.div_ceil(1024);

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
