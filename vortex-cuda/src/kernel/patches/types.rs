// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! GPU patches loading for fused exception patching during bit-unpacking.

use std::mem::size_of;
use std::ops::Range;

use num_traits::ToPrimitive;
use vortex::array::buffer::BufferHandle;
use vortex::buffer::Alignment;
use vortex::buffer::Buffer;
use vortex::buffer::BufferMut;
use vortex::buffer::ByteBufferMut;
use vortex::dtype::PType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::patches::PATCH_CHUNK_SIZE;
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
use crate::kernel::patches::gpu::PATCH_DERIVE_INDICES_BASE;

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
    /// Base patch index for chunk offsets when indices/values remain unsliced.
    /// `None` means derive it in the kernel via `PATCH_DERIVE_INDICES_BASE`.
    pub(crate) indices_base: Option<usize>,
}

/// Build [`DevicePatches`] from patches by ensuring their buffers are on GPU.
///
/// This prepares `chunk_offsets`, `indices`, and `values` as device buffers and collects the
/// metadata needed to build a [`GPUPatches`] descriptor. It does not upload the descriptor itself.
///
/// # Errors
///
/// If the patches do not have `chunk_offsets`. They have been written by
/// default in new Vortex files since 0.54.0.
pub(crate) async fn load_device_patches(
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
        indices_base: None,
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

/// Build a [`GPUPatches`] struct from [`DevicePatches`], serialize it to bytes, and upload to the
/// device. Returns the device pointer and a buffer handle that must be kept alive for the kernel
/// launch.
#[expect(clippy::cast_possible_truncation)]
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
    gpu_patches.offset = dp.offset as u32;
    gpu_patches.offset_within_chunk = dp.offset_within_chunk as u32;
    gpu_patches.num_patches = dp.num_patches as u32;
    // n_chunks must match the chunk_offsets array length, not array_len / 1024.
    // When patches are sliced, chunk_offsets is sliced to only include chunks
    // overlapping the slice range — matching the CPU's patch_chunk which uses
    // chunk_offsets_slice.len().
    gpu_patches.n_chunks = dp.n_chunks as u32;
    gpu_patches.indices_base = dp
        .indices_base
        .map_or(PATCH_DERIVE_INDICES_BASE, |base| base as u32);

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

/// Transfers patches to the GPU and builds a [`GPUPatches`] descriptor.
///
/// When `range` is set, this builds a sliced patch view without reading patch metadata on the
/// host. It slices `chunk_offsets` by chunk boundaries, keeps `indices` and `values` unsliced, and
/// records an explicit `indices_base` so the kernel can interpret the sliced chunk offsets against
/// the original patch arrays.
///
/// Returns the device pointer and all buffer handles that must be kept alive for the kernel launch.
pub(crate) async fn load_patches_to_gpu(
    patches: &Patches,
    range: Option<Range<usize>>,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<(u64, Vec<BufferHandle>)> {
    let mut device_patches = load_device_patches(patches, ctx).await?;

    if let Some(range) = range {
        slice_device_patches(patches, range, &mut device_patches);
    }

    let (gpu_buf, ptr) = build_gpu_patches(&device_patches, ctx)?;
    let DevicePatches {
        chunk_offsets,
        indices,
        values,
        ..
    } = device_patches;
    Ok((ptr, vec![chunk_offsets, indices, values, gpu_buf]))
}

fn slice_device_patches(
    patches: &Patches,
    range: Range<usize>,
    device_patches: &mut DevicePatches,
) {
    let offset = patches.offset() + range.start;
    let len = range.len();
    let chunk_start = offset / PATCH_CHUNK_SIZE;
    let chunk_end = offset.saturating_add(len).div_ceil(PATCH_CHUNK_SIZE);
    let first_chunk = patches.offset() / PATCH_CHUNK_SIZE;
    let start = chunk_start.saturating_sub(first_chunk);
    let end = (chunk_end.saturating_sub(first_chunk) + 1).min(device_patches.n_chunks);
    if start < end {
        let width = device_patches.chunk_offset_ptype.byte_width();
        device_patches.chunk_offsets = device_patches
            .chunk_offsets
            .slice(start * width..end * width);
        device_patches.n_chunks = end - start;
    }

    // Keep indices/values unsliced. This is a conservative chunk-aligned
    // patch view for lookup-style consumers: patches outside `range` may be
    // scanned, but their indices cannot match positions inside the range.
    device_patches.offset = offset;
    device_patches.offset_within_chunk = 0;
    device_patches.indices_base = Some(0);
}

#[cfg(test)]
mod tests {
    use vortex::buffer::Buffer;
    use vortex::session::VortexSession;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::patches::Patches;
    use vortex_error::VortexResult;

    use super::load_device_patches;
    use super::slice_device_patches;
    use crate::CudaSession;

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

    #[rstest::rstest]
    #[case::full_range(0..5120, 0, vec![0, 1, 2, 3, 4, 5])]
    #[case::single_first_chunk(0..1024, 0, vec![0, 1])]
    #[case::partial_first_chunk(100..3000, 100, vec![0, 1, 2, 3])]
    #[case::chunk_aligned_start(1024..3000, 1024, vec![1, 2, 3])]
    #[case::single_middle_chunk(2048..3072, 2048, vec![2, 3])]
    #[case::middle_of_chunks(1500..2800, 1500, vec![1, 2, 3])]
    #[case::skip_first_chunks(3072..4500, 3072, vec![3, 4, 5])]
    #[case::tail_chunk(4500..5000, 4500, vec![4, 5])]
    #[crate::test]
    async fn test_slice_device_patches(
        #[case] range: std::ops::Range<usize>,
        #[case] expected_offset: usize,
        #[case] expected_chunk_offsets: Vec<u32>,
    ) -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let indices = PrimitiveArray::from_iter([100u32, 1100, 2100, 3100, 4100]);
        let values = PrimitiveArray::from_iter([10u32, 11, 12, 13, 14]);
        let chunk_offsets = PrimitiveArray::from_iter([0u32, 1, 2, 3, 4, 5]);
        let patches = Patches::new(
            5120,
            0,
            indices.into_array(),
            values.into_array(),
            Some(chunk_offsets.into_array()),
        )?;

        let mut device_patches = load_device_patches(&patches, &mut cuda_ctx).await?;
        slice_device_patches(&patches, range, &mut device_patches);

        let chunk_offsets =
            Buffer::<u32>::from_byte_buffer(device_patches.chunk_offsets.to_host().await);
        assert_eq!(chunk_offsets.as_slice(), expected_chunk_offsets.as_slice());
        assert_eq!(device_patches.n_chunks, expected_chunk_offsets.len());
        assert_eq!(device_patches.offset, expected_offset);
        assert_eq!(device_patches.offset_within_chunk, 0);
        assert_eq!(device_patches.indices_base, Some(0));

        Ok(())
    }

    #[rstest::rstest]
    #[case::u8(PrimitiveArray::from_iter([0u8, 1, 2, 3, 4, 5]).into_array())]
    #[case::u16(PrimitiveArray::from_iter([0u16, 1, 2, 3, 4, 5]).into_array())]
    #[case::u64(PrimitiveArray::from_iter([0u64, 1, 2, 3, 4, 5]).into_array())]
    #[crate::test]
    async fn test_slice_device_patches_chunk_offset_widths(
        #[case] chunk_offsets: vortex_array::ArrayRef,
    ) -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let indices = PrimitiveArray::from_iter([100u32, 1100, 2100, 3100, 4100]);
        let values = PrimitiveArray::from_iter([10u32, 11, 12, 13, 14]);
        let patches = Patches::new(
            5120,
            0,
            indices.into_array(),
            values.into_array(),
            Some(chunk_offsets),
        )?;

        let mut device_patches = load_device_patches(&patches, &mut cuda_ctx).await?;
        slice_device_patches(&patches, 1024..3000, &mut device_patches);

        assert_eq!(
            device_patches.chunk_offsets.len(),
            3 * device_patches.chunk_offset_ptype.byte_width()
        );
        assert_eq!(device_patches.n_chunks, 3);
        assert_eq!(device_patches.offset, 1024);
        assert_eq!(device_patches.offset_within_chunk, 0);
        assert_eq!(device_patches.indices_base, Some(0));

        Ok(())
    }
}
