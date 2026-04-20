// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! GPU patches loading for fused exception patching during bit-unpacking.

use std::mem::align_of;
use std::mem::size_of;

use num_traits::ToPrimitive;
use vortex::array::buffer::BufferHandle;
use vortex::buffer::Buffer;
use vortex::buffer::BufferMut;
use vortex::buffer::ByteBufferMut;
use vortex::dtype::NativePType;
use vortex::dtype::PType;

use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
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

/// Pack patches into the per-chunk buffer format for the fused dispatch kernel.
///
/// The packed buffer is self-describing via `PackedPatchesHeader` and can be
/// passed directly as a device pointer to the kernel's `patches_ptr` field.
#[allow(deprecated)]
pub(crate) fn pack_patches_for_fused_dispatch(
    patches: &Patches,
    element_offset: u16,
    array_len: usize,
    ptype: PType,
) -> VortexResult<BufferHandle> {
    // Map to the unsigned integer type of the same width — the kernel
    // reinterprets floats as unsigned ints, so f32→u32 and f64→u64.
    let unsigned_ptype = match ptype {
        PType::F32 => PType::U32,
        PType::F64 => PType::U64,
        other => other.to_unsigned(),
    };
    match_each_unsigned_integer_ptype!(unsigned_ptype, |V| {
        pack_patches_typed::<V>(patches, element_offset, array_len)
    })
}

#[allow(deprecated)]
fn pack_patches_typed<V: NativePType>(
    patches: &Patches,
    element_offset: u16,
    array_len: usize,
) -> VortexResult<BufferHandle> {
    let n_chunks = (array_len + element_offset as usize).div_ceil(1024);
    let num_patches = patches.num_patches();

    // Canonicalize indices to u32 and values to V.
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let indices_prim = patches
        .indices()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    let indices_u32: Vec<u32> = match_each_unsigned_integer_ptype!(indices_prim.ptype(), |I| {
        indices_prim
            .as_slice::<I>()
            .iter()
            .map(|&idx| idx.to_u32().unwrap_or(0))
            .collect()
    });

    let values_prim = patches
        .values()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    // Reinterpret the raw bytes as V. For float patches (f32→u32, f64→u64)
    // this preserves the bit pattern rather than numerically converting.
    assert_eq!(
        values_prim.ptype().byte_width(),
        size_of::<V>(),
        "patch value width {} != target width {}",
        values_prim.ptype().byte_width(),
        size_of::<V>(),
    );
    let raw_bytes = values_prim.buffer_handle().to_host_sync();
    let values_slice: &[V] = unsafe {
        std::slice::from_raw_parts(raw_bytes.as_slice().as_ptr().cast::<V>(), num_patches)
    };

    // For each patch, compute (chunk, within_chunk) using element_offset.
    struct PatchEntry<V> {
        chunk: usize,
        within_chunk: u16,
        value: V,
    }

    let mut entries: Vec<PatchEntry<V>> = Vec::with_capacity(num_patches);
    for i in 0..num_patches {
        let abs_pos = indices_u32[i] as usize + element_offset as usize;
        let chunk = abs_pos / 1024;
        let within_chunk = (abs_pos % 1024) as u16;
        entries.push(PatchEntry {
            chunk,
            within_chunk,
            value: values_slice[i],
        });
    }

    // Sort by chunk, then by within-chunk position for determinism.
    entries.sort_by(|a, b| {
        a.chunk
            .cmp(&b.chunk)
            .then(a.within_chunk.cmp(&b.within_chunk))
    });

    // Build CSR chunk_offsets[n_chunks + 1] (with sentinel).
    let mut chunk_offsets: Vec<u32> = vec![0u32; n_chunks + 1];
    for entry in &entries {
        if entry.chunk < n_chunks {
            chunk_offsets[entry.chunk + 1] += 1;
        }
    }
    // Prefix sum.
    for i in 1..=n_chunks {
        chunk_offsets[i] += chunk_offsets[i - 1];
    }

    // Pack indices (u16) and values (V) in sorted order.
    let mut packed_indices: Vec<u16> = Vec::with_capacity(num_patches);
    let mut packed_values: Vec<V> = Vec::with_capacity(num_patches);
    for entry in &entries {
        packed_indices.push(entry.within_chunk);
        packed_values.push(entry.value);
    }

    // Compute layout sizes.
    // Header: 3 × u32 = 12 bytes.
    let header_size = 3 * size_of::<u32>();
    // chunk_offsets: (n_chunks + 1) × u32.
    let chunk_offsets_size = (n_chunks + 1) * size_of::<u32>();
    // indices: num_patches × u16.
    let indices_size = num_patches * size_of::<u16>();

    let indices_byte_offset = header_size + chunk_offsets_size;
    let values_unaligned_offset = indices_byte_offset + indices_size;

    // Align values to align_of::<V>().
    let val_align = align_of::<V>();
    let values_byte_offset = (values_unaligned_offset + val_align - 1) & !(val_align - 1);
    let padding = values_byte_offset - values_unaligned_offset;

    let values_size = num_patches * size_of::<V>();
    let total_size = values_byte_offset + values_size;

    // Write the buffer.
    let mut buffer =
        ByteBufferMut::with_capacity_aligned(total_size, vortex::buffer::Alignment::of::<u32>());

    // Header: n_chunks, indices_byte_offset, values_byte_offset (all as u32, little-endian).
    buffer.extend_from_slice(&(n_chunks as u32).to_le_bytes());
    buffer.extend_from_slice(&(indices_byte_offset as u32).to_le_bytes());
    buffer.extend_from_slice(&(values_byte_offset as u32).to_le_bytes());

    // chunk_offsets.
    for &co in &chunk_offsets {
        buffer.extend_from_slice(&co.to_le_bytes());
    }

    // indices (u16, little-endian).
    for &idx in &packed_indices {
        buffer.extend_from_slice(&idx.to_le_bytes());
    }

    // Padding between indices and values.
    for _ in 0..padding {
        buffer.extend_from_slice(&[0u8]);
    }

    // Values (V, little-endian).
    for &val in &packed_values {
        buffer.extend_from_slice(val.to_le_bytes().as_ref());
    }

    debug_assert_eq!(buffer.len(), total_size);

    Ok(BufferHandle::new_host(buffer.freeze()))
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
