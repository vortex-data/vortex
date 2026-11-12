// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub mod types;

#[rustfmt::skip]
#[expect(warnings, clippy::all, clippy::pedantic, clippy::nursery)]
#[allow(clippy::absolute_paths)]
pub mod gpu {
    include!(concat!(env!("OUT_DIR"), "/patches.rs"));
}

use vortex::dtype::PType;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex_cuda_driver::DeviceRepr;

use crate::CudaBufferExt;
use crate::kernel::patches::gpu::ChunkOffsetType;
use crate::kernel::patches::gpu::ChunkOffsetType_CO_U8;
use crate::kernel::patches::gpu::ChunkOffsetType_CO_U16;
use crate::kernel::patches::gpu::ChunkOffsetType_CO_U32;
use crate::kernel::patches::gpu::ChunkOffsetType_CO_U64;
use crate::kernel::patches::gpu::GPUPatches;
use crate::kernel::patches::gpu::PATCH_DERIVE_INDICES_BASE;
use crate::kernel::patches::types::DevicePatches;

// Safe because `GPUPatches` contains only raw pointers, POD integers, and an enum.
unsafe impl DeviceRepr for GPUPatches {}

impl GPUPatches {
    /// Sentinel value passed to kernels when no patches are present. A NULL
    /// `chunk_offsets` pointer is the signal `PatchesCursor` checks for.
    pub(crate) const NULL_PATCHES: Self = Self {
        chunk_offsets: std::ptr::null_mut(),
        chunk_offset_type: ChunkOffsetType_CO_U32,
        indices: std::ptr::null_mut(),
        values: std::ptr::null_mut(),
        offset: 0,
        offset_within_chunk: 0,
        num_patches: 0,
        n_chunks: 0,
        indices_base: 0,
    };
}

/// Convert a [`PType`] to the corresponding [`ChunkOffsetType`] for GPU patches.
pub(crate) fn ptype_to_chunk_offset_type(ptype: PType) -> VortexResult<ChunkOffsetType> {
    match ptype {
        PType::U8 => Ok(ChunkOffsetType_CO_U8),
        PType::U16 => Ok(ChunkOffsetType_CO_U16),
        PType::U32 => Ok(ChunkOffsetType_CO_U32),
        PType::U64 => Ok(ChunkOffsetType_CO_U64),
        _ => vortex_bail!("Invalid PType for chunk_offsets: {:?}", ptype),
    }
}

/// Build a [`GPUPatches`] kernel argument from optional device-resident patches.
///
/// When `device_patches` is `None`, returns a sentinel value whose NULL
/// `chunk_offsets` signals "no patches" to the kernel.
pub(crate) fn build_gpu_patches(
    device_patches: Option<&DevicePatches>,
) -> VortexResult<GPUPatches> {
    #[expect(clippy::cast_possible_truncation)]
    match device_patches {
        Some(p) => Ok(GPUPatches {
            chunk_offsets: p.chunk_offsets.cuda_device_ptr()? as _,
            chunk_offset_type: ptype_to_chunk_offset_type(p.chunk_offset_ptype)?,
            indices: p.indices.cuda_device_ptr()? as _,
            values: p.values.cuda_device_ptr()? as _,
            offset: p.offset as u32,
            offset_within_chunk: p.offset_within_chunk as u32,
            num_patches: p.num_patches as u32,
            n_chunks: p.n_chunks as u32,
            indices_base: p
                .indices_base
                .map_or(PATCH_DERIVE_INDICES_BASE, |base| base as u32),
        }),
        None => Ok(GPUPatches::NULL_PATCHES),
    }
}
