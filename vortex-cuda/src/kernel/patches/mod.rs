// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub mod types;

#[rustfmt::skip]
#[expect(warnings, clippy::all, clippy::pedantic, clippy::nursery)]
#[allow(clippy::absolute_paths)]
pub mod gpu {
    include!(concat!(env!("OUT_DIR"), "/patches.rs"));
}

use cudarc::driver::DeviceRepr;
use cudarc::driver::PushKernelArg;
use tracing::instrument;
use vortex::array::arrays::primitive::PrimitiveDataParts;
use vortex::array::patches::Patches;
use vortex::array::validity::Validity;
use vortex::dtype::NativePType;
use vortex::dtype::PType;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;

use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::CudaExecutionCtx;
use crate::executor::CudaArrayExt;
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
fn ptype_to_chunk_offset_type(ptype: PType) -> VortexResult<ChunkOffsetType> {
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

/// Apply a set of patches in-place onto a [`CudaDeviceBuffer`] holding `ValuesT`.
///
/// Naive scatter kernel. Kept as a reusable fallback for encoders that cannot
/// use the chunk-based fused patching path (e.g., where `chunk_offsets` are
/// unavailable); no production caller uses it today.
#[allow(dead_code)]
#[instrument(skip_all)]
pub(crate) async fn execute_patches<
    ValuesT: NativePType + DeviceRepr,
    IndicesT: NativePType + DeviceRepr,
>(
    patches: Patches,
    target: CudaDeviceBuffer,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<CudaDeviceBuffer> {
    let indices = patches.indices().clone();
    let values = patches.values().clone();
    drop(patches);

    let indices = indices.execute_cuda(ctx).await?.into_primitive();
    let values = values.execute_cuda(ctx).await?.into_primitive();

    let supported = matches!(
        values.validity()?,
        Validity::NonNullable | Validity::AllValid
    );
    vortex_ensure!(
        supported,
        "Applying patches with null values not currently supported on the GPU"
    );

    vortex_ensure!(
        indices.ptype() == IndicesT::PTYPE,
        "expected PType {} for patch indices, was {}",
        IndicesT::PTYPE,
        indices.ptype()
    );

    vortex_ensure!(
        values.ptype() == ValuesT::PTYPE,
        "expected PType {} for patch values, was {}",
        ValuesT::PTYPE,
        values.ptype()
    );

    let patches_len = indices.len();
    let patches_len_u64 = patches_len as u64;

    let PrimitiveDataParts {
        buffer: indices_buffer,
        ..
    } = indices.into_data_parts();

    let PrimitiveDataParts {
        buffer: values_buffer,
        ..
    } = values.into_data_parts();

    let d_patch_indices = ctx.ensure_on_device(indices_buffer).await?;
    let d_patch_values = ctx.ensure_on_device(values_buffer).await?;

    let d_target_view = target.as_view::<ValuesT>();
    let d_patch_indices_view = d_patch_indices.cuda_view::<IndicesT>()?;
    let d_patch_values_view = d_patch_values.cuda_view::<ValuesT>()?;

    let kernel_func = ctx.load_function("patches", &[ValuesT::PTYPE, IndicesT::PTYPE])?;

    ctx.launch_kernel(&kernel_func, patches_len, |args| {
        args.arg(&d_target_view)
            .arg(&d_patch_indices_view)
            .arg(&d_patch_values_view)
            .arg(&patches_len_u64);
    })?;

    Ok(target)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use cudarc::driver::DeviceRepr;
    use vortex::array::ExecutionCtx;
    use vortex::array::IntoArray;
    use vortex::array::LEGACY_SESSION;
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::primitive::PrimitiveDataParts;
    use vortex::array::assert_arrays_eq;
    use vortex::array::buffer::BufferHandle;
    use vortex::array::builtins::ArrayBuiltins;
    use vortex::array::patches::Patches;
    use vortex::array::validity::Validity;
    use vortex::buffer::buffer;
    use vortex::dtype::DType;
    use vortex::dtype::NativePType;
    use vortex::dtype::Nullability;
    use vortex::session::VortexSession;

    use crate::CanonicalCudaExt;
    use crate::CudaDeviceBuffer;
    use crate::CudaSession;
    use crate::kernel::patches::execute_patches;

    #[rstest::rstest]
    #[case::u8(0_u8)]
    #[case::u16(0_u16)]
    #[case::u32(0_u32)]
    #[case::u64(0_u64)]
    #[case::i8(0_i8)]
    #[case::i16(0_i16)]
    #[case::i32(0_i32)]
    #[case::i64(0_i64)]
    #[case::f32(0_f32)]
    #[case::f64(0_f64)]
    #[crate::test]
    async fn test_patches<Values: NativePType + DeviceRepr>(#[case] _v: Values) {
        tokio::join!(
            full_test_case::<Values, u8>(),
            full_test_case::<Values, u16>(),
            full_test_case::<Values, u32>(),
            full_test_case::<Values, u64>(),
        );
    }

    async fn full_test_case<Values: NativePType + DeviceRepr, Indices: NativePType + DeviceRepr>() {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty()).unwrap();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();

        let values = PrimitiveArray::from_iter(0..128);
        let values = force_cast::<Values>(values, &mut ctx);

        let patch_idx = PrimitiveArray::new(buffer![0, 8, 16, 32], Validity::NonNullable);
        let patch_idx = force_cast::<Indices>(patch_idx, &mut ctx);

        let patch_val = PrimitiveArray::new(buffer![99, 99, 99, 99], Validity::NonNullable);
        let patch_val = force_cast::<Values>(patch_val, &mut ctx);

        // Copy all to GPU
        let patches =
            Patches::new(128, 0, patch_idx.into_array(), patch_val.into_array(), None).unwrap();

        let cpu_result = values.clone().patch(&patches, &mut ctx).unwrap();

        let PrimitiveDataParts {
            buffer: cuda_buffer,
            ..
        } = values.into_data_parts();

        let handle = cuda_ctx.ensure_on_device(cuda_buffer).await.unwrap();
        let device_buf = handle
            .as_device()
            .as_any()
            .downcast_ref::<CudaDeviceBuffer>()
            .unwrap()
            .clone();

        let patched_buf = execute_patches::<Values, Indices>(patches, device_buf, &mut cuda_ctx)
            .await
            .unwrap();

        let gpu_result = PrimitiveArray::from_buffer_handle(
            BufferHandle::new_device(Arc::new(patched_buf)),
            Values::PTYPE,
            Validity::NonNullable,
        )
        .into_array()
        .execute::<vortex::array::Canonical>(&mut ctx)
        .unwrap()
        .into_host()
        .await
        .unwrap()
        .into_primitive();

        assert_arrays_eq!(cpu_result, gpu_result);
    }

    fn force_cast<T: NativePType>(array: PrimitiveArray, ctx: &mut ExecutionCtx) -> PrimitiveArray {
        array
            .into_array()
            .cast(DType::Primitive(T::PTYPE, Nullability::NonNullable))
            .unwrap()
            .execute::<PrimitiveArray>(ctx)
            .unwrap()
    }
}
