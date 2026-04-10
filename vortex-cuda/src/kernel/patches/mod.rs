// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub mod types;

#[rustfmt::skip]
#[expect(warnings, clippy::all, clippy::pedantic, clippy::nursery)]
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
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;

use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::CudaExecutionCtx;
use crate::executor::CudaArrayExt;

/// Apply a set of patches in-place onto a [`CudaDeviceBuffer`] holding `ValuesT`.
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
    use vortex::array::IntoArray;
    use vortex::array::ToCanonical;
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
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty()).unwrap();

        let values = PrimitiveArray::from_iter(0..128);
        let values = force_cast::<Values>(values);

        let patch_idx = PrimitiveArray::new(buffer![0, 8, 16, 32], Validity::NonNullable);
        let patch_idx = force_cast::<Indices>(patch_idx);

        let patch_val = PrimitiveArray::new(buffer![99, 99, 99, 99], Validity::NonNullable);
        let patch_val = force_cast::<Values>(patch_val);

        // Copy all to GPU
        let patches =
            Patches::new(128, 0, patch_idx.into_array(), patch_val.into_array(), None).unwrap();

        let cpu_result = values
            .clone()
            .patch(
                &patches,
                &mut vortex::array::LEGACY_SESSION.create_execution_ctx(),
            )
            .unwrap();

        let PrimitiveDataParts {
            buffer: cuda_buffer,
            ..
        } = values.into_data_parts();

        let handle = ctx.ensure_on_device(cuda_buffer).await.unwrap();
        let device_buf = handle
            .as_device()
            .as_any()
            .downcast_ref::<CudaDeviceBuffer>()
            .unwrap()
            .clone();

        let patched_buf = execute_patches::<Values, Indices>(patches, device_buf, &mut ctx)
            .await
            .unwrap();

        let gpu_result = PrimitiveArray::from_buffer_handle(
            BufferHandle::new_device(Arc::new(patched_buf)),
            Values::PTYPE,
            Validity::NonNullable,
        )
        .to_canonical()
        .unwrap()
        .into_host()
        .await
        .unwrap()
        .into_primitive();

        assert_arrays_eq!(cpu_result, gpu_result);
    }

    fn force_cast<T: NativePType>(array: PrimitiveArray) -> PrimitiveArray {
        array
            .into_array()
            .cast(DType::Primitive(T::PTYPE, Nullability::NonNullable))
            .unwrap()
            .to_primitive()
    }
}
