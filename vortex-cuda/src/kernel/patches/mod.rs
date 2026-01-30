// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use cudarc::driver::DeviceRepr;
use cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING;
use vortex_array::arrays::PrimitiveArrayParts;
use vortex_array::patches::Patches;
use vortex_cuda_macros::cuda_tests;
use vortex_dtype::NativePType;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::CudaDeviceBuffer;
use crate::CudaExecutionCtx;
use crate::executor::CudaArrayExt;
use crate::launch_cuda_kernel;

/// Apply a set of patches in-place onto a [`CudaDeviceBuffer`] holding `ValuesT`.
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

    let PrimitiveArrayParts {
        buffer: indices_buffer,
        ..
    } = indices.into_parts();

    let PrimitiveArrayParts {
        buffer: values_buffer,
        validity: values_validity,
        ..
    } = values.into_parts();

    let d_patch_indices = if indices_buffer.is_on_device() {
        indices_buffer
    } else {
        ctx.move_to_device::<IndicesT>(indices_buffer)?.await?
    };

    let d_patch_values = if values_buffer.is_on_device() {
        values_buffer
    } else {
        ctx.move_to_device::<ValuesT>(values_buffer)?.await?
    };

    let d_patch_indices_buf = d_patch_indices
        .as_device()
        .as_any()
        .downcast_ref::<CudaDeviceBuffer>()
        .ok_or_else(|| vortex_err!("d_patch_indices must be CudaDeviceBuffer"))?;

    let d_patch_values_buf = d_patch_values
        .as_device()
        .as_any()
        .downcast_ref::<CudaDeviceBuffer>()
        .ok_or_else(|| vortex_err!("d_patch_values must be CudaDeviceBuffer"))?;

    let d_target_view = target.as_view::<ValuesT>();
    let d_patch_indices_view = d_patch_indices_buf.as_view::<IndicesT>();
    let d_patch_values_view = d_patch_values_buf.as_view::<ValuesT>();

    // kernel arg order for patches is values, patchIndices, patchValues, patchesLen
    let _events = launch_cuda_kernel!(
        execution_ctx: ctx,
        module: "patches",
        ptypes: &[ValuesT::PTYPE, IndicesT::PTYPE],
        launch_args: [
            d_target_view,
            d_patch_indices_view,
            d_patch_values_view,
            patches_len_u64,
        ],
        event_recording: CU_EVENT_DISABLE_TIMING,
        array_len: patches_len
    );

    Ok(target)
}

#[cuda_tests]
mod tests {
    #[test]
    fn test_impl() {}
}
