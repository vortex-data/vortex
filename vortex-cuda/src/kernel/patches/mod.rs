// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;
use std::sync::Arc;

use cudarc::driver::DeviceRepr;
use vortex_array::arrays::PrimitiveArrayParts;
use vortex_array::patches::Patches;
use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityHelper;
use vortex_cuda_macros::cuda_tests;
use vortex_dtype::NativePType;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::CudaDeviceBuffer;
use crate::CudaExecutionCtx;
use crate::arrow::ensure_device_resident;
use crate::executor::CudaArrayExt;
use crate::kernel::scalar_launch_config;
use crate::launcher::Function;
use crate::launcher::Kernel;
use crate::launcher::Launcher;

struct PatchesKernel<Values, Indices> {
    function: Function,
    _marker: PhantomData<(Values, Indices)>,
}

impl<Values: NativePType, Indices: NativePType> Kernel for PatchesKernel<Values, Indices> {
    type Args = (CudaDeviceBuffer, CudaDeviceBuffer, CudaDeviceBuffer, u64);

    fn new(function: Function) -> Self {
        Self {
            function,
            _marker: PhantomData,
        }
    }

    unsafe fn launch(self, args: Self::Args, launcher: &Arc<dyn Launcher>) -> VortexResult<()> {
        let (target, indices, values, mut array_len) = args;
        let cfg = scalar_launch_config(array_len as usize);

        unsafe {
            launcher.launch(
                self.function,
                cfg,
                vec![
                    target.device_ptr(),
                    indices.device_ptr(),
                    values.device_ptr(),
                    std::ptr::addr_of_mut!(array_len).cast(),
                ],
            )
        }
    }
}

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

    let supported = matches!(
        values.validity(),
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

    let PrimitiveArrayParts {
        buffer: indices_buffer,
        ..
    } = indices.into_parts();

    let PrimitiveArrayParts {
        buffer: values_buffer,
        ..
    } = values.into_parts();

    let d_patch_indices = ensure_device_resident(indices_buffer, ctx).await?;
    let d_patch_values = ensure_device_resident(values_buffer, ctx).await?;

    let kernel = ctx
        .load_module("patches")?
        .load::<PatchesKernel<ValuesT, IndicesT>>(format!(
            "patches_{}_{}",
            ValuesT::PTYPE,
            IndicesT::PTYPE
        ))?;

    ctx.launch(
        kernel,
        (
            target.clone(),
            d_patch_indices,
            d_patch_values,
            patches_len_u64,
        ),
    )?;

    Ok(target)
}

#[cuda_tests]
mod tests {
    use std::sync::Arc;

    use cudarc::driver::DeviceRepr;
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::PrimitiveArrayParts;
    use vortex_array::assert_arrays_eq;
    use vortex_array::buffer::BufferHandle;
    use vortex_array::compute::cast;
    use vortex_array::patches::Patches;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::NativePType;
    use vortex_dtype::Nullability;
    use vortex_session::VortexSession;

    use crate::CanonicalCudaExt;
    use crate::CudaDeviceBuffer;
    use crate::CudaSession;
    use crate::kernel::patches::execute_patches;

    #[tokio::test]
    async fn test_patches() {
        test_case::<u8>().await;
        test_case::<u16>().await;
        test_case::<u32>().await;
        test_case::<u64>().await;

        test_case::<i8>().await;
        test_case::<i16>().await;
        test_case::<i32>().await;
        test_case::<i64>().await;

        test_case::<f32>().await;
        test_case::<f64>().await;
    }

    async fn test_case<Values: NativePType + DeviceRepr>() {
        full_test_case::<Values, u8>().await;
        full_test_case::<Values, u16>().await;
        full_test_case::<Values, u32>().await;
        full_test_case::<Values, u64>().await;
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

        let cpu_result = values.clone().patch(&patches).unwrap();

        let PrimitiveArrayParts {
            buffer: cuda_buffer,
            ..
        } = values.into_parts();

        let handle = ctx.move_to_device(cuda_buffer).unwrap().await.unwrap();

        let patched_buf = execute_patches::<Values, Indices>(patches, handle, &mut ctx)
            .await
            .unwrap();

        let gpu_result = PrimitiveArray::from_buffer_handle(
            patched_buf.into(),
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
        cast(
            array.as_ref(),
            &DType::Primitive(T::PTYPE, Nullability::NonNullable),
        )
        .unwrap()
        .to_primitive()
    }
}
