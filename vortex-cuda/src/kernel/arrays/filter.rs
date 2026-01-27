// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A CUDA kernel executor for `FilterArray`.
//!
//! The filter array is constructed with a mask and a child. We first execute the child on the GPU
//! into its canonical form, and then execute filter over it.

use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::DeviceRepr;
use cudarc::driver::LaunchArgs;
use cudarc::driver::PushKernelArg;
use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING;
use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::arrays::FilterArray;
use vortex_array::arrays::FilterArrayParts;
use vortex_array::arrays::FilterVTable;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_cuda_macros::cuda_tests;
use vortex_dtype::DType;
use vortex_dtype::NativePType;
use vortex_dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::CudaExecutionCtx;
use crate::CudaKernelEvents;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;

#[derive(Debug)]
pub struct FilterExecutor;

#[async_trait]
impl CudaExecute for FilterExecutor {
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let Ok(filter_array) = array.try_into::<FilterVTable>() else {
            vortex_bail!("FilterExecutor expected FilterArray input");
        };

        match filter_array.dtype() {
            DType::Primitive(ptype, _) => {
                match ptype {
                    PType::U8 => execute_filter_primitive::<u8>(filter_array, ctx).await,
                    PType::U16 => execute_filter_primitive::<u16>(filter_array, ctx).await,
                    PType::U32 => execute_filter_primitive::<u32>(filter_array, ctx).await,
                    PType::U64 => execute_filter_primitive::<u64>(filter_array, ctx).await,
                    PType::I8 => execute_filter_primitive::<i8>(filter_array, ctx).await,
                    PType::I16 => execute_filter_primitive::<i16>(filter_array, ctx).await,
                    PType::I32 => execute_filter_primitive::<i32>(filter_array, ctx).await,
                    PType::I64 => execute_filter_primitive::<i64>(filter_array, ctx).await,
                    PType::F16 => {
                        // TODO(aduffy): filter as u16 instead
                        vortex_bail!("f16 not supported for filter_primitive kernel");
                    }
                    PType::F32 => execute_filter_primitive::<f32>(filter_array, ctx).await,
                    PType::F64 => execute_filter_primitive::<f64>(filter_array, ctx).await,
                }
            }
            // DType::Decimal(_, _) => {}
            dtype => vortex_bail!("unsupported DType for GPU filter kernel {dtype}"),
        }
    }
}

async fn execute_filter_primitive<T: NativePType + DeviceRepr>(
    filter_array: FilterArray,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical> {
    let FilterArrayParts { child, mask } = filter_array.into_parts();

    match mask {
        Mask::AllTrue(_) => child.execute_cuda(ctx).await,
        Mask::AllFalse(_) => Ok(Canonical::empty(child.dtype())),
        m @ Mask::Values(_) => {
            let output_validity = child.validity()?.filter(&m)?;

            let bits = m.into_bit_buffer().shrink_offset();
            // TODO(aduffy): eliminate the true counting here?
            let kept = bits.true_count();

            // Allocate an output buffer of T's to hold the result.
            let output = ctx.device_alloc::<T>(kept)?;

            // Copy the mask and the child onto device buffer
            let (mask_offset, mask_len, buffer) = bits.into_inner();
            let mask_offset = u32::try_from(mask_offset)?;
            let mask_len = u32::try_from(mask_len)?;
            let mask_buf = ctx.copy_to_device(buffer)?.await?;

            let child = child.execute_cuda(ctx).await?.into_primitive();
            let input_buf = child.buffer_handle().clone();
            let input_device = if input_buf.is_on_device() {
                input_buf
            } else {
                ctx.move_to_device::<T>(input_buf)?.await?
            };

            let input_view = input_device.cuda_view::<T>()?;
            let mask_view = mask_buf.cuda_view::<u8>()?;
            let output_view = output.as_view();

            // What is the array len?
            let kernel = ctx.load_function_ptype("filter_primitive", &[T::PTYPE])?;
            let mut launch_builder = ctx.launch_builder(&kernel);
            launch_builder.arg(&input_view);
            launch_builder.arg(&mask_view);
            launch_builder.arg(&output_view);
            launch_builder.arg(&mask_offset);
            launch_builder.arg(&mask_len);

            let _cuda_events = launch_filter_kernel(&mut launch_builder, CU_EVENT_DISABLE_TIMING)?;

            // TODO(aduffy): actually filter validity too.
            Ok(Canonical::Primitive(PrimitiveArray::from_buffer_handle(
                BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(output))),
                T::PTYPE,
                output_validity,
            )))
        }
    }
}

fn launch_filter_kernel(
    launch_builder: &mut LaunchArgs,
    event_flags: CUevent_flags,
) -> VortexResult<CudaKernelEvents> {
    // We cap this at 1 thread block.
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (512, 1, 1),
        shared_mem_bytes: 0,
    };

    launch_builder.record_kernel_launch(event_flags);

    unsafe {
        launch_builder
            .launch(config)
            .map_err(|e| vortex_err!("Failed to launch kernel: {}", e))
            .and_then(|events| {
                events
                    .ok_or_else(|| vortex_err!("CUDA events not recorded"))
                    .map(|(before_launch, after_launch)| CudaKernelEvents {
                        before_launch,
                        after_launch,
                    })
            })
    }
}

#[cuda_tests]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity::NonNullable;
    use vortex_buffer::BitBufferMut;
    use vortex_buffer::Buffer;
    use vortex_error::VortexExpect;
    use vortex_session::VortexSession;

    use super::*;
    use crate::session::CudaSession;

    /// Copy a CUDA primitive array result to host memory.
    fn cuda_primitive_to_host(prim: PrimitiveArray) -> VortexResult<PrimitiveArray> {
        Ok(PrimitiveArray::from_byte_buffer(
            prim.buffer_handle().try_to_host_sync()?,
            prim.ptype(),
            prim.validity()?,
        ))
    }

    #[tokio::test]
    async fn test_filter_u32() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let values: Buffer<u32> = (0..512).collect();
        let values = PrimitiveArray::new(values, NonNullable);

        // Create a filter mask that takes the first item from every block of 8.
        let mut mask = BitBufferMut::with_capacity(512);
        for idx in 0..512 {
            if idx % 16 == 0 {
                mask.append_true();
            } else {
                mask.append_false();
            }
        }
        let mask = mask.freeze();

        // Construct the filter operation and execute it on GPU.
        let filter_array = FilterArray::new(values.into_array(), Mask::from_buffer(mask));

        // Get baseline from CPU canonicalization
        let baseline = filter_array.to_canonical()?;

        // Execute on CUDA
        let cuda_result = FilterExecutor
            .execute(filter_array.to_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_primitive();
        cuda_ctx.synchronize_stream()?;
        let cuda_result = cuda_primitive_to_host(cuda_result)?;

        // Compare CUDA result with baseline
        assert_arrays_eq!(cuda_result.into_array(), baseline.into_array());
        Ok(())
    }
}
