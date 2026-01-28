// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! GPU filter implementation using CUB DeviceSelect::Flagged.

use std::ffi::c_void;
use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::DeviceRepr;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::arrays::FilterVTable;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_cub::filter::CubFilterable;
use vortex_cub::filter::cudaStream_t;
use vortex_cuda_macros::cuda_tests;
use vortex_dtype::NativePType;
use vortex_dtype::match_each_native_simd_ptype;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_mask::MaskValues;

use crate::CudaDeviceBuffer;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;
use crate::stream::await_stream_callback;

/// CUDA executor for FilterArray using CUB DeviceSelect::Flagged.
#[derive(Debug)]
pub struct FilterExecutor;

#[async_trait]
impl CudaExecute for FilterExecutor {
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let filter_array = array
            .as_opt::<FilterVTable>()
            .ok_or_else(|| vortex_err!("Expected FilterArray"))?;

        let mask = filter_array.filter_mask();

        // Early return for trivial cases.
        match mask {
            Mask::AllTrue(_) => {
                return filter_array.child().clone().execute_cuda(ctx).await;
            }
            Mask::AllFalse(_) => {
                return Ok(Canonical::empty(filter_array.dtype()));
            }
            _ => {}
        }

        let mask_values = mask
            .values()
            .ok_or_else(|| vortex_err!("Expected Mask::Values but got different variant"))?;

        let canonical = filter_array.child().clone().execute_cuda(ctx).await?;

        match canonical {
            Canonical::Primitive(ref prim) => {
                match_each_native_simd_ptype!(prim.ptype(), |T| {
                    filter_primitive::<T>(prim, mask_values, mask, ctx).await
                })
            }
            _ => unimplemented!(),
        }
    }
}

async fn filter_primitive<T>(
    array: &PrimitiveArray,
    mask_values: &MaskValues,
    mask: &Mask,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical>
where
    T: NativePType + DeviceRepr + CubFilterable + Send + Sync + 'static,
{
    let ptype = array.ptype();
    let num_items = array.len() as i64;
    let output_len = mask_values.true_count();

    if output_len == 0 {
        return Ok(Canonical::empty(array.dtype()));
    }

    let input_handle = array.buffer_handle();
    let d_input: BufferHandle = if input_handle.is_on_device() {
        input_handle.clone()
    } else {
        ctx.move_to_device::<T>(input_handle.clone())?.await?
    };

    // Upload packed bits to device. They are unpacked to bytes in the filter kernel.
    let bit_buffer = mask_values.bit_buffer();
    let packed = bit_buffer.inner().as_ref();
    let bit_offset = bit_buffer.offset() as u64;
    let d_packed_flags = ctx.copy_to_device(packed.to_vec())?.await?;

    let temp_bytes = T::get_temp_size(num_items)
        .map_err(|e| vortex_err!("CUB filter_temp_size failed: {}", e))?;

    // Allocate device buffers.
    let d_temp: CudaSlice<u8> = ctx.device_alloc(temp_bytes.max(1))?;
    let mut d_output: CudaSlice<T> = ctx.device_alloc(output_len)?;
    let mut d_num_selected: CudaSlice<i64> = ctx.device_alloc(1)?;

    // Get raw pointers for FFI.
    let stream = ctx.stream();
    let stream_ptr = stream.cu_stream() as cudaStream_t;

    // Downcast input buffer to get device pointer.
    let d_input_cuda = d_input
        .as_device()
        .as_any()
        .downcast_ref::<CudaDeviceBuffer<T>>()
        .ok_or_else(|| vortex_err!("Expected CudaDeviceBuffer<T> for input"))?;
    let d_input_ptr = d_input_cuda.as_view().device_ptr(stream).0 as *const T;

    // Downcast to get device pointer.
    let d_packed_cuda = d_packed_flags
        .as_device()
        .as_any()
        .downcast_ref::<CudaDeviceBuffer<u8>>()
        .ok_or_else(|| vortex_err!("Expected CudaDeviceBuffer<u8> for packed flags"))?;
    let d_packed_ptr = d_packed_cuda.as_view().device_ptr(stream).0 as *const u8;

    let d_temp_ptr = d_temp.device_ptr(stream).0 as *mut c_void;
    let d_output_ptr = d_output.device_ptr_mut(stream).0 as *mut T;
    let d_num_selected_ptr = d_num_selected.device_ptr_mut(stream).0 as *mut i64;

    // CUB uses TransformInputIterator internally to read bits on-the-fly.
    unsafe {
        T::filter_bitmask(
            d_temp_ptr,
            temp_bytes,
            d_input_ptr,
            d_packed_ptr,
            bit_offset,
            d_output_ptr,
            d_num_selected_ptr,
            num_items,
            stream_ptr,
        )
        .map_err(|e| vortex_err!("CUB filter_bitmask failed: {}", e))?;
    }

    // Wait for completion
    await_stream_callback(stream).await?;

    let filtered_validity = array.validity()?.filter(mask)?;
    let output_handle = BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(d_output)));

    Ok(Canonical::Primitive(PrimitiveArray::from_buffer_handle(
        output_handle,
        ptype,
        filtered_validity,
    )))
}

#[cuda_tests]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::FilterArray;
    use vortex_array::assert_arrays_eq;
    use vortex_error::VortexExpect;
    use vortex_session::VortexSession;

    use super::*;
    use crate::CanonicalCudaExt;
    use crate::session::CudaSession;

    #[rstest]
    #[case::i32_sparse(
            PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6, 7, 8]),
            Mask::from_iter([true, false, true, false, true, false, true, false])
        )]
    #[case::i32_dense(
            PrimitiveArray::from_iter([10i32, 20, 30, 40, 50]),
            Mask::from_iter([true, true, true, false, true])
        )]
    #[case::i64_large(
            PrimitiveArray::from_iter((0..1000i64).collect::<Vec<_>>()),
            Mask::from_iter((0..1000).map(|i| i % 3 == 0))
        )]
    #[case::f64_values(
            PrimitiveArray::from_iter([1.1f64, 2.2, 3.3, 4.4, 5.5]),
            Mask::from_iter([false, true, false, true, false])
        )]
    #[case::u8_all_true(
            PrimitiveArray::from_iter([1u8, 2, 3, 4, 5]),
            Mask::from_iter([true, true, true, true, true])
        )]
    #[case::u32_all_false(
            PrimitiveArray::from_iter([1u32, 2, 3, 4, 5]),
            Mask::from_iter([false, false, false, false, false])
        )]
    #[tokio::test]
    async fn test_gpu_filter(
        #[case] input: PrimitiveArray,
        #[case] mask: Mask,
    ) -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create CUDA execution context");

        let filter_array = FilterArray::try_new(input.clone().into_array(), mask.clone())?;

        let cpu_result = filter_array.to_canonical()?.into_array();

        let gpu_result = FilterExecutor
            .execute(filter_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU filter failed")
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(cpu_result, gpu_result);

        Ok(())
    }

    #[tokio::test]
    async fn test_gpu_filter_large_array() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create CUDA execution context");

        // Create a large array to test multi-block execution
        let data: Vec<i32> = (0..100_000).collect();
        let input = PrimitiveArray::from_iter(data);

        // Select every 7th element
        let mask = Mask::from_iter((0..100_000).map(|i| i % 7 == 0));

        let filter_array = FilterArray::try_new(input.into_array(), mask)?;

        let cpu_result = filter_array.to_canonical()?.into_array();

        let gpu_result = FilterExecutor
            .execute(filter_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU filter failed")
            .into_host()
            .await?
            .into_array();

        assert_eq!(cpu_result.len(), gpu_result.len());
        assert_arrays_eq!(cpu_result, gpu_result);

        Ok(())
    }
}
