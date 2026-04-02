// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! GPU filter implementation using CUB DeviceSelect::Flagged.

mod decimal;
mod primitive;
mod varbinview;

use std::ffi::c_void;
use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::DeviceRepr;
use tracing::instrument;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::arrays::Filter;
use vortex::array::arrays::filter::FilterArrayParts;
use vortex::array::buffer::BufferHandle;
use vortex::array::match_each_decimal_value_type;
use vortex::array::match_each_native_simd_ptype;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::mask::Mask;
use vortex_cub::filter::CubFilterable;
use vortex_cub::filter::cudaStream_t;

use crate::CudaDeviceBuffer;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;
use crate::kernel::filter::decimal::filter_decimal;
use crate::kernel::filter::primitive::filter_primitive;
use crate::kernel::filter::varbinview::filter_varbinview;

/// CUDA executor for FilterArray using CUB DeviceSelect::Flagged.
#[derive(Debug)]
pub struct FilterExecutor;

#[async_trait]
impl CudaExecute for FilterExecutor {
    #[instrument(level = "trace", skip_all, fields(executor = ?self))]
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let filter_array = array
            .try_into::<Filter>()
            .map_err(|_| vortex_err!("Expected FilterArray"))?;

        let FilterArrayParts { child, mask } = filter_array.into_data().into_parts();

        // Early return for trivial cases.
        match mask {
            Mask::AllTrue(_) => {
                // No data filtered => execute child without any post-processing
                child.execute_cuda(ctx).await
            }
            Mask::AllFalse(_) => {
                // All data filtered => empty canonical
                Ok(Canonical::empty(child.dtype()))
            }
            m @ Mask::Values(_) => {
                let canonical = child.execute_cuda(ctx).await?;
                match canonical {
                    Canonical::Primitive(prim) => {
                        match_each_native_simd_ptype!(prim.ptype(), |T| {
                            filter_primitive::<T>(prim, m, ctx).await
                        })
                    }
                    Canonical::Decimal(decimal) => {
                        match_each_decimal_value_type!(decimal.values_type(), |D| {
                            filter_decimal::<D>(decimal, m, ctx).await
                        })
                    }
                    Canonical::VarBinView(varbinview) => {
                        filter_varbinview(varbinview, m, ctx).await
                    }
                    _ => unimplemented!(),
                }
            }
        }
    }
}

async fn filter_sized<T: DeviceRepr + CubFilterable + Debug + Send + Sync + 'static>(
    input: BufferHandle,
    mask: Mask,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<BufferHandle> {
    let d_input = ctx.ensure_on_device(input).await?;

    // Construct the inputs for the cub::DeviceSelect::Flagged call.
    let output_len = mask.true_count();
    let (offset, len, flags) = mask.into_bit_buffer().into_inner();

    let d_flags = ctx.copy_to_device(flags.to_vec())?.await?;

    let offset = offset as u64;
    let len = len as i64;

    let temp_bytes =
        T::get_temp_size(len).map_err(|e| vortex_err!("CUB filter_temp_size failed: {}", e))?;

    // Allocate device buffers for input, output, mask, and temp space
    let d_temp = ctx.device_alloc::<u8>(temp_bytes.max(1))?;
    let mut d_output = ctx.device_alloc::<T>(output_len)?;
    let mut d_num_selected = ctx.device_alloc::<i64>(1)?;
    // Get raw pointers for FFI.
    let stream = ctx.stream();
    let stream_ptr = stream.cu_stream() as cudaStream_t;

    // Downcast input buffer to get device pointer.
    let d_input_cuda = d_input
        .as_device()
        .as_any()
        .downcast_ref::<CudaDeviceBuffer>()
        .ok_or_else(|| vortex_err!("Expected CudaDeviceBuffer for input, was {d_input:?}",))?;
    let d_input_view = d_input_cuda.as_view::<T>();
    let (d_input_ptr, record_d_input) = d_input_view.device_ptr(stream);

    // Downcast to get device pointer.
    let d_packed_cuda = d_flags
        .as_device()
        .as_any()
        .downcast_ref::<CudaDeviceBuffer>()
        .ok_or_else(|| vortex_err!("Expected CudaDeviceBuffer for packed flags"))?;
    let d_packed_view = d_packed_cuda.as_view::<u8>();
    let (d_packed_ptr, record_d_packed) = d_packed_view.device_ptr(stream);

    let (d_temp_ptr, record_d_temp) = d_temp.device_ptr(stream);
    let (d_output_ptr, record_d_output) = d_output.device_ptr_mut(stream);
    let (d_num_selected_ptr, record_d_num_selected) = d_num_selected.device_ptr_mut(stream);

    // CUB uses TransformInputIterator internally to read bits on-the-fly.
    ctx.launch_external(output_len, || unsafe {
        T::filter_bitmask(
            d_temp_ptr as *mut c_void,
            temp_bytes,
            d_input_ptr as *const T,
            d_packed_ptr as *const u8,
            offset,
            d_output_ptr as *mut T,
            d_num_selected_ptr as *mut i64,
            len,
            stream_ptr,
        )
        .map_err(|e| vortex_err!("CUB filter_bitmask failed: {}", e))
    })?;
    drop((
        record_d_input,
        record_d_packed,
        record_d_temp,
        record_d_output,
        record_d_num_selected,
    ));

    // Wrap the device buffer of outputs back up into a BufferHandle.
    Ok(BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(
        d_output,
    ))))
}
