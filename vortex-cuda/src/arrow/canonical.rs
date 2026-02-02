// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use cudarc::driver::sys;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::PrimitiveArrayParts;
use vortex_array::buffer::BufferHandle;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::CudaBufferExt;
use crate::CudaExecutionCtx;
use crate::arrow::ArrowArray;
use crate::arrow::ArrowDeviceArray;
use crate::arrow::CudaDeviceArrayExecute;
use crate::arrow::CudaPrivateData;
use crate::arrow::DeviceType;
use crate::executor::CudaArrayExt;

// Impl it for the execution context instead here...I think this is right?
impl CudaDeviceArrayExecute for Canonical {
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<ArrowDeviceArray> {
        let cuda_array = array.execute_cuda(ctx).await?;

        let arrow_array = match cuda_array {
            Canonical::Primitive(primitive) => export_primitive(primitive, ctx).await,
            c => todo!("implement support for exporting {}", c.dtype()),
        };

        Ok(ArrowDeviceArray {
            array: arrow_array,
            device_id: 0,
            device_type: DeviceType::Cuda,
            sync_event: None,
            _reserved: Default::default(),
        })
    }
}

fn export_primitive(array: PrimitiveArray, ctx: &mut CudaExecutionCtx) -> VortexResult<ArrowArray> {
    let len = array.len();
    let PrimitiveArrayParts {
        buffer,
        ptype,
        validity,
        ..
    } = array.into_parts();

    unsafe extern "C" fn release(array: *mut ArrowArray) {
        // SAFETY: this is only safe if the caller provides a valid pointer to an `ArrowArray`.
        drop(unsafe { Box::from_raw(array) });
    }

    let null_count = match validity {
        Validity::NonNullable | Validity::AllValid => 0,
        Validity::AllInvalid => len,
        Validity::Array(_) => {
            vortex_bail!("Exporting PrimitiveArray with non-trivial validity not supported yet")
        }
    };

    // TODO(aduffy): currently the null buffer is always empty, in the future we will need
    //  to pass it.
    let buffers: Box<[Option<BufferHandle>]> = vec![None, Some(buffer)].into_boxed_slice();

    let buffer_ptrs: Box<[sys::CUdeviceptr]> = buffers
        .iter()
        .map(|buf| {
            match buf {
                None => {
                    // null pointer
                    Ok(sys::CUdeviceptr::default())
                }
                Some(handle) => handle.cuda_device_ptr(),
            }
        })
        .collect::<VortexResult<Vec<_>>>()?
        .into_boxed_slice();

    let mut private_data = Box::new(CudaPrivateData {
        cuda_stream: Arc::clone(ctx.stream()),
        buffers,
        buffer_ptrs,
    });

    Ok(ArrowArray {
        length: len as i64,
        null_count: null_count as i64,
        offset: 0,
        // 1 (optional) buffer for nulls, one buffer for data
        n_buffers: 2,
        buffers: private_data.buffer_ptrs.as_mut_ptr(),
        n_children: 0,
        children: std::ptr::null_mut(),
        release: Some(release),
        dictionary: std::ptr::null_mut(),
        private_data: Box::into_raw(private_data).cast(),
    })
}

// Get the DecimalArray and the VarBinViewArray so we know
// how to treat all of these timestamps and such.

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn test_export_primitive() {
    }
}
