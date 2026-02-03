// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::result;
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
use crate::arrow::CudaPrivateData;
use crate::arrow::DeviceType;
use crate::arrow::ExportDeviceArray;
use crate::executor::CudaArrayExt;

/// An implementation of `ExportDeviceArray` that exports Vortex arrays to `ArrowDeviceArray` by
/// first decoding the array on the GPU and then converting the canonical type to the nearest
/// Arrow equivalent.
#[derive(Debug)]
pub(crate) struct CanonicalDeviceArrayExport;

#[async_trait]
impl ExportDeviceArray for CanonicalDeviceArrayExport {
    async fn export_device_array(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<ArrowDeviceArray> {
        let cuda_array = array.execute_cuda(ctx).await?;

        let arrow_array = match cuda_array {
            Canonical::Primitive(primitive) => export_primitive(primitive, ctx).await?,
            // Canonical::Decimal(decimal) => todo!("export decimal"),
            // Canonical::VarBinView(varbinview) => todo!("export varbinview"),
            c => todo!("implement support for exporting {}", c.dtype()),
        };

        ctx.stream()
            .record_event();

        Ok(ArrowDeviceArray {
            array: arrow_array,
            device_id: ctx.stream().context().ordinal() as i64,
            device_type: DeviceType::Cuda,
            sync_event: None,
            _reserved: Default::default(),
        })
    }
}

async fn export_primitive(
    array: PrimitiveArray,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<ArrowArray> {
    unsafe extern "C" fn release(array: *mut ArrowArray) {
        // SAFETY: this is only safe if we're dropping an ArrowArray that was created from Rust
        //  code. This is necessary to ensure that the fields inside the CudaPrivateData
        //  get dropped to free native/GPU memory.
        unsafe {
            let private_data_ptr =
                std::ptr::replace(&raw mut (*array).private_data, std::ptr::null_mut());
            drop(Box::from_raw(private_data_ptr.cast::<CudaPrivateData>()));

            // update the release function to NULL to avoid any possibility of double-frees.
            (*array).release = None;
        }
    }
    let len = array.len();
    let PrimitiveArrayParts {
        buffer, validity, ..
    } = array.into_parts();

    let buffer = if buffer.is_on_device() {
        buffer
    } else {
        // TODO(aduffy): I don't think this type parameter does anything
        ctx.move_to_device::<u8>(buffer)?.await?
    };

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
