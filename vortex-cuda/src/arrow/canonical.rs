// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ptr::NonNull;
use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::sys;
use futures::future::BoxFuture;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::DecimalArrayParts;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::PrimitiveArrayParts;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::StructArrayParts;
use vortex_array::buffer::BufferHandle;
use vortex_array::validity::Validity;
use vortex_dtype::DecimalType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::CudaBufferExt;
use crate::CudaExecutionCtx;
use crate::arrow::ArrowArray;
use crate::arrow::ArrowDeviceArray;
use crate::arrow::DeviceType;
use crate::arrow::ExportDeviceArray;
use crate::arrow::PrivateData;
use crate::arrow::SyncEvent;
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

        let (arrow_array, sync_event) = export_canonical(cuda_array, ctx).await?;

        Ok(ArrowDeviceArray {
            array: arrow_array,
            sync_event,
            device_id: ctx.stream().context().ordinal() as i64,
            device_type: DeviceType::Cuda,
            _reserved: Default::default(),
        })
    }
}

fn export_canonical(
    cuda_array: Canonical,
    ctx: &mut CudaExecutionCtx,
) -> BoxFuture<'_, VortexResult<(ArrowArray, SyncEvent)>> {
    Box::pin(async {
        match cuda_array {
            Canonical::Struct(struct_array) => export_struct(struct_array, ctx).await,
            Canonical::Primitive(primitive) => export_primitive(primitive, ctx).await,
            Canonical::Decimal(decimal) => export_decimal(decimal, ctx).await,
            c => todo!("support for exporting {} arrays", c.dtype()),
        }
    })
}

async fn export_struct(
    array: StructArray,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<(ArrowArray, SyncEvent)> {
    let len = array.len();
    let StructArrayParts {
        validity, fields, ..
    } = array.into_parts();

    let null_count = match validity {
        Validity::NonNullable | Validity::AllValid => 0,
        _ => {
            vortex_bail!("Exporting PrimitiveArray with non-trivial validity not supported yet")
        }
    };

    // We need the children to be held across await points.
    let mut children = Vec::with_capacity(fields.len());

    for field in fields.iter() {
        let cuda_field = field.clone().execute_cuda(ctx).await?;
        let (arrow_field, _) = export_canonical(cuda_field, ctx).await?;
        children.push(arrow_field);
    }

    let cuda_event = ctx
        .stream()
        .record_event(None)
        .map_err(|_| vortex_err!("failed to create cudaEvent_t"))?;

    let children = children
        .into_iter()
        .map(|array| Box::into_raw(Box::new(array)))
        .collect::<Box<[_]>>();

    let buffer_ptrs = vec![sys::CUdeviceptr::default()].into_boxed_slice();

    let mut private_data = Box::new(PrivateData {
        cuda_stream: Arc::clone(ctx.stream()),
        buffers: Box::new([None]),
        buffer_ptrs,
        cuda_event_ptr: cuda_event.cu_event().cast(),
        cuda_event,
        children,
    });

    let sync_event: SyncEvent = NonNull::new(&raw mut private_data.cuda_event_ptr);

    // Populate the ArrowArray with the child arrays.
    let mut arrow_struct = ArrowArray::empty();
    arrow_struct.length = len as i64;
    arrow_struct.null_count = null_count as i64;
    arrow_struct.n_children = fields.len() as i64;
    arrow_struct.children = private_data.children.as_mut_ptr();

    // StructArray _can_ contain a validity buffer. In this case, we just write a null pointer
    // for it.
    arrow_struct.n_buffers = 1;
    arrow_struct.buffers = private_data.buffer_ptrs.as_mut_ptr();
    arrow_struct.release = Some(release_array);
    arrow_struct.private_data = Box::into_raw(private_data).cast();

    Ok((arrow_struct, sync_event))
}

async fn export_primitive(
    array: PrimitiveArray,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<(ArrowArray, SyncEvent)> {
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
        _ => {
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

    // Create an event object that can be synchronized on to wait for any writes in this stream
    // to complete.
    // This is stored in the PrivateData so that it will be dropped when the native code calls
    // the arrow_array->release callback.
    let cuda_event = ctx
        .stream()
        .record_event(None)
        .map_err(|_| vortex_err!("failed to create cudaEvent_t"))?;

    let mut private_data = Box::new(PrivateData {
        cuda_stream: Arc::clone(ctx.stream()),
        children: Box::new([]),
        buffers,
        buffer_ptrs,
        cuda_event_ptr: cuda_event.cu_event().cast(),
        cuda_event,
    });

    // The sync_event should point to the cudaEvent_t saved in the private data
    let sync_event: SyncEvent = NonNull::new(&raw mut private_data.cuda_event_ptr);

    // Return a copy of the CudaEvent
    let arrow_array = ArrowArray {
        length: len as i64,
        null_count: null_count as i64,
        offset: 0,
        // 1 (optional) buffer for nulls, one buffer for data
        n_buffers: 2,
        buffers: private_data.buffer_ptrs.as_mut_ptr(),
        n_children: 0,
        children: std::ptr::null_mut(),
        release: Some(release_array),
        dictionary: std::ptr::null_mut(),
        private_data: Box::into_raw(private_data).cast(),
    };

    Ok((arrow_array, sync_event))
}

async fn export_decimal(
    array: DecimalArray,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<(ArrowArray, SyncEvent)> {
    let len = array.len();
    let DecimalArrayParts {
        values,
        values_type,
        validity,
        ..
    } = array.into_parts();

    // TODO(aduffy): GPU kernel for upcasting.
    vortex_ensure!(
        values_type >= DecimalType::I32,
        "cannot export DecimalArray with values type {values_type}. must be i32 or wider."
    );

    let buffer = if values.is_on_device() {
        values
    } else {
        // TODO(aduffy): I don't think this type parameter does anything
        ctx.move_to_device::<u8>(values)?.await?
    };

    let null_count = match validity {
        Validity::NonNullable | Validity::AllValid => 0,
        _ => {
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

    // Create an event object that can be synchronized on to wait for any writes in this stream
    // to complete.
    // This is stored in the PrivateData so that it will be dropped when the native code calls
    // the arrow_array->release callback.
    let cuda_event = ctx
        .stream()
        .record_event(None)
        .map_err(|_| vortex_err!("failed to create cudaEvent_t"))?;

    let mut private_data = Box::new(PrivateData {
        cuda_stream: Arc::clone(ctx.stream()),
        children: Box::new([]),
        buffers,
        buffer_ptrs,
        cuda_event_ptr: cuda_event.cu_event().cast(),
        cuda_event,
    });

    // The sync_event should point to the cudaEvent_t saved in the private data
    let sync_event: SyncEvent = NonNull::new(&raw mut private_data.cuda_event_ptr);

    // Return a copy of the CudaEvent
    let arrow_array = ArrowArray {
        length: len as i64,
        null_count: null_count as i64,
        offset: 0,
        // 1 (optional) buffer for nulls, one buffer for data
        n_buffers: 2,
        buffers: private_data.buffer_ptrs.as_mut_ptr(),
        n_children: 0,
        children: std::ptr::null_mut(),
        release: Some(release_array),
        dictionary: std::ptr::null_mut(),
        private_data: Box::into_raw(private_data).cast(),
    };

    Ok((arrow_array, sync_event))
}

unsafe extern "C" fn release_array(array: *mut ArrowArray) {
    // SAFETY: this is only safe if we're dropping an ArrowArray that was created from Rust
    //  code. This is necessary to ensure that the fields inside the CudaPrivateData
    //  get dropped to free native/GPU memory.
    unsafe {
        let private_data_ptr =
            std::ptr::replace(&raw mut (*array).private_data, std::ptr::null_mut());

        if !private_data_ptr.is_null() {
            let mut private_data = Box::from_raw(private_data_ptr.cast::<PrivateData>());
            let children = std::mem::take(&mut private_data.children);
            for child in children {
                release_array(child);
            }
            drop(private_data);
        }

        // update the release function to NULL to avoid any possibility of double-frees.
        (*array).release = None;
    }
}
