// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use futures::future::BoxFuture;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ToCanonical;
use vortex_array::arrays::BoolArrayParts;
use vortex_array::arrays::DecimalArrayParts;
use vortex_array::arrays::PrimitiveArrayParts;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::StructArrayParts;
use vortex_array::buffer::BufferHandle;
use vortex_array::vtable::ValidityHelper;
use vortex_dtype::DecimalType;
use vortex_dtype::datetime::AnyTemporal;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::CudaExecutionCtx;
use crate::arrow::ArrowArray;
use crate::arrow::ArrowDeviceArray;
use crate::arrow::DeviceType;
use crate::arrow::ExportDeviceArray;
use crate::arrow::PrivateData;
use crate::arrow::SyncEvent;
use crate::arrow::check_validity_empty;
use crate::arrow::ensure_device_resident;
use crate::arrow::varbinview::BinaryParts;
use crate::arrow::varbinview::copy_varbinview_to_varbin;
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

        let (arrow_array, _) = export_canonical(cuda_array, ctx).await?;

        Ok(ArrowDeviceArray {
            array: arrow_array,
            sync_event: None,
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
            Canonical::Primitive(primitive) => {
                let len = primitive.len();
                let PrimitiveArrayParts {
                    buffer, validity, ..
                } = primitive.into_parts();

                check_validity_empty(&validity)?;

                let buffer = ensure_device_resident(buffer, ctx).await?;

                export_fixed_size(buffer, len, 0, ctx)
            }
            Canonical::Null(null_array) => {
                let len = null_array.len();

                // The null array has no buffers, no children, just metadata.
                let mut array = ArrowArray::empty();
                array.length = len as i64;
                array.null_count = len as i64;
                array.release = Some(release_array);

                // we don't need a sync event for Null since no data is copied.
                Ok((array, None))
            }
            Canonical::Decimal(decimal) => {
                let len = decimal.len();
                let DecimalArrayParts {
                    values,
                    values_type,
                    validity,
                    ..
                } = decimal.into_parts();

                // verify that there is no null buffer
                check_validity_empty(&validity)?;

                // TODO(aduffy): GPU kernel for upcasting.
                vortex_ensure!(
                    values_type >= DecimalType::I32,
                    "cannot export DecimalArray with values type {values_type}. must be i32 or wider."
                );

                let buffer = ensure_device_resident(values, ctx).await?;

                export_fixed_size(buffer, len, 0, ctx)
            }
            Canonical::Extension(extension) => {
                if !extension.ext_dtype().is::<AnyTemporal>() {
                    vortex_bail!("only support temporal extension types currently");
                }

                let values = extension.storage().to_primitive();
                let len = extension.len();

                let PrimitiveArrayParts {
                    buffer, validity, ..
                } = values.into_parts();

                check_validity_empty(&validity)?;

                let buffer = ensure_device_resident(buffer, ctx).await?;
                export_fixed_size(buffer, len, 0, ctx)
            }
            Canonical::Bool(bool_array) => {
                let BoolArrayParts {
                    bits,
                    offset,
                    len,
                    validity,
                    ..
                } = bool_array.into_parts();

                check_validity_empty(&validity)?;

                export_fixed_size(bits, len, offset, ctx)
            }
            Canonical::VarBinView(varbinview) => {
                let len = varbinview.len();
                check_validity_empty(varbinview.validity())?;

                let BinaryParts { offsets, bytes } =
                    copy_varbinview_to_varbin(varbinview, ctx).await?;

                let offsets = ensure_device_resident(offsets, ctx).await?;
                let bytes = ensure_device_resident(bytes, ctx).await?;

                let buffers = vec![None, Some(offsets), Some(bytes)];
                let mut private_data = PrivateData::new(buffers, vec![], ctx)?;
                let sync_event = private_data.sync_event();
                //
                let arrow_array = ArrowArray {
                    length: len as i64,
                    null_count: 0,
                    offset: 0,
                    // 1 (optional) buffer for nulls, one buffer for the data
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
            // TODO(aduffy): implement VarBinView. cudf doesn't support it, so we need to
            //  execute a kernel to translate from VarBinView -> VarBin.
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

    check_validity_empty(&validity)?;

    // We need the children to be held across await points.
    let mut children = Vec::with_capacity(fields.len());

    for field in fields.iter() {
        let cuda_field = field.clone().execute_cuda(ctx).await?;
        let (arrow_field, _) = export_canonical(cuda_field, ctx).await?;
        children.push(arrow_field);
    }

    let mut private_data = PrivateData::new(vec![None], children, ctx)?;
    let sync_event: SyncEvent = private_data.sync_event();

    // Populate the ArrowArray with the child arrays.
    let mut arrow_struct = ArrowArray::empty();
    arrow_struct.length = len as i64;
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

/// Export fixed-size array data that owns a single buffer of values.
fn export_fixed_size(
    buffer: BufferHandle,
    len: usize,
    offset: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<(ArrowArray, SyncEvent)> {
    vortex_ensure!(
        buffer.is_on_device(),
        "buffer must already be copied to device before calling"
    );

    // TODO(aduffy): currently the null buffer is always None, in the future we will need
    //  to pass it.
    let mut private_data = PrivateData::new(vec![None, Some(buffer)], vec![], ctx)?;
    let sync_event: SyncEvent = private_data.sync_event();

    // Return a copy of the CudaEvent
    let arrow_array = ArrowArray {
        length: len as i64,
        null_count: 0,
        offset: offset as i64,
        // 1 (optional) buffer for nulls, one buffer for the data
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

// export some nested data

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
