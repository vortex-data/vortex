// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem;
use std::ptr;

use async_trait::async_trait;
use futures::future::BoxFuture;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::bool::BoolDataParts;
use vortex::array::arrays::decimal::DecimalDataParts;
use vortex::array::arrays::extension::ExtensionArrayExt;
use vortex::array::arrays::primitive::PrimitiveDataParts;
use vortex::array::arrays::struct_::StructDataParts;
use vortex::array::arrays::varbinview::VarBinViewDataParts;
use vortex::array::buffer::BufferHandle;
use vortex::buffer::Buffer;
use vortex::dtype::DecimalType;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;
use vortex::extension::datetime::AnyTemporal;

use crate::CudaExecutionCtx;
use crate::arrow::ARROW_DEVICE_CUDA;
use crate::arrow::ArrowArray;
use crate::arrow::ArrowDeviceArray;
use crate::arrow::ExportDeviceArray;
use crate::arrow::PrivateData;
use crate::arrow::SyncEvent;
use crate::arrow::check_validity_empty;
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
            device_id: ctx.stream().context().ordinal() as i64,
            device_type: ARROW_DEVICE_CUDA,
            sync_event,
            reserved: Default::default(),
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
                let PrimitiveDataParts {
                    buffer, validity, ..
                } = primitive.into_data_parts();

                check_validity_empty(&validity)?;

                let buffer = ctx.ensure_on_device(buffer).await?;

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
                Ok((array, ptr::null_mut()))
            }
            Canonical::Decimal(decimal) => {
                let len = decimal.len();
                let DecimalDataParts {
                    values,
                    values_type,
                    validity,
                    ..
                } = decimal.into_data_parts();

                // verify that there is no null buffer
                check_validity_empty(&validity)?;

                // TODO(aduffy): GPU kernel for upcasting.
                vortex_ensure!(
                    values_type >= DecimalType::I32,
                    "cannot export DecimalArray with values type {values_type}. must be i32 or wider."
                );

                let buffer = ctx.ensure_on_device(values).await?;

                export_fixed_size(buffer, len, 0, ctx)
            }
            Canonical::Extension(extension) => {
                if !extension.ext_dtype().is::<AnyTemporal>() {
                    vortex_bail!("only support temporal extension types currently");
                }

                let values = extension
                    .storage_array()
                    .clone()
                    .execute::<PrimitiveArray>(ctx.execution_ctx())?;
                let len = extension.len();

                let PrimitiveDataParts {
                    buffer, validity, ..
                } = values.into_data_parts();

                check_validity_empty(&validity)?;

                let buffer = ctx.ensure_on_device(buffer).await?;
                export_fixed_size(buffer, len, 0, ctx)
            }
            Canonical::Bool(bool_array) => {
                let len = bool_array.len();
                let validity = bool_array.validity()?;
                let BoolDataParts {
                    bits, offset, len, ..
                } = bool_array.into_data().into_parts(len);

                check_validity_empty(&validity)?;

                let bits = ctx.ensure_on_device(bits).await?;
                export_fixed_size(bits, len, offset, ctx)
            }
            Canonical::VarBinView(varbinview) => {
                let len = varbinview.len();
                let VarBinViewDataParts {
                    views,
                    buffers: data_buffers,
                    validity,
                    ..
                } = varbinview.into_data_parts();

                check_validity_empty(&validity)?;

                let views = ctx.ensure_on_device(views).await?;
                let mut buffers = Vec::with_capacity(data_buffers.len() + 3);
                buffers.push(None);
                buffers.push(Some(views));
                for buffer in data_buffers.iter() {
                    buffers.push(Some(ctx.ensure_on_device(buffer.clone()).await?));
                }
                // Nanoarrow's Utf8View/BinaryView C layout stores the variadic data buffer sizes
                // as the final buffer slot, after the null bitmap, views, and data buffers.
                let variadic_buffer_sizes = data_buffers
                    .iter()
                    .map(|buffer| i64::try_from(buffer.len()))
                    .collect::<Result<Vec<_>, _>>()?;
                buffers.push(Some(
                    ctx.ensure_on_device(BufferHandle::new_host(
                        Buffer::from(variadic_buffer_sizes).into_byte_buffer(),
                    ))
                    .await?,
                ));

                let n_buffers = i64::try_from(buffers.len())?;
                let mut private_data = PrivateData::new(buffers, vec![], ctx)?;
                let sync_event = private_data.sync_event();
                let arrow_array = ArrowArray {
                    length: len as i64,
                    null_count: 0,
                    offset: 0,
                    // Arrow Utf8View/BinaryView layout: optional null bitmap, views, data buffers,
                    // and trailing variadic buffer sizes.
                    n_buffers,
                    buffers: private_data.buffer_ptrs.as_mut_ptr(),
                    n_children: 0,
                    children: ptr::null_mut(),
                    release: Some(release_array),
                    dictionary: ptr::null_mut(),
                    private_data: Box::into_raw(private_data).cast(),
                };

                Ok((arrow_array, sync_event))
            }
            c => vortex_bail!("unsupported Arrow Device export for {} array", c.dtype()),
        }
    })
}

async fn export_struct(
    array: StructArray,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<(ArrowArray, SyncEvent)> {
    let len = array.len();
    let StructDataParts {
        validity, fields, ..
    } = array.into_data_parts();

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

    // Non-trivial validity is rejected before fixed-size export, so the Arrow null bitmap slot is
    // always null for now. Future nullable export support should pass the validity bitmap here.
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
        children: ptr::null_mut(),
        release: Some(release_array),
        dictionary: ptr::null_mut(),
        private_data: Box::into_raw(private_data).cast(),
    };

    Ok((arrow_array, sync_event))
}

unsafe extern "C" fn release_array(array: *mut ArrowArray) {
    // SAFETY: this is only safe if we're dropping an ArrowArray that was created from Rust
    //  code. This is necessary to ensure that the fields inside the CudaPrivateData
    //  get dropped to free native/GPU memory.
    unsafe {
        if array.is_null() || (*array).release.is_none() {
            return;
        }

        let private_data_ptr = ptr::replace(&raw mut (*array).private_data, ptr::null_mut());

        if !private_data_ptr.is_null() {
            let mut private_data = Box::from_raw(private_data_ptr.cast::<PrivateData>());
            release_children(&mut private_data);
        }

        // update the release function to NULL to avoid any possibility of double-frees.
        (*array).release = None;
    }
}

unsafe fn release_children(private_data: &mut PrivateData) {
    unsafe {
        let children = mem::take(&mut private_data.children);
        for child in children {
            if !child.is_null() {
                if let Some(release) = (*child).release {
                    release(child);
                }
                // Children are allocated with Box::into_raw in PrivateData::new, so the
                // release callback must also reclaim the ArrowArray allocation itself.
                drop(Box::from_raw(child));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Schema;
    use rstest::rstest;
    use vortex::array::ArrayRef;
    use vortex::array::IntoArray;
    use vortex::array::arrays::BoolArray;
    use vortex::array::arrays::DecimalArray;
    use vortex::array::arrays::NullArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::arrays::TemporalArray;
    use vortex::array::arrays::VarBinViewArray;
    use vortex::array::validity::Validity;
    use vortex::dtype::DecimalDType;
    use vortex::dtype::FieldNames;
    use vortex::error::VortexExpect;
    use vortex::error::VortexResult;
    use vortex::extension::datetime::TimeUnit;
    use vortex::session::VortexSession;

    use crate::arrow::ARROW_DEVICE_CUDA;
    use crate::arrow::ArrowArray;
    use crate::arrow::DeviceArrayExt;
    use crate::session::CudaSession;

    unsafe fn release_exported_array(array: *mut ArrowArray) {
        unsafe {
            if let Some(release) = (*array).release {
                release(array);
            }
        }
    }

    #[rstest]
    #[case::u8(PrimitiveArray::from_iter(0u8..10).into_array(), 10)]
    #[case::u16(PrimitiveArray::from_iter(0u16..10).into_array(), 10)]
    #[case::u32(PrimitiveArray::from_iter(0u32..10).into_array(), 10)]
    #[case::u64(PrimitiveArray::from_iter(0u64..10).into_array(), 10)]
    #[case::i32(PrimitiveArray::from_iter(0i32..10).into_array(), 10)]
    #[case::i64(PrimitiveArray::from_iter(0i64..10).into_array(), 10)]
    #[case::f32(PrimitiveArray::from_iter([1.0f32, 2.0, 3.0]).into_array(), 3)]
    #[case::f64(PrimitiveArray::from_iter([1.0f64, 2.0, 3.0]).into_array(), 3)]
    #[crate::test]
    async fn test_export_primitive(
        #[case] array: ArrayRef,
        #[case] expected_len: i64,
    ) -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let mut device_array = array.export_device_array(&mut ctx).await?;

        assert_eq!(device_array.array.length, expected_len);
        assert_eq!(device_array.array.null_count, 0);
        assert_eq!(device_array.array.offset, 0);
        assert_eq!(device_array.array.n_buffers, 2);
        assert_eq!(device_array.array.n_children, 0);
        assert!(device_array.array.release.is_some());
        assert_eq!(device_array.device_type, ARROW_DEVICE_CUDA);

        unsafe { release_exported_array(&raw mut device_array.array) };
        Ok(())
    }

    #[crate::test]
    async fn test_export_null() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let array = NullArray::new(7).into_array();
        let mut device_array = array.export_device_array(&mut ctx).await?;

        assert_eq!(device_array.array.length, 7);
        assert_eq!(device_array.array.null_count, 7);
        assert_eq!(device_array.device_type, ARROW_DEVICE_CUDA);

        unsafe { release_exported_array(&raw mut device_array.array) };
        Ok(())
    }

    #[crate::test]
    async fn test_export_decimal() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let array = DecimalArray::from_iter(0i128..5, DecimalDType::new(38, 2)).into_array();
        let mut device_array = array.export_device_array(&mut ctx).await?;

        assert_eq!(device_array.array.length, 5);
        assert_eq!(device_array.array.null_count, 0);
        assert_eq!(device_array.array.n_buffers, 2);
        assert_eq!(device_array.array.n_children, 0);
        assert!(device_array.array.release.is_some());
        assert_eq!(device_array.device_type, ARROW_DEVICE_CUDA);

        unsafe { release_exported_array(&raw mut device_array.array) };
        Ok(())
    }

    #[crate::test]
    async fn test_export_temporal() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let array = TemporalArray::new_date(
            PrimitiveArray::from_iter([100i32, 200, 300]).into_array(),
            TimeUnit::Days,
        )
        .into_array();
        let mut device_array = array.export_device_array(&mut ctx).await?;

        assert_eq!(device_array.array.length, 3);
        assert_eq!(device_array.array.null_count, 0);
        assert_eq!(device_array.array.n_buffers, 2);
        assert_eq!(device_array.array.n_children, 0);
        assert!(device_array.array.release.is_some());
        assert_eq!(device_array.device_type, ARROW_DEVICE_CUDA);

        unsafe { release_exported_array(&raw mut device_array.array) };
        Ok(())
    }

    #[crate::test]
    async fn test_export_bool() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let array = BoolArray::from_iter([true, false, true]).into_array();
        let mut device_array = array.export_device_array(&mut ctx).await?;

        assert_eq!(device_array.array.length, 3);
        assert_eq!(device_array.array.null_count, 0);
        assert_eq!(device_array.array.n_buffers, 2);
        assert_eq!(device_array.array.n_children, 0);
        assert!(device_array.array.release.is_some());
        assert_eq!(device_array.device_type, ARROW_DEVICE_CUDA);

        unsafe { release_exported_array(&raw mut device_array.array) };
        Ok(())
    }

    #[crate::test]
    async fn test_export_varbinview() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let array = VarBinViewArray::from_iter_str([
            "hello",
            "world",
            "this is a longer string for out-of-line storage",
        ])
        .into_array();
        let mut device_array = array.export_device_array(&mut ctx).await?;

        assert_eq!(device_array.array.length, 3);
        assert_eq!(device_array.array.null_count, 0);
        // VarBinView export: null buffer + views + data buffers + variadic buffer sizes
        assert_eq!(device_array.array.n_buffers, 4);
        let n_buffers = usize::try_from(device_array.array.n_buffers)?;
        let buffers = unsafe { std::slice::from_raw_parts(device_array.array.buffers, n_buffers) };
        assert!(buffers[0].is_null());
        assert!(!buffers[1].is_null());
        assert!(!buffers[2].is_null());
        assert!(!buffers[3].is_null());
        assert_eq!(device_array.array.n_children, 0);
        assert!(device_array.array.release.is_some());
        assert_eq!(device_array.device_type, ARROW_DEVICE_CUDA);

        unsafe { release_exported_array(&raw mut device_array.array) };
        Ok(())
    }

    #[crate::test]
    async fn test_export_struct() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let array = StructArray::new(
            FieldNames::from_iter(["a", "b"]),
            vec![
                PrimitiveArray::from_iter(0u32..5).into_array(),
                PrimitiveArray::from_iter(0i64..5).into_array(),
            ],
            5,
            Validity::NonNullable,
        )
        .into_array();
        let mut device_array = array.export_device_array(&mut ctx).await?;

        assert_eq!(device_array.array.length, 5);
        assert_eq!(device_array.array.null_count, 0);
        // Struct has a single (null) validity buffer
        assert_eq!(device_array.array.n_buffers, 1);
        assert_eq!(device_array.array.n_children, 2);
        assert!(device_array.array.release.is_some());
        assert_eq!(device_array.device_type, ARROW_DEVICE_CUDA);

        unsafe { release_exported_array(&raw mut device_array.array) };
        Ok(())
    }

    #[crate::test]
    async fn test_export_struct_with_schema() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let array = StructArray::new(
            FieldNames::from_iter(["a", "b", "c"]),
            vec![
                PrimitiveArray::from_iter(0u32..5).into_array(),
                PrimitiveArray::from_iter(0i64..5).into_array(),
                VarBinViewArray::from_iter_str(["one", "two", "three", "four", "five"])
                    .into_array(),
            ],
            5,
            Validity::NonNullable,
        )
        .into_array();
        let mut exported = array.export_device_array_with_schema(&mut ctx).await?;

        let schema = Schema::try_from(&exported.schema)?;
        assert_eq!(
            schema,
            Schema::new(vec![
                Field::new("a", DataType::UInt32, false),
                Field::new("b", DataType::Int64, false),
                Field::new("c", DataType::Utf8View, false),
            ])
        );
        assert_eq!(exported.array.array.length, 5);
        assert_eq!(exported.array.array.n_buffers, 1);
        assert_eq!(exported.array.array.n_children, 3);
        assert_eq!(exported.array.device_type, ARROW_DEVICE_CUDA);

        unsafe { release_exported_array(&raw mut exported.array.array) };
        Ok(())
    }

    #[crate::test]
    async fn test_export_primitive_with_schema_is_column_shaped() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let array = PrimitiveArray::from_iter(0u32..5).into_array();
        let mut exported = array.export_device_array_with_schema(&mut ctx).await?;

        let field = Field::try_from(&exported.schema)?;
        assert_eq!(field, Field::new("", DataType::UInt32, false));
        assert_eq!(exported.array.array.length, 5);
        assert_eq!(exported.array.array.n_buffers, 2);
        assert_eq!(exported.array.array.n_children, 0);
        assert_eq!(exported.array.device_type, ARROW_DEVICE_CUDA);

        unsafe { release_exported_array(&raw mut exported.array.array) };
        Ok(())
    }

    #[crate::test]
    async fn test_export_varbinview_with_schema_uses_utf8_view_layout() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let array = VarBinViewArray::from_iter_str([
            "one",
            "two",
            "this is a longer string for out-of-line storage",
        ])
        .into_array();
        let mut exported = array.export_device_array_with_schema(&mut ctx).await?;

        let field = Field::try_from(&exported.schema)?;
        assert_eq!(field, Field::new("", DataType::Utf8View, false));
        assert_eq!(exported.array.array.length, 3);
        assert_eq!(exported.array.array.n_buffers, 4);
        assert_eq!(exported.array.array.n_children, 0);
        assert_eq!(exported.array.device_type, ARROW_DEVICE_CUDA);

        unsafe { release_exported_array(&raw mut exported.array.array) };
        Ok(())
    }
}
