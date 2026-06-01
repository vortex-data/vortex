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
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::buffer::ByteBuffer;
use vortex::dtype::DecimalType;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;
use vortex::extension::datetime::AnyTemporal;
use vortex::mask::Mask;

use crate::CudaExecutionCtx;
use crate::arrow::ARROW_DEVICE_CUDA;
use crate::arrow::ArrowArray;
use crate::arrow::ArrowDeviceArray;
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

                let (validity_buffer, null_count) =
                    export_arrow_validity_buffer(validity, len, 0, ctx).await?;
                let buffer = ctx.ensure_on_device(buffer).await?;

                export_fixed_size(buffer, len, 0, validity_buffer, null_count, ctx)
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

                // TODO(aduffy): GPU kernel for upcasting.
                vortex_ensure!(
                    values_type >= DecimalType::I32,
                    "cannot export DecimalArray with values type {values_type}. must be i32 or wider."
                );

                let (validity_buffer, null_count) =
                    export_arrow_validity_buffer(validity, len, 0, ctx).await?;
                let buffer = ctx.ensure_on_device(values).await?;

                export_fixed_size(buffer, len, 0, validity_buffer, null_count, ctx)
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

                let (validity_buffer, null_count) =
                    export_arrow_validity_buffer(validity, len, 0, ctx).await?;

                let buffer = ctx.ensure_on_device(buffer).await?;
                export_fixed_size(buffer, len, 0, validity_buffer, null_count, ctx)
            }
            Canonical::Bool(bool_array) => {
                let len = bool_array.len();
                let validity = bool_array.validity()?;
                let BoolDataParts {
                    bits, offset, len, ..
                } = bool_array.into_data().into_parts(len);

                let (validity_buffer, null_count) =
                    export_arrow_validity_buffer(validity, len, offset, ctx).await?;

                let bits = ctx.ensure_on_device(bits).await?;
                export_fixed_size(bits, len, offset, validity_buffer, null_count, ctx)
            }
            Canonical::VarBinView(varbinview) => {
                let len = varbinview.len();
                let VarBinViewDataParts {
                    views,
                    buffers: data_buffers,
                    validity,
                    ..
                } = varbinview.into_data_parts();

                let (validity_buffer, null_count) =
                    export_arrow_validity_buffer(validity, len, 0, ctx).await?;

                let views = ctx.ensure_on_device(views).await?;
                let mut buffers = Vec::with_capacity(data_buffers.len() + 3);
                buffers.push(validity_buffer);
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
                    null_count,
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

/// Export Vortex validity as an Arrow validity byte buffer.
///
/// Returns `None` for the buffer when Arrow can omit validity because all rows are valid.
async fn export_arrow_validity_buffer(
    validity: Validity,
    len: usize,
    arrow_offset: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<(Option<BufferHandle>, i64)> {
    let mask = validity.execute_mask(len, ctx.execution_ctx())?;
    let null_count = i64::try_from(mask.false_count())?;
    let validity_bits = len + arrow_offset;
    let validity_bytes = validity_bits.div_ceil(8);

    let validity_buffer = match mask {
        Mask::AllTrue(_) => return Ok((None, 0)),
        Mask::AllFalse(_) => ByteBuffer::zeroed(validity_bytes),
        values @ Mask::Values(_) => values.into_bit_buffer().into_inner().2,
    };
    let validity = ctx
        .ensure_on_device(BufferHandle::new_host(validity_buffer))
        .await?;

    Ok((Some(validity), null_count))
}

async fn export_struct(
    array: StructArray,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<(ArrowArray, SyncEvent)> {
    let len = array.len();
    let StructDataParts {
        validity, fields, ..
    } = array.into_data_parts();

    let (validity_buffer, null_count) = export_arrow_validity_buffer(validity, len, 0, ctx).await?;

    // We need the children to be held across await points.
    let mut children = Vec::with_capacity(fields.len());

    for field in fields.iter() {
        let cuda_field = field.clone().execute_cuda(ctx).await?;
        let (arrow_field, _) = export_canonical(cuda_field, ctx).await?;
        children.push(arrow_field);
    }

    let mut private_data = PrivateData::new(vec![validity_buffer], children, ctx)?;
    let sync_event: SyncEvent = private_data.sync_event();

    // Populate the ArrowArray with the child arrays.
    let mut arrow_struct = ArrowArray::empty();
    arrow_struct.length = len as i64;
    arrow_struct.null_count = null_count;
    arrow_struct.n_children = fields.len() as i64;
    arrow_struct.children = private_data.children.as_mut_ptr();

    // StructArray has one buffer slot for its optional validity bitmap.
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
    validity: Option<BufferHandle>,
    null_count: i64,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<(ArrowArray, SyncEvent)> {
    vortex_ensure!(
        buffer.is_on_device(),
        "buffer must already be copied to device before calling"
    );

    let mut private_data = PrivateData::new(vec![validity, Some(buffer)], vec![], ctx)?;
    let sync_event: SyncEvent = private_data.sync_event();

    // Return a copy of the CudaEvent
    let arrow_array = ArrowArray {
        length: len as i64,
        null_count,
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
    use std::mem::size_of;
    use std::sync::Arc;

    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Fields;
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
    use vortex::array::arrays::varbinview::BinaryView;
    use vortex::array::validity::Validity;
    use vortex::buffer::Buffer;
    use vortex::buffer::ByteBuffer;
    use vortex::dtype::DType;
    use vortex::dtype::DecimalDType;
    use vortex::dtype::FieldNames;
    use vortex::dtype::Nullability;
    use vortex::dtype::half::f16;
    use vortex::error::VortexExpect;
    use vortex::error::VortexResult;
    use vortex::extension::datetime::TimeUnit;
    use vortex::session::VortexSession;

    use crate::CudaExecutionCtx;
    use crate::arrow::ARROW_DEVICE_CUDA;
    use crate::arrow::ArrowArray;
    use crate::arrow::ArrowDeviceArray;
    use crate::arrow::DeviceArrayExt;
    use crate::arrow::PrivateData;
    use crate::session::CudaSession;

    unsafe fn release_exported_array(array: *mut ArrowArray) {
        unsafe {
            if let Some(release) = (*array).release {
                release(array);
            }
        }
    }

    // Assert Arrow Device metadata that consumers use before reading buffers.
    fn assert_device_metadata(
        device_array: &ArrowDeviceArray,
        expected_device_id: i64,
        expect_sync_event: bool,
    ) {
        assert_eq!(device_array.device_id, expected_device_id);
        assert_eq!(device_array.device_type, ARROW_DEVICE_CUDA);
        assert_eq!(device_array.reserved, [0, 0, 0]);
        assert_eq!(device_array.sync_event.is_null(), !expect_sync_event);
    }

    // Assert an exported array has a device null bitmap in buffer slot 0.
    fn assert_null_buffer(array: &ArrowArray, expected_null_count: i64) -> VortexResult<()> {
        assert_eq!(array.null_count, expected_null_count);
        let buffers =
            unsafe { std::slice::from_raw_parts(array.buffers, usize::try_from(array.n_buffers)?) };
        assert!(!buffers[0].is_null());
        Ok(())
    }

    // Export a nullable array and assert its null-buffer metadata.
    async fn assert_nullable_export(
        array: ArrayRef,
        expected_n_buffers: i64,
        expected_null_count: i64,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<ArrowDeviceArray> {
        let device_array = array.export_device_array(ctx).await?;
        assert_eq!(device_array.array.n_buffers, expected_n_buffers);
        assert_null_buffer(&device_array.array, expected_null_count)?;
        Ok(device_array)
    }

    // Assert common Utf8View/BinaryView export metadata and buffers.
    fn assert_varbinview_shape(
        array: &ArrowArray,
        expected_len: i64,
        expected_null_count: i64,
    ) -> VortexResult<()> {
        assert_eq!(array.length, expected_len);
        assert_eq!(array.null_count, expected_null_count);
        assert_eq!(array.offset, 0);
        assert_eq!(array.n_children, 0);
        assert!(array.release.is_some());
        assert!(!array.private_data.is_null());
        assert!(array.n_buffers >= 3);

        let n_buffers = usize::try_from(array.n_buffers)?;
        let buffers = unsafe { std::slice::from_raw_parts(array.buffers, n_buffers) };
        assert_eq!(buffers[0].is_null(), expected_null_count == 0);
        assert!(buffers[1..].iter().all(|buffer| !buffer.is_null()));

        let private_data = unsafe { &*array.private_data.cast::<PrivateData>() };
        assert_eq!(
            private_data.buffers[1]
                .as_ref()
                .vortex_expect("views buffer should be present")
                .len(),
            usize::try_from(expected_len)? * size_of::<BinaryView>()
        );
        assert_eq!(
            private_data.buffers[n_buffers - 1]
                .as_ref()
                .vortex_expect("variadic buffer sizes should be present")
                .len(),
            (n_buffers - 3) * size_of::<i64>()
        );

        Ok(())
    }

    // Assert exact variadic buffer count and data-buffer lengths.
    fn assert_varbinview_layout(
        array: &ArrowArray,
        expected_len: i64,
        expected_null_count: i64,
        expected_data_buffer_lengths: &[usize],
    ) -> VortexResult<()> {
        assert_varbinview_shape(array, expected_len, expected_null_count)?;

        let expected_n_buffers = expected_data_buffer_lengths.len() + 3;
        assert_eq!(usize::try_from(array.n_buffers)?, expected_n_buffers);

        let private_data = unsafe { &*array.private_data.cast::<PrivateData>() };
        for (buffer, expected_len) in private_data.buffers
            [2..2 + expected_data_buffer_lengths.len()]
            .iter()
            .zip(expected_data_buffer_lengths)
        {
            assert_eq!(
                buffer
                    .as_ref()
                    .vortex_expect("variadic data buffer should be present")
                    .len(),
                *expected_len
            );
        }
        Ok(())
    }

    // Build a VarBinView fixture with out-of-line values in separate data buffers.
    fn multi_buffer_varbinview(dtype: DType) -> (ArrayRef, [usize; 2]) {
        let first = ByteBuffer::copy_from("first value stored out-of-line".as_bytes());
        let second = ByteBuffer::copy_from("second value stored out-of-line".as_bytes());
        let buffer_lengths = [first.len(), second.len()];
        let views = Buffer::from_iter([
            BinaryView::make_view(b"inline", 0, 0),
            BinaryView::make_view(&first, 0, 0),
            BinaryView::make_view(&second, 1, 0),
        ]);

        let array = VarBinViewArray::try_new(
            views,
            Arc::from([first, second]),
            dtype,
            Validity::NonNullable,
        )
        .vortex_expect("valid multi-buffer VarBinViewArray")
        .into_array();

        (array, buffer_lengths)
    }

    // Build a nested struct fixture with an out-of-line string-view value.
    fn nested_struct_array() -> ArrayRef {
        let nested = StructArray::new(
            FieldNames::from_iter(["b", "c"]),
            vec![
                PrimitiveArray::from_iter(0i64..5).into_array(),
                VarBinViewArray::from_iter_str([
                    "one",
                    "two",
                    "this is a longer string for out-of-line storage",
                    "four",
                    "five",
                ])
                .into_array(),
            ],
            5,
            Validity::NonNullable,
        )
        .into_array();

        StructArray::new(
            FieldNames::from_iter(["a", "nested"]),
            vec![PrimitiveArray::from_iter(0u32..5).into_array(), nested],
            5,
            Validity::NonNullable,
        )
        .into_array()
    }

    #[rstest]
    #[case::u8(PrimitiveArray::from_iter(0u8..10).into_array(), 10, DataType::UInt8)]
    #[case::u16(PrimitiveArray::from_iter(0u16..10).into_array(), 10, DataType::UInt16)]
    #[case::u32(PrimitiveArray::from_iter(0u32..10).into_array(), 10, DataType::UInt32)]
    #[case::u64(PrimitiveArray::from_iter(0u64..10).into_array(), 10, DataType::UInt64)]
    #[case::i8(PrimitiveArray::from_iter(0i8..10).into_array(), 10, DataType::Int8)]
    #[case::i16(PrimitiveArray::from_iter(0i16..10).into_array(), 10, DataType::Int16)]
    #[case::i32(PrimitiveArray::from_iter(0i32..10).into_array(), 10, DataType::Int32)]
    #[case::i64(PrimitiveArray::from_iter(0i64..10).into_array(), 10, DataType::Int64)]
    #[case::f16(
        PrimitiveArray::from_iter([f16::from_f32(1.0), f16::from_f32(2.0)]).into_array(),
        2,
        DataType::Float16
    )]
    #[case::f32(
        PrimitiveArray::from_iter([1.0f32, 2.0, 3.0]).into_array(),
        3,
        DataType::Float32
    )]
    #[case::f64(
        PrimitiveArray::from_iter([1.0f64, 2.0, 3.0]).into_array(),
        3,
        DataType::Float64
    )]
    #[crate::test]
    async fn test_export_primitive(
        #[case] array: ArrayRef,
        #[case] expected_len: i64,
        #[case] expected_data_type: DataType,
    ) -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let mut exported = array.export_device_array_with_schema(&mut ctx).await?;

        let field = Field::try_from(&exported.schema)?;
        assert_eq!(field, Field::new("", expected_data_type, false));
        assert_eq!(exported.array.array.length, expected_len);
        assert_eq!(exported.array.array.null_count, 0);
        assert_eq!(exported.array.array.offset, 0);
        assert_eq!(exported.array.array.n_buffers, 2);
        assert_eq!(exported.array.array.n_children, 0);
        assert!(exported.array.array.release.is_some());
        assert_eq!(exported.array.device_type, ARROW_DEVICE_CUDA);

        unsafe { release_exported_array(&raw mut exported.array.array) };
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

        let out_of_line = "this is a longer string for out-of-line storage";
        let array = VarBinViewArray::from_iter_str(["hello", "world", out_of_line]).into_array();
        let mut device_array = array.export_device_array(&mut ctx).await?;

        assert_varbinview_layout(&device_array.array, 3, 0, &[out_of_line.len()])?;
        assert_eq!(device_array.device_type, ARROW_DEVICE_CUDA);

        unsafe { release_exported_array(&raw mut device_array.array) };
        Ok(())
    }

    #[crate::test]
    async fn test_export_binaryview_inline_outline_values() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let out_of_line = b"this binary payload is longer than twelve bytes";
        let array = VarBinViewArray::from_iter_nullable_bin([
            Some(b"" as &[u8]),
            Some(b"\x00\xff\xfe"),
            None,
            Some(b"short"),
            Some(out_of_line),
        ])
        .into_array();
        let mut exported = array.export_device_array_with_schema(&mut ctx).await?;

        let field = Field::try_from(&exported.schema)?;
        assert_eq!(field, Field::new("", DataType::BinaryView, true));
        assert_varbinview_layout(&exported.array.array, 5, 1, &[out_of_line.len()])?;
        assert_eq!(exported.array.device_type, ARROW_DEVICE_CUDA);

        unsafe { release_exported_array(&raw mut exported.array.array) };
        Ok(())
    }

    #[rstest]
    #[case::utf8(
        multi_buffer_varbinview(DType::Utf8(Nullability::NonNullable)),
        DataType::Utf8View
    )]
    #[case::binary(
        multi_buffer_varbinview(DType::Binary(Nullability::NonNullable)),
        DataType::BinaryView
    )]
    #[crate::test]
    async fn test_export_varbinview_multiple_variadic_buffers(
        #[case] fixture: (ArrayRef, [usize; 2]),
        #[case] expected_data_type: DataType,
    ) -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let (array, expected_data_buffer_lengths) = fixture;
        let mut exported = array.export_device_array_with_schema(&mut ctx).await?;

        let field = Field::try_from(&exported.schema)?;
        assert_eq!(field, Field::new("", expected_data_type, false));
        assert_varbinview_layout(&exported.array.array, 3, 0, &expected_data_buffer_lengths)?;
        assert_eq!(exported.array.device_type, ARROW_DEVICE_CUDA);

        unsafe { release_exported_array(&raw mut exported.array.array) };
        Ok(())
    }

    // Check device metadata for data-bearing and metadata-only exports.
    #[crate::test]
    async fn test_export_device_metadata() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");
        let expected_device_id = ctx.stream().context().ordinal() as i64;

        let array = PrimitiveArray::from_iter(0u32..5).into_array();
        let mut device_array = array.export_device_array(&mut ctx).await?;
        assert_device_metadata(&device_array, expected_device_id, true);
        assert!(!device_array.array.private_data.is_null());
        unsafe { release_exported_array(&raw mut device_array.array) };

        let array = NullArray::new(5).into_array();
        let mut device_array = array.export_device_array(&mut ctx).await?;
        assert_device_metadata(&device_array, expected_device_id, false);
        assert!(device_array.array.private_data.is_null());
        unsafe { release_exported_array(&raw mut device_array.array) };

        Ok(())
    }

    // Check sliced arrays preserve the expected Arrow length/offset metadata.
    #[crate::test]
    async fn test_export_sliced_arrays() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let primitive = PrimitiveArray::from_iter(0u32..10)
            .into_array()
            .slice(3..8)?;
        let mut device_array = primitive.export_device_array(&mut ctx).await?;
        assert_eq!(device_array.array.length, 5);
        assert_eq!(device_array.array.offset, 0);
        assert_eq!(device_array.array.n_buffers, 2);
        unsafe { release_exported_array(&raw mut device_array.array) };

        let bools = BoolArray::from_iter([true, false, true, true, false, false, true, false])
            .into_array()
            .slice(1..6)?;
        let mut device_array = bools.export_device_array(&mut ctx).await?;
        assert_eq!(device_array.array.length, 5);
        assert_eq!(device_array.array.offset, 1);
        assert_eq!(device_array.array.n_buffers, 2);
        unsafe { release_exported_array(&raw mut device_array.array) };

        Ok(())
    }

    #[crate::test]
    async fn test_export_sliced_varbinview_arrays() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let utf8 = VarBinViewArray::from_iter_str([
            "skip this out-of-line value before the slice",
            "hello",
            "こんにちは",
            "this out-of-line value remains in the slice",
        ])
        .into_array()
        .slice(1..4)?;
        let mut exported = utf8.export_device_array_with_schema(&mut ctx).await?;
        let field = Field::try_from(&exported.schema)?;
        assert_eq!(field, Field::new("", DataType::Utf8View, false));
        assert_varbinview_shape(&exported.array.array, 3, 0)?;
        assert_eq!(exported.array.device_type, ARROW_DEVICE_CUDA);
        unsafe { release_exported_array(&raw mut exported.array.array) };

        let binary = VarBinViewArray::from_iter_nullable_bin([
            Some(b"skip this out-of-line value before the slice" as &[u8]),
            None,
            Some(b"\x00\xff"),
            Some(b"this out-of-line binary value remains in the slice"),
        ])
        .into_array()
        .slice(1..4)?;
        let mut exported = binary.export_device_array_with_schema(&mut ctx).await?;
        let field = Field::try_from(&exported.schema)?;
        assert_eq!(field, Field::new("", DataType::BinaryView, true));
        assert_varbinview_shape(&exported.array.array, 3, 1)?;
        assert_eq!(exported.array.device_type, ARROW_DEVICE_CUDA);
        unsafe { release_exported_array(&raw mut exported.array.array) };

        Ok(())
    }

    // Check nullable primitives export Arrow null bitmaps on device.
    #[crate::test]
    async fn test_export_nullable_primitive() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let mut primitive = assert_nullable_export(
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array(),
            2,
            1,
            &mut ctx,
        )
        .await?;
        unsafe { release_exported_array(&raw mut primitive.array) };

        let mut all_null_primitive = assert_nullable_export(
            PrimitiveArray::from_option_iter([None::<i32>, None]).into_array(),
            2,
            2,
            &mut ctx,
        )
        .await?;
        unsafe { release_exported_array(&raw mut all_null_primitive.array) };

        Ok(())
    }

    // Check nullable bool exports preserve Arrow offset metadata.
    #[crate::test]
    async fn test_export_nullable_bool() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let mut bools = assert_nullable_export(
            BoolArray::from_iter([Some(true), None, Some(false), Some(true)])
                .into_array()
                .slice(1..4)?,
            2,
            1,
            &mut ctx,
        )
        .await?;
        assert_eq!(bools.array.offset, 1);
        unsafe { release_exported_array(&raw mut bools.array) };

        Ok(())
    }

    // Check synthesized all-null bool validity is large enough for Arrow offset-based reads.
    #[crate::test]
    async fn test_export_all_null_sliced_bool_validity_covers_arrow_offset() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let mut bools = assert_nullable_export(
            BoolArray::from_iter([None; 10]).into_array().slice(7..9)?,
            2,
            2,
            &mut ctx,
        )
        .await?;
        assert_eq!(bools.array.offset, 7);

        let private_data = unsafe { &*bools.array.private_data.cast::<PrivateData>() };
        let null_buffer = private_data.buffers[0]
            .as_ref()
            .vortex_expect("null buffer should be present");
        assert_eq!(null_buffer.len(), 2);

        unsafe { release_exported_array(&raw mut bools.array) };

        Ok(())
    }

    // Check nullable decimal exports include Arrow null bitmaps.
    #[crate::test]
    async fn test_export_nullable_decimal() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let mut decimal = assert_nullable_export(
            DecimalArray::from_option_iter(
                [Some(100i32), None, Some(300)],
                DecimalDType::new(10, 2),
            )
            .into_array(),
            2,
            1,
            &mut ctx,
        )
        .await?;
        unsafe { release_exported_array(&raw mut decimal.array) };

        Ok(())
    }

    // Check nullable temporal exports include Arrow null bitmaps.
    #[crate::test]
    async fn test_export_nullable_temporal() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let mut temporal = assert_nullable_export(
            TemporalArray::new_date(
                PrimitiveArray::from_option_iter([Some(100i32), None, Some(300)]).into_array(),
                TimeUnit::Days,
            )
            .into_array(),
            2,
            1,
            &mut ctx,
        )
        .await?;
        unsafe { release_exported_array(&raw mut temporal.array) };

        Ok(())
    }

    // Check nullable string-view exports include Arrow null bitmaps.
    #[crate::test]
    async fn test_export_nullable_varbinview() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let mut varbinview = assert_nullable_export(
            VarBinViewArray::from_iter_nullable_str([
                Some("one"),
                None,
                Some("this is a longer string for out-of-line storage"),
            ])
            .into_array(),
            4,
            1,
            &mut ctx,
        )
        .await?;
        unsafe { release_exported_array(&raw mut varbinview.array) };

        Ok(())
    }

    // Check nullable struct exports include Arrow null bitmaps.
    #[crate::test]
    async fn test_export_nullable_struct() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let mut struct_array = assert_nullable_export(
            StructArray::try_new(
                FieldNames::from_iter(["a"]),
                vec![PrimitiveArray::from_iter(0u32..3).into_array()],
                3,
                Validity::from_iter([true, false, true]),
            )?
            .into_array(),
            1,
            1,
            &mut ctx,
        )
        .await?;
        unsafe { release_exported_array(&raw mut struct_array.array) };

        Ok(())
    }

    // Check nested struct children expose cuDF-compatible Arrow Device layouts.
    #[crate::test]
    async fn test_export_nested_struct_child_layout() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let mut device_array = nested_struct_array().export_device_array(&mut ctx).await?;

        assert_eq!(device_array.array.n_buffers, 1);
        assert_eq!(device_array.array.n_children, 2);
        let children = unsafe {
            std::slice::from_raw_parts(
                device_array.array.children,
                usize::try_from(device_array.array.n_children)?,
            )
        };

        let primitive_child = unsafe { &*children[0] };
        assert_eq!(primitive_child.n_buffers, 2);
        assert_eq!(primitive_child.n_children, 0);

        let nested_child = unsafe { &*children[1] };
        assert_eq!(nested_child.n_buffers, 1);
        assert_eq!(nested_child.n_children, 2);
        let nested_children = unsafe {
            std::slice::from_raw_parts(
                nested_child.children,
                usize::try_from(nested_child.n_children)?,
            )
        };

        let nested_primitive_child = unsafe { &*nested_children[0] };
        assert_eq!(nested_primitive_child.n_buffers, 2);
        assert_eq!(nested_primitive_child.n_children, 0);

        let string_child = unsafe { &*nested_children[1] };
        assert_eq!(string_child.n_buffers, 4);
        assert_eq!(string_child.n_children, 0);
        let string_buffers = unsafe {
            std::slice::from_raw_parts(
                string_child.buffers,
                usize::try_from(string_child.n_buffers)?,
            )
        };
        assert!(string_buffers[0].is_null());
        assert!(!string_buffers[1].is_null());
        assert!(!string_buffers[2].is_null());
        assert!(!string_buffers[3].is_null());

        unsafe { release_exported_array(&raw mut device_array.array) };
        Ok(())
    }

    // Check parent release recursively releases children and is safe to repeat.
    #[crate::test]
    async fn test_release_is_idempotent_and_releases_children() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let mut device_array = nested_struct_array().export_device_array(&mut ctx).await?;

        assert!(device_array.array.release.is_some());
        assert!(!device_array.array.private_data.is_null());
        assert_eq!(device_array.array.n_children, 2);
        let children = unsafe {
            std::slice::from_raw_parts(
                device_array.array.children,
                usize::try_from(device_array.array.n_children)?,
            )
        };
        assert!(children.iter().all(|child| !child.is_null()));
        assert!(
            children
                .iter()
                .all(|child| unsafe { (**child).release.is_some() })
        );

        let nested_child = children[1];
        assert_eq!(unsafe { (*nested_child).n_children }, 2);
        let nested_children = unsafe {
            std::slice::from_raw_parts(
                (*nested_child).children,
                usize::try_from((*nested_child).n_children)?,
            )
        };
        assert!(nested_children.iter().all(|child| !child.is_null()));
        assert!(
            nested_children
                .iter()
                .all(|child| unsafe { (**child).release.is_some() })
        );

        unsafe { release_exported_array(&raw mut device_array.array) };
        assert!(device_array.array.release.is_none());
        assert!(device_array.array.private_data.is_null());

        unsafe { release_exported_array(&raw mut device_array.array) };
        assert!(device_array.array.release.is_none());

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

    // Check nested struct device exports carry the matching Arrow schema.
    #[crate::test]
    async fn test_export_nested_struct_with_schema() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let mut exported = nested_struct_array()
            .export_device_array_with_schema(&mut ctx)
            .await?;

        let schema = Schema::try_from(&exported.schema)?;
        assert_eq!(
            schema,
            Schema::new(vec![
                Field::new("a", DataType::UInt32, false),
                Field::new(
                    "nested",
                    DataType::Struct(Fields::from(vec![
                        Field::new("b", DataType::Int64, false),
                        Field::new("c", DataType::Utf8View, false),
                    ])),
                    false,
                ),
            ])
        );
        assert_eq!(exported.array.array.length, 5);
        assert_eq!(exported.array.array.n_children, 2);
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

        let japanese = "こんにちは";
        let long_emoji = "🦀 and 🚀 make this string out-of-line";
        let array = VarBinViewArray::from_iter_str(["", "hello", "é", "🦀", japanese, long_emoji])
            .into_array();
        let mut exported = array.export_device_array_with_schema(&mut ctx).await?;

        let field = Field::try_from(&exported.schema)?;
        assert_eq!(field, Field::new("", DataType::Utf8View, false));
        assert_varbinview_layout(
            &exported.array.array,
            6,
            0,
            &[japanese.len() + long_emoji.len()],
        )?;
        assert_eq!(exported.array.device_type, ARROW_DEVICE_CUDA);

        unsafe { release_exported_array(&raw mut exported.array.array) };
        Ok(())
    }
}
