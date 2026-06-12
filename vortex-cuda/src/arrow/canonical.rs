// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem;
use std::mem::size_of;
use std::ptr;
use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::CudaSlice;
use cudarc::driver::DeviceRepr;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use futures::future::BoxFuture;
use futures::future::join;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::arrays::DecimalArray;
use vortex::array::arrays::Dict;
use vortex::array::arrays::DictArray;
use vortex::array::arrays::FixedSizeList;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::List;
use vortex::array::arrays::ListArray;
use vortex::array::arrays::ListView;
use vortex::array::arrays::ListViewArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::Struct;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::arrays::bool::BoolDataParts;
use vortex::array::arrays::decimal::DecimalDataParts;
use vortex::array::arrays::dict::DictOwnedExt;
use vortex::array::arrays::extension::ExtensionArrayExt;
use vortex::array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex::array::arrays::fixed_size_list::FixedSizeListDataParts;
use vortex::array::arrays::list::ListDataParts;
use vortex::array::arrays::listview::ListViewArrayExt;
use vortex::array::arrays::listview::list_from_list_view;
use vortex::array::arrays::primitive::PrimitiveDataParts;
use vortex::array::arrays::struct_::StructDataParts;
use vortex::array::arrays::varbinview::VarBinViewDataParts;
use vortex::array::buffer::BufferHandle;
use vortex::array::builtins::ArrayBuiltins;
use vortex::array::match_each_decimal_value_type;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::dtype::DType;
use vortex::dtype::DecimalType;
use vortex::dtype::NativeDecimalType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::dtype::i256;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex::extension::datetime::AnyTemporal;
use vortex::mask::Mask;

use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::CudaExecutionCtx;
use crate::arrow::ARROW_DEVICE_CUDA;
use crate::arrow::ArrowArray;
use crate::arrow::ArrowDeviceArray;
use crate::arrow::ArrowDeviceArrayWithSchema;
use crate::arrow::ExportDeviceArray;
use crate::arrow::PrivateData;
use crate::arrow::SyncEvent;
use crate::arrow::arrow_device_export_dictionary_codes_dtype;
use crate::arrow::arrow_schema_for_array;
use crate::arrow::cuda_decimal_value_type;
use crate::arrow::list_view::export_device_list_view;
use crate::cub::exclusive_sum_i32;
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
        let (arrow_array, sync_event) = export_array(array, ctx).await?;

        Ok(ArrowDeviceArray {
            array: arrow_array,
            device_id: ctx.stream().context().ordinal() as i64,
            device_type: ARROW_DEVICE_CUDA,
            sync_event,
            reserved: Default::default(),
        })
    }

    async fn export_device_array_with_schema(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<ArrowDeviceArrayWithSchema> {
        let array = rebuild_array_for_export_schema(array, ctx.execution_ctx())?;
        let schema = arrow_schema_for_array(&array, ctx)?;
        let array = self.export_device_array(array, ctx).await?;
        Ok(ArrowDeviceArrayWithSchema { schema, array })
    }
}

/// Rebuild arrays whose exported layout differs from their original layout.
fn rebuild_array_for_export_schema(
    array: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let array = match array.try_downcast::<Dict>() {
        Ok(dict) => {
            let parts = dict.into_parts();
            let values = rebuild_array_for_export_schema(parts.values, ctx)?;
            return Ok(DictArray::try_new(parts.codes, values)?.into_array());
        }
        Err(array) => array,
    };
    let array = match array.try_downcast::<Struct>() {
        Ok(struct_array) => {
            let len = struct_array.len();
            let StructDataParts {
                struct_fields,
                fields,
                validity,
            } = struct_array.into_data_parts();
            let fields = fields
                .iter()
                .map(|field| rebuild_array_for_export_schema(field.clone(), ctx))
                .collect::<VortexResult<Vec<_>>>()?;
            return Ok(
                StructArray::try_new(struct_fields.names().clone(), fields, len, validity)?
                    .into_array(),
            );
        }
        Err(array) => array,
    };
    let array = match array.try_downcast::<List>() {
        Ok(list) => {
            let ListDataParts {
                elements,
                offsets,
                validity,
                ..
            } = list.into_data_parts();
            let elements = rebuild_array_for_export_schema(elements, ctx)?;
            return Ok(ListArray::try_new(elements, offsets, validity)?.into_array());
        }
        Err(array) => array,
    };
    let array = match array.try_downcast::<FixedSizeList>() {
        Ok(fixed_size_list) => {
            let len = fixed_size_list.len();
            let list_size = fixed_size_list.list_size();
            let FixedSizeListDataParts {
                elements, validity, ..
            } = fixed_size_list.into_data_parts();
            let elements = rebuild_array_for_export_schema(elements, ctx)?;
            return Ok(
                FixedSizeListArray::try_new(elements, list_size, validity, len)?.into_array(),
            );
        }
        Err(array) => array,
    };
    let array = match array.try_downcast::<ListView>() {
        Ok(listview)
            if listview.as_ref().is_host() && listview.elements().as_opt::<Dict>().is_some() =>
        {
            return rebuild_array_for_export_schema(
                list_from_list_view(listview, ctx)?.into_array(),
                ctx,
            );
        }
        Ok(listview) => return Ok(listview.into_array()),
        Err(array) => array,
    };

    Ok(array)
}

/// Export arrays whose Arrow layout depends on their concrete children before CUDA
/// canonicalization can erase that structure (for example nested dictionaries).
/// All other arrays are executed on CUDA, then exported directly from the canonical result.
fn export_array(
    array: ArrayRef,
    ctx: &mut CudaExecutionCtx,
) -> BoxFuture<'_, VortexResult<(ArrowArray, SyncEvent)>> {
    Box::pin(async {
        let array = match array.try_downcast::<Dict>() {
            Ok(dict) => return export_dict(dict, ctx).await,
            Err(array) => array,
        };
        let array = match array.try_downcast::<Struct>() {
            Ok(struct_array) => return export_struct(struct_array, ctx).await,
            Err(array) => array,
        };
        let array = match array.try_downcast::<List>() {
            Ok(list) => {
                return export_list(list, ListChildExport::PreserveConcreteLayout, ctx).await;
            }
            Err(array) => array,
        };
        let array = match array.try_downcast::<FixedSizeList>() {
            Ok(fixed_size_list) => return export_fixed_size_list(fixed_size_list, ctx).await,
            Err(array) => array,
        };
        let array = match array.try_downcast::<ListView>() {
            Ok(list_view) => return export_list_view(list_view, ctx).await,
            Err(array) => array,
        };

        let cuda_array = array.execute_cuda(ctx).await?;
        export_canonical(cuda_array, ctx).await
    })
}

/// Export a canonical CUDA array using the Arrow C Device layout for its logical type.
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
            Canonical::Decimal(decimal) => export_decimal(decimal, ctx).await,
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
                let BoolDataParts { bits, meta } = bool_array.into_data().into_parts(len);

                let (validity_buffer, null_count) =
                    export_arrow_validity_buffer(validity, len, meta.offset(), ctx).await?;

                let bits = ctx.ensure_on_device(bits).await?;
                export_fixed_size(
                    bits,
                    meta.len(),
                    meta.offset(),
                    validity_buffer,
                    null_count,
                    ctx,
                )
            }
            Canonical::List(listview) => export_list_view(listview, ctx).await,
            Canonical::FixedSizeList(fixed_size_list) => {
                export_fixed_size_list(fixed_size_list, ctx).await
            }
            Canonical::VarBinView(varbinview) => {
                if matches!(varbinview.dtype(), DType::Binary(_)) {
                    return export_binary(varbinview, ctx).await;
                }

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

/// Export a Vortex dictionary array as an Arrow Device dictionary array.
///
/// Owns the codes buffers and recursively exported dictionary values.
async fn export_dict(
    array: DictArray,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<(ArrowArray, SyncEvent)> {
    let len = array.len();
    let parts = array.into_parts();
    let PrimitiveDataParts {
        buffer, validity, ..
    } = export_dictionary_codes(parts.codes, ctx).await?;
    let (validity_buffer, null_count) = export_arrow_validity_buffer(validity, len, 0, ctx).await?;
    let codes_buffer = ctx.ensure_on_device(buffer).await?;
    let (dictionary, _) = export_array(parts.values, ctx).await?;

    let mut private_data = PrivateData::new_with_dictionary(
        vec![validity_buffer, Some(codes_buffer)],
        vec![],
        Some(dictionary),
        ctx,
    )?;
    let sync_event = private_data.sync_event();
    let dictionary = private_data.dictionary;

    let arrow_array = ArrowArray {
        length: len as i64,
        null_count,
        offset: 0,
        n_buffers: 2,
        buffers: private_data.buffer_ptrs.as_mut_ptr(),
        n_children: 0,
        children: ptr::null_mut(),
        release: Some(release_array),
        dictionary,
        private_data: Box::into_raw(private_data).cast(),
    };

    Ok((arrow_array, sync_event))
}

async fn export_dictionary_codes(
    codes: ArrayRef,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<PrimitiveDataParts> {
    let target_dtype = arrow_device_export_dictionary_codes_dtype(codes.dtype())?;
    let target_ptype = target_dtype.as_ptype();
    let codes = if codes.dtype() == &target_dtype {
        codes
    } else {
        codes.cast(target_dtype)?
    }
    .execute_cuda(ctx)
    .await?;
    let Canonical::Primitive(codes) = codes else {
        vortex_bail!("dictionary codes must be primitive, got {}", codes.dtype());
    };

    let parts = codes.into_data_parts();
    vortex_ensure!(
        parts.ptype == target_ptype,
        "dictionary codes export produced {}",
        parts.ptype
    );
    Ok(parts)
}

/// Exports decimals with value buffers cast to Arrow's Decimal32/64/128/256 layout.
///
/// Decimal values are already decoded; this only adapts the physical buffer width. Storage-to-Arrow
/// narrowing is rejected instead of checked on-device to avoid a device-to-host synchronization
/// point.
async fn export_decimal(
    decimal: DecimalArray,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<(ArrowArray, SyncEvent)> {
    let len = decimal.len();
    let DecimalDataParts {
        decimal_dtype,
        values,
        values_type,
        validity,
    } = decimal.into_data_parts();

    let (validity_buffer, null_count) = export_arrow_validity_buffer(validity, len, 0, ctx).await?;
    let target_type = cuda_decimal_value_type(decimal_dtype);
    let values = export_decimal_values(values, values_type, target_type, len, ctx).await?;

    export_fixed_size(values, len, 0, validity_buffer, null_count, ctx)
}

/// Ensure the values buffer is on-device and has the Arrow-required decimal width.
///
/// Storage wider than the precision-implied Arrow width is rejected. Callers that hit this
/// should narrow the storage via a decimal cast before exporting.
async fn export_decimal_values(
    values: BufferHandle,
    values_type: DecimalType,
    target_type: DecimalType,
    len: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<BufferHandle> {
    if values_type.byte_width() > target_type.byte_width() {
        vortex_bail!(
            "cannot export decimal values from {values_type} storage to Arrow {target_type}: narrowing would require a device-to-host overflow check",
        );
    }
    let values = ctx.ensure_on_device(values).await?;
    if values_type == target_type {
        return Ok(values);
    }

    match_each_decimal_value_type!(values_type, |S| {
        export_decimal_values_from::<S>(values, target_type, len, ctx).await
    })
}

/// Dispatch from a concrete storage type `S` to the Arrow-required output width.
async fn export_decimal_values_from<S>(
    values: BufferHandle,
    target_type: DecimalType,
    len: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<BufferHandle>
where
    S: NativeDecimalType + DeviceRepr,
{
    match target_type {
        DecimalType::I32 => decimal_cast::<S, i32>(values, len, ctx).await,
        DecimalType::I64 => decimal_cast::<S, i64>(values, len, ctx).await,
        DecimalType::I128 => decimal_cast::<S, i128>(values, len, ctx).await,
        DecimalType::I256 => decimal_cast::<S, i256>(values, len, ctx).await,
        target_type => {
            vortex_bail!("cannot export DecimalArray as Arrow decimal value type {target_type}")
        }
    }
}

/// Launches the CUDA kernel that casts from Vortex storage type `S` to Arrow output type `D`.
///
/// The caller must ensure this is not a narrowing cast.
async fn decimal_cast<S, D>(
    values: BufferHandle,
    len: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<BufferHandle>
where
    S: NativeDecimalType + DeviceRepr,
    D: NativeDecimalType + DeviceRepr,
{
    if len == 0 {
        return ctx
            .ensure_on_device(BufferHandle::new_host(
                Buffer::<D>::empty().into_byte_buffer(),
            ))
            .await;
    }

    let output_buffer = ctx.device_alloc::<D>(len)?;
    let output_device = CudaDeviceBuffer::new(output_buffer);

    let values_view = values.cuda_view::<S>()?;
    let output_view = output_device.as_view::<D>();
    let len_u64 = len as u64;
    let cuda_function = ctx.load_function_with_suffixes(
        "decimal_cast",
        &[&S::DECIMAL_TYPE.to_string(), &D::DECIMAL_TYPE.to_string()],
    )?;

    ctx.launch_kernel(&cuda_function, len, |args| {
        args.arg(&values_view).arg(&output_view).arg(&len_u64);
    })?;

    Ok(BufferHandle::new_device(Arc::new(output_device)))
}

/// Export Vortex binary views as an Arrow Device array with standard `Binary` layout.
async fn export_binary(
    varbinview: VarBinViewArray,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<(ArrowArray, SyncEvent)> {
    let len = varbinview.len();
    let VarBinViewDataParts {
        views,
        buffers: data_buffers,
        validity,
        ..
    } = varbinview.into_data_parts();

    let (validity_buffer, null_count) = export_arrow_validity_buffer(validity, len, 0, ctx).await?;
    let views = ctx.ensure_on_device(views).await?;
    let (offsets, values) =
        export_binary_buffers(&views, &data_buffers, validity_buffer.as_ref(), len, ctx).await?;

    let buffers = vec![validity_buffer, Some(offsets), Some(values)];

    let mut private_data = PrivateData::new(buffers, vec![], ctx)?;
    let sync_event = private_data.sync_event();
    let arrow_array = ArrowArray {
        length: len as i64,
        null_count,
        offset: 0,
        // Arrow Binary layout: optional null bitmap, i32 offsets, contiguous bytes.
        n_buffers: 3,
        buffers: private_data.buffer_ptrs.as_mut_ptr(),
        n_children: 0,
        children: ptr::null_mut(),
        release: Some(release_array),
        dictionary: ptr::null_mut(),
        private_data: Box::into_raw(private_data).cast(),
    };

    Ok((arrow_array, sync_event))
}

/// Build Arrow Binary offsets and values from VarBinView buffers on the active CUDA stream.
async fn export_binary_buffers(
    views: &BufferHandle,
    data_buffers: &[BufferHandle],
    validity: Option<&BufferHandle>,
    len: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<(BufferHandle, BufferHandle)> {
    let mut device_data_buffers = Vec::with_capacity(data_buffers.len());
    for buffer in data_buffers {
        device_data_buffers.push(ctx.ensure_on_device(buffer.clone()).await?);
    }

    let mut ptr_values = Vec::with_capacity(device_data_buffers.len());
    let mut len_values = Vec::with_capacity(device_data_buffers.len());
    for buffer in &device_data_buffers {
        ptr_values.push(buffer.cuda_device_ptr()?);
        len_values.push(u64::try_from(buffer.len())?);
    }
    if device_data_buffers.is_empty() {
        // Kernels never dereference these when data_buffer_count is zero, but the arguments
        // still need a real device allocation.
        ptr_values.push(0);
        len_values.push(0);
    }
    let data_buffer_ptrs = device_buffer_from(ptr_values, ctx).await?;
    let data_buffer_lens = device_buffer_from(len_values, ctx).await?;
    let status = device_buffer_from(vec![0u32], ctx).await?;

    let scan_input = init_binary_scan(
        views,
        validity,
        &data_buffer_lens,
        device_data_buffers.len(),
        &status,
        len,
        ctx,
    )?;
    let output_offsets = BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(
        exclusive_sum_i32(&scan_input, len + 1, ctx)?,
    )));
    validate_binary_offsets(&output_offsets, len, &status, ctx)?;

    // One status read covers init_scan and offset validation. Both must pass before gather may
    // dereference view payloads through the scanned offsets. Enqueue both copies up front so the
    // readbacks share one stream round-trip; await both so an error cannot drop a copy mid-flight.
    let status_copy = status.try_to_host()?;
    let total_copy = output_offsets
        .slice_typed::<i32>(len..len + 1)
        .try_to_host()?;
    let (status_value, total_value) = join(status_copy, total_copy).await;

    check_binary_status(Buffer::<u32>::from_byte_buffer(status_value?)[0])?;
    let total_bytes = usize::try_from(Buffer::<i32>::from_byte_buffer(total_value?)[0])?;
    let output_values = gather_binary_values(
        views,
        &data_buffer_ptrs,
        &output_offsets,
        total_bytes,
        len,
        ctx,
    )?;

    Ok((output_offsets, output_values))
}

/// Copy a small host vector to a device-resident buffer.
async fn device_buffer_from<T>(
    values: Vec<T>,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<BufferHandle>
where
    Buffer<T>: From<Vec<T>>,
{
    ctx.ensure_on_device(BufferHandle::new_host(
        Buffer::from(values).into_byte_buffer(),
    ))
    .await
}

fn check_binary_status(status: u32) -> VortexResult<()> {
    match status {
        0 => Ok(()),
        1 => vortex_bail!(
            "cannot export BinaryView as Arrow Binary: a view references an invalid data buffer"
        ),
        2 => vortex_bail!(
            "cannot export BinaryView as Arrow Binary: offsets exceed i32 range required by Arrow Binary"
        ),
        status => vortex_bail!("unexpected Arrow Binary export status {status}"),
    }
}

fn init_binary_scan(
    views: &BufferHandle,
    validity: Option<&BufferHandle>,
    data_buffer_lens: &BufferHandle,
    data_buffer_count: usize,
    status: &BufferHandle,
    len: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<CudaSlice<i32>> {
    let scan_len = len + 1;
    let views_view = views.cuda_view::<u8>()?;
    // A null pointer signals an all-valid array to the kernel.
    let validity_ptr = validity
        .map(|v| v.cuda_device_ptr())
        .transpose()?
        .unwrap_or(0);
    let lens_view = data_buffer_lens.cuda_view::<u64>()?;
    let status_view = status.cuda_view::<u32>()?;
    let data_buffer_count_u64 = data_buffer_count as u64;
    let len_u64 = len as u64;
    let scan_input = ctx.device_alloc::<i32>(scan_len)?;
    let kernel = ctx.load_function_with_suffixes("arrow_binary", &["init_scan"])?;

    ctx.launch_kernel(&kernel, scan_len, |args| {
        args.arg(&views_view)
            .arg(&validity_ptr)
            .arg(&lens_view)
            .arg(&scan_input)
            .arg(&status_view)
            .arg(&data_buffer_count_u64)
            .arg(&len_u64);
    })?;

    Ok(scan_input)
}

/// Flag scanned offsets that overflowed i32. A negative offset is proof of overflow because
/// init_scan caps every row size at `i32::MAX`, so the first overflowing prefix sum always wraps
/// into the negative range.
fn validate_binary_offsets(
    offsets: &BufferHandle,
    len: usize,
    status: &BufferHandle,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<()> {
    let scan_len = len + 1;
    let offsets_view = offsets.cuda_view::<i32>()?;
    let status_view = status.cuda_view::<u32>()?;
    let scan_len_u64 = scan_len as u64;
    let kernel = ctx.load_function_with_suffixes("arrow_binary", &["validate_offsets"])?;

    ctx.launch_kernel(&kernel, scan_len, |args| {
        args.arg(&offsets_view).arg(&status_view).arg(&scan_len_u64);
    })
}

fn gather_binary_values(
    views: &BufferHandle,
    data_buffer_ptrs: &BufferHandle,
    offsets: &BufferHandle,
    total_bytes: usize,
    len: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<BufferHandle> {
    let output_values = ctx.device_alloc::<u8>(total_bytes.max(1))?;

    if total_bytes != 0 {
        let views_view = views.cuda_view::<u8>()?;
        let ptrs_view = data_buffer_ptrs.cuda_view::<u64>()?;
        let offsets_view = offsets.cuda_view::<i32>()?;
        let len_u64 = len as u64;
        let total_bytes_u64 = total_bytes as u64;
        let kernel = ctx.load_function_with_suffixes("arrow_binary", &["gather"])?;

        ctx.launch_kernel(&kernel, total_bytes, |args| {
            args.arg(&views_view)
                .arg(&ptrs_view)
                .arg(&offsets_view)
                .arg(&output_values)
                .arg(&len_u64)
                .arg(&total_bytes_u64);
        })?;
    }

    Ok(
        BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(output_values)))
            .slice(0..total_bytes),
    )
}

/// Export Vortex validity as an Arrow Device validity byte buffer.
///
/// Returns `None` for the buffer when Arrow can omit validity because all rows are valid.
///
/// Returned buffers use zeroed 4-byte padding so cuDF's word-sized mask reads stay in bounds.
/// Bits at positions `>= len + arrow_offset` within the final data byte are unspecified, as
/// Arrow permits.
pub(super) async fn export_arrow_validity_buffer(
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
        Mask::AllFalse(_) => device_zeroed_byte_buffer(validity_bytes, ctx)?,
        Mask::Values(values) => {
            // Shrinking the offset below 8 bounds the host-to-device copy to the slice's bytes
            // instead of the whole backing bitmap.
            let bits = values.bit_buffer().clone().shrink_offset();
            if arrow_offset == 0 && bits.offset() == 0 {
                // Fast path: the Vortex bitmap already matches Arrow's byte-addressed layout.
                let (_, _, buffer) = bits.into_inner();
                ctx.ensure_on_device(BufferHandle::new_host(buffer)).await?
            } else {
                // Slow path: bit offsets cannot be represented by the Arrow buffer pointer.
                // Repack on the GPU so compact/sliced exports keep Arrow offset semantics.
                let (input_offset, _, input_buffer) = bits.into_inner();
                let input_buffer = ctx
                    .ensure_on_device(BufferHandle::new_host(input_buffer))
                    .await?;
                repack_arrow_validity_buffer(&input_buffer, input_offset, len, arrow_offset, ctx)?
            }
        }
    };

    Ok((Some(validity_buffer), null_count))
}

/// Allocate a zeroed device buffer with cuDF-safe padding for Arrow validity masks.
fn device_zeroed_byte_buffer(
    byte_len: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<BufferHandle> {
    let allocation_len = byte_len.next_multiple_of(size_of::<u32>()).max(1);
    let mut buffer = ctx.device_alloc::<u8>(allocation_len)?;
    ctx.stream()
        .memset_zeros(&mut buffer)
        .map_err(|err| vortex_err!("Failed to zero Arrow validity buffer: {err}"))?;
    Ok(BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(buffer))).slice(0..byte_len))
}

/// Repack a validity bitmap into Arrow layout without copying bitmap bits back to the CPU.
///
/// Vortex bitmaps may start at any bit offset. Arrow exposes only a byte-addressed validity buffer
/// plus an array offset, so sliced compact exports need a GPU rewrite when either side has a
/// bit-level offset. The kernel writes the output one 64-bit word at a time, funnel-shifting two
/// adjacent input words, so the allocation is padded to whole words (zeroed by the edge masks).
pub(super) fn repack_arrow_validity_buffer(
    input_buffer: &BufferHandle,
    input_offset: usize,
    len: usize,
    arrow_offset: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<BufferHandle> {
    let validity_bits = len
        .checked_add(arrow_offset)
        .ok_or_else(|| vortex_err!("Arrow validity bit length overflows usize"))?;
    let output_bytes = validity_bits.div_ceil(8);
    let output_words = validity_bits.div_ceil(u64::BITS as usize);

    // The kernel loads the input bitmap as 64-bit words.
    if !input_buffer
        .cuda_device_ptr()?
        .is_multiple_of(size_of::<u64>() as u64)
    {
        vortex_bail!("Arrow validity repack requires an 8-byte aligned device buffer");
    }

    let output = ctx.device_alloc::<u64>(output_words.max(1))?;
    let output_device = CudaDeviceBuffer::new(output);

    if output_words > 0 {
        let input_view = input_buffer.cuda_view::<u8>()?;
        let output_view = output_device.as_view::<u64>();
        let len = u64::try_from(len)?;
        let input_offset = u64::try_from(input_offset)?;
        let arrow_offset = u64::try_from(arrow_offset)?;
        let input_bytes = u64::try_from(input_buffer.len())?;

        let kernel = ctx.load_function_with_suffixes("arrow_validity", &["repack"])?;
        const REPACK_THREADS_PER_BLOCK: u32 = 256;
        let num_blocks = u32::try_from(output_words.div_ceil(REPACK_THREADS_PER_BLOCK as usize))?;
        let config = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (REPACK_THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        ctx.launch_kernel_config(&kernel, config, output_words, |args| {
            args.arg(&input_view)
                .arg(&output_view)
                .arg(&len)
                .arg(&input_offset)
                .arg(&arrow_offset)
                .arg(&input_bytes);
        })?;
    }

    Ok(BufferHandle::new_device(Arc::new(output_device)).slice(0..output_bytes))
}

/// Export a Vortex list-view as an Arrow Device array with `List` layout.
async fn export_list_view(
    listview: ListViewArray,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<(ArrowArray, SyncEvent)> {
    // cuDF imports standard Arrow `List`, while Vortex canonical lists are list-views.
    // Try the GPU path first; host list-views can fall back to a CPU rebuild.
    let is_host = listview.as_ref().is_host();
    let gpu_err = match export_device_list_view(listview.clone(), ctx).await {
        Ok(exported) => return Ok(exported),
        Err(err) => err,
    };

    // CPU rebuild requires host-resident buffers; device-resident arrays keep the GPU error.
    if !is_host {
        return Err(gpu_err);
    }

    // The CPU fallback packs ListView ranges into a contiguous List.
    export_list(
        list_from_list_view(listview, ctx.execution_ctx())?,
        ListChildExport::RebuiltListViewChild,
        ctx,
    )
    .await
}

/// Export a standard Vortex list as an Arrow Device array with `List` layout.
async fn export_list(
    array: ListArray,
    child_export: ListChildExport,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<(ArrowArray, SyncEvent)> {
    let len = array.len();
    let ListDataParts {
        elements,
        offsets,
        validity,
        ..
    } = array.into_data_parts();

    let (validity_buffer, null_count) = export_arrow_validity_buffer(validity, len, 0, ctx).await?;
    let offsets_buffer = export_arrow_list_offsets(offsets, ctx).await?;
    export_list_layout(
        elements,
        len,
        validity_buffer,
        null_count,
        offsets_buffer,
        child_export,
        ctx,
    )
    .await
}

#[derive(Clone, Copy)]
pub(super) enum ListChildExport {
    /// Preserve concrete child layouts, such as dictionaries,
    /// so exported data matches the schema.
    PreserveConcreteLayout,
    /// Canonicalize temporary encodings introduced by the host ListView
    /// rebuild, while still preserving rebuilt dictionary children.
    ///
    /// This is not equivalent to `export_array`: the take-based rebuild wraps
    /// children in transient encodings (for example `take` returns `Dict`
    /// arrays nested inside struct fields) that the pre-computed export schema
    /// does not include, so they must be canonicalized away before export.
    RebuiltListViewChild,
}

impl ListChildExport {
    async fn export(
        self,
        elements: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<ArrowArray> {
        let (elements_child, _) = match self {
            ListChildExport::PreserveConcreteLayout => export_array(elements, ctx).await?,
            ListChildExport::RebuiltListViewChild if elements.as_opt::<Dict>().is_some() => {
                export_array(elements, ctx).await?
            }
            ListChildExport::RebuiltListViewChild => {
                export_canonical(elements.execute_cuda(ctx).await?, ctx).await?
            }
        };
        Ok(elements_child)
    }
}

/// Build the shared Arrow Device `List` parent once offsets and validity are ready.
pub(super) async fn export_list_layout(
    elements: ArrayRef,
    len: usize,
    validity_buffer: Option<BufferHandle>,
    null_count: i64,
    offsets_buffer: BufferHandle,
    child_export: ListChildExport,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<(ArrowArray, SyncEvent)> {
    let elements_child = child_export.export(elements, ctx).await?;

    let mut private_data = PrivateData::new(
        vec![validity_buffer, Some(offsets_buffer)],
        vec![elements_child],
        ctx,
    )?;
    let sync_event = private_data.sync_event();

    let mut arrow_list = ArrowArray::empty();
    arrow_list.length = len as i64;
    arrow_list.null_count = null_count;
    arrow_list.n_buffers = 2;
    arrow_list.buffers = private_data.buffer_ptrs.as_mut_ptr();
    arrow_list.n_children = 1;
    arrow_list.children = private_data.children.as_mut_ptr();
    arrow_list.release = Some(release_array);
    arrow_list.private_data = Box::into_raw(private_data).cast();

    Ok((arrow_list, sync_event))
}

/// Export a Vortex fixed-size-list as an Arrow Device array with `List` layout.
///
/// cuDF's Arrow Device import accepts `List`/`LargeList` as cuDF `LIST`, but rejects
/// `FixedSizeList`, so emit equivalent standard Arrow `List` offsets.
async fn export_fixed_size_list(
    array: FixedSizeListArray,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<(ArrowArray, SyncEvent)> {
    let len = array.len();
    let list_size = array.list_size();
    let FixedSizeListDataParts {
        elements, validity, ..
    } = array.into_data_parts();

    let (validity_buffer, null_count) = export_arrow_validity_buffer(validity, len, 0, ctx).await?;
    let offsets_buffer = fixed_size_list_offsets(len, list_size, ctx).await?;

    export_list_layout(
        elements,
        len,
        validity_buffer,
        null_count,
        offsets_buffer,
        ListChildExport::PreserveConcreteLayout,
        ctx,
    )
    .await
}

async fn fixed_size_list_offsets(
    len: usize,
    list_size: u32,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<BufferHandle> {
    let list_size = i32::try_from(list_size).map_err(|_| {
        vortex_err!(
            "cannot export FixedSizeList with list size {list_size}: Arrow List offsets require i32"
        )
    })?;
    let offsets = (0..=i32::try_from(len)?)
        .map(|idx| {
            idx.checked_mul(list_size)
                .ok_or_else(|| vortex_err!("FixedSizeList Arrow List offsets exceed i32 range"))
        })
        .collect::<VortexResult<Vec<_>>>()?;

    ctx.ensure_on_device(BufferHandle::new_host(
        Buffer::from(offsets).into_byte_buffer(),
    ))
    .await
}

/// Return Arrow Device `List` offsets as an `i32` device buffer.
async fn export_arrow_list_offsets(
    offsets: ArrayRef,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<BufferHandle> {
    let offsets = if offsets.dtype().as_ptype() == PType::I32 {
        offsets
    } else {
        offsets.cast(DType::Primitive(PType::I32, Nullability::NonNullable))?
    };
    let offsets = offsets.execute_cuda(ctx).await?;
    let Canonical::Primitive(offsets) = offsets else {
        vortex_bail!("list offsets must be primitive, got {}", offsets.dtype());
    };

    let PrimitiveDataParts { ptype, buffer, .. } = offsets.into_data_parts();
    vortex_ensure!(
        ptype == PType::I32,
        "list offsets cast to i32 produced {ptype}"
    );

    ctx.ensure_on_device(buffer).await
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
        let (arrow_field, _) = export_array(field.clone(), ctx).await?;
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
    // SAFETY: this is only safe if we're dropping an ArrowArray that was
    // created from Rust code. This is necessary to ensure that the fields
    // inside the CudaPrivateData get dropped to free native/GPU memory.
    unsafe {
        if array.is_null() || (*array).release.is_none() {
            return;
        }

        let private_data_ptr = ptr::replace(&raw mut (*array).private_data, ptr::null_mut());

        if !private_data_ptr.is_null() {
            let mut private_data = Box::from_raw(private_data_ptr.cast::<PrivateData>());
            // Queue a non-blocking wait so CudaSlice::Drop frees after export work.
            let wait_result = private_data.cuda_stream.wait(&private_data.export_event);
            // Arrow release callbacks cannot return errors, so record wait-enqueue
            // failures for later CUDA operations to observe on this context.
            private_data.cuda_stream.context().record_err(wait_result);
            release_children(&mut private_data);
            release_dictionary(&mut private_data);
        }

        // update the release function to NULL to avoid any possibility of double-frees.
        (*array).release = None;
    }
}

/// Release all owned child arrays stored in private data.
unsafe fn release_children(private_data: &mut PrivateData) {
    unsafe {
        let children = mem::take(&mut private_data.children);
        for child in children {
            release_array_ptr(child);
        }
    }
}

/// Release the owned dictionary array, if this Arrow array has one.
unsafe fn release_dictionary(private_data: &mut PrivateData) {
    unsafe {
        let dictionary = ptr::replace(&raw mut private_data.dictionary, ptr::null_mut());
        release_array_ptr(dictionary);
    }
}

/// Release an ArrowArray pointer allocated for private data.
unsafe fn release_array_ptr(array: *mut ArrowArray) {
    unsafe {
        if !array.is_null() {
            if let Some(release) = (*array).release {
                release(array);
            }
            // Owned arrays are allocated with Box::into_raw in PrivateData, so the release callback
            // must also reclaim the ArrowArray allocation itself.
            drop(Box::from_raw(array));
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
    use vortex::array::arrays::DictArray;
    use vortex::array::arrays::FixedSizeListArray;
    use vortex::array::arrays::ListArray;
    use vortex::array::arrays::ListViewArray;
    use vortex::array::arrays::NullArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::arrays::TemporalArray;
    use vortex::array::arrays::VarBinViewArray;
    use vortex::array::arrays::primitive::PrimitiveArrayExt;
    use vortex::array::arrays::varbinview::BinaryView;
    use vortex::array::buffer::BufferHandle;
    use vortex::array::validity::Validity;
    use vortex::buffer::BitBuffer;
    use vortex::buffer::Buffer;
    use vortex::buffer::ByteBuffer;
    use vortex::dtype::DType;
    use vortex::dtype::DecimalDType;
    use vortex::dtype::FieldNames;
    use vortex::dtype::NativeDecimalType;
    use vortex::dtype::NativePType;
    use vortex::dtype::Nullability;
    use vortex::dtype::PType;
    use vortex::dtype::half::f16;
    use vortex::dtype::i256;
    use vortex::error::VortexExpect;
    use vortex::error::VortexResult;
    use vortex::error::vortex_bail;
    use vortex::extension::datetime::TimeUnit;
    use vortex::session::VortexSession;

    use crate::CudaExecutionCtx;
    use crate::arrow::ARROW_DEVICE_CUDA;
    use crate::arrow::ArrowArray;
    use crate::arrow::ArrowDeviceArray;
    use crate::arrow::DeviceArrayExt;
    use crate::arrow::PrivateData;
    use crate::arrow::canonical::export_arrow_validity_buffer;
    use crate::arrow::canonical::repack_arrow_validity_buffer;
    use crate::device_buffer::cuda_backing_allocation;
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

    async fn primitive_on_device<T: NativePType>(
        values: impl IntoIterator<Item = T>,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let primitive = PrimitiveArray::from_iter(values);
        let handle = ctx
            .ensure_on_device(primitive.buffer_handle().clone())
            .await?;
        Ok(
            PrimitiveArray::from_buffer_handle(handle, T::PTYPE, Validity::NonNullable)
                .into_array(),
        )
    }

    async fn primitive_i32_on_device(
        values: impl IntoIterator<Item = i32>,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        primitive_on_device(values, ctx).await
    }

    #[expect(clippy::cast_possible_truncation)]
    async fn integer_array_on_device(
        ptype: PType,
        values: &[i64],
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        match ptype {
            PType::U8 => primitive_on_device(values.iter().map(|&value| value as u8), ctx).await,
            PType::U16 => primitive_on_device(values.iter().map(|&value| value as u16), ctx).await,
            PType::U32 => primitive_on_device(values.iter().map(|&value| value as u32), ctx).await,
            PType::U64 => primitive_on_device(values.iter().map(|&value| value as u64), ctx).await,
            PType::I8 => primitive_on_device(values.iter().map(|&value| value as i8), ctx).await,
            PType::I16 => primitive_on_device(values.iter().map(|&value| value as i16), ctx).await,
            PType::I32 => primitive_on_device(values.iter().map(|&value| value as i32), ctx).await,
            PType::I64 => primitive_on_device(values.iter().copied(), ctx).await,
            ptype => vortex_bail!("test helper only supports integer PTypes, got {ptype}"),
        }
    }

    async fn nullable_primitive_i32_on_device(
        values: impl IntoIterator<Item = Option<i32>>,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let primitive = PrimitiveArray::from_option_iter(values);
        let handle = ctx
            .ensure_on_device(primitive.buffer_handle().clone())
            .await?;
        Ok(
            PrimitiveArray::from_buffer_handle(handle, PType::I32, primitive.validity()?)
                .into_array(),
        )
    }

    fn private_data_buffer_i32_values(
        array: &ArrowArray,
        buffer_idx: usize,
    ) -> VortexResult<Vec<i32>> {
        let private_data = unsafe { &*array.private_data.cast::<PrivateData>() };
        let buffer = private_data.buffers[buffer_idx]
            .as_ref()
            .vortex_expect("buffer should be present");
        Ok(Buffer::<i32>::from_byte_buffer(buffer.to_host_sync())
            .iter()
            .copied()
            .collect())
    }

    fn private_data_buffer_i16_values(
        array: &ArrowArray,
        buffer_idx: usize,
    ) -> VortexResult<Vec<i16>> {
        let private_data = unsafe { &*array.private_data.cast::<PrivateData>() };
        let buffer = private_data.buffers[buffer_idx]
            .as_ref()
            .vortex_expect("buffer should be present");
        Ok(Buffer::<i16>::from_byte_buffer(buffer.to_host_sync())
            .iter()
            .copied()
            .collect())
    }

    fn private_data_buffer_bytes(
        array: &ArrowArray,
        buffer_idx: usize,
    ) -> VortexResult<ByteBuffer> {
        let private_data = unsafe { &*array.private_data.cast::<PrivateData>() };
        let buffer = private_data.buffers[buffer_idx]
            .as_ref()
            .vortex_expect("buffer should be present");
        Ok(buffer.to_host_sync())
    }

    // Assert Arrow Binary export uses the standard null bitmap, i32 offsets, and values layout.
    fn assert_binary_layout(
        array: &ArrowArray,
        expected_len: i64,
        expected_null_count: i64,
        expected_offsets: &[i32],
        expected_values: &[u8],
    ) -> VortexResult<()> {
        assert_eq!(array.length, expected_len);
        assert_eq!(array.null_count, expected_null_count);
        assert_eq!(array.offset, 0);
        assert_eq!(array.n_buffers, 3);
        assert_eq!(array.n_children, 0);
        assert!(array.release.is_some());
        assert!(!array.private_data.is_null());

        let buffers =
            unsafe { std::slice::from_raw_parts(array.buffers, usize::try_from(array.n_buffers)?) };
        assert_eq!(buffers[0].is_null(), expected_null_count == 0);
        assert!(!buffers[1].is_null());
        assert!(!buffers[2].is_null());
        assert_eq!(private_data_buffer_i32_values(array, 1)?, expected_offsets);
        assert_eq!(
            private_data_buffer_bytes(array, 2)?.as_ref(),
            expected_values
        );

        Ok(())
    }

    fn assert_exported_decimal_values<T: NativeDecimalType>(
        value_buffer: &BufferHandle,
        expected: &[T],
    ) {
        let values = Buffer::<T>::from_byte_buffer(value_buffer.to_host_sync());
        assert_eq!(values.as_slice(), expected);
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
    async fn test_export_dictionary() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let out_of_line = "a dictionary value stored out-of-line";
        let array = DictArray::try_new(
            PrimitiveArray::from_option_iter([Some(0u8), None, Some(1), Some(0)]).into_array(),
            VarBinViewArray::from_iter_str(["alpha", out_of_line]).into_array(),
        )?
        .into_array();
        let mut exported = array.export_device_array_with_schema(&mut ctx).await?;

        let field = Field::try_from(&exported.schema)?;
        assert_eq!(
            field,
            Field::new(
                "",
                DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Utf8View)),
                true,
            )
        );
        assert_eq!(exported.array.array.length, 4);
        assert_eq!(exported.array.array.null_count, 1);
        assert_eq!(exported.array.array.n_buffers, 2);
        assert_eq!(exported.array.array.n_children, 0);
        assert!(!exported.array.array.dictionary.is_null());
        assert_eq!(exported.array.device_type, ARROW_DEVICE_CUDA);

        let private_data = unsafe { &*exported.array.array.private_data.cast::<PrivateData>() };
        assert_eq!(
            private_data.buffers[1]
                .as_ref()
                .vortex_expect("codes buffer should be present")
                .len(),
            4 * size_of::<i16>()
        );

        let dictionary = unsafe { &*exported.array.array.dictionary };
        assert_varbinview_layout(dictionary, 2, 0, &[out_of_line.len()])?;

        unsafe { release_exported_array(&raw mut exported.array.array) };
        Ok(())
    }

    #[crate::test]
    async fn test_export_struct_preserves_dictionary_child() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let dictionary = DictArray::try_new(
            PrimitiveArray::from_option_iter([Some(0u8), None, Some(1)]).into_array(),
            VarBinViewArray::from_iter_str(["alpha", "beta"]).into_array(),
        )?
        .into_array();
        let array = StructArray::new(
            FieldNames::from_iter(["dict"]),
            vec![dictionary],
            3,
            Validity::NonNullable,
        )
        .into_array();
        let mut exported = array.export_device_array_with_schema(&mut ctx).await?;

        let field = Field::try_from(&exported.schema)?;
        assert_eq!(
            field,
            Field::new(
                "",
                DataType::Struct(Fields::from(vec![Field::new(
                    "dict",
                    DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Utf8View)),
                    true,
                )])),
                false,
            )
        );
        assert_eq!(exported.array.array.n_children, 1);
        let children = unsafe { std::slice::from_raw_parts(exported.array.array.children, 1) };
        let dict_child = unsafe { &*children[0] };
        assert!(!dict_child.dictionary.is_null());
        assert_eq!(dict_child.null_count, 1);

        unsafe { release_exported_array(&raw mut exported.array.array) };
        Ok(())
    }

    #[crate::test]
    async fn test_export_dictionary_with_nullable_values() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let array = DictArray::try_new(
            PrimitiveArray::from_iter([0u8, 1, 0]).into_array(),
            PrimitiveArray::from_option_iter([Some(10i32), None]).into_array(),
        )?
        .into_array();
        let mut exported = array.export_device_array_with_schema(&mut ctx).await?;

        let field = Field::try_from(&exported.schema)?;
        assert_eq!(
            field,
            Field::new(
                "",
                DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Int32)),
                true,
            )
        );
        assert_eq!(exported.array.array.null_count, 0);
        assert_eq!(
            private_data_buffer_i16_values(&exported.array.array, 1)?,
            [0, 1, 0]
        );
        let dictionary = unsafe { &*exported.array.array.dictionary };
        assert_eq!(dictionary.null_count, 1);
        assert_eq!(private_data_buffer_i32_values(dictionary, 1)?.len(), 2);

        unsafe { release_exported_array(&raw mut exported.array.array) };
        Ok(())
    }

    async fn assert_exported_decimal<T: NativeDecimalType>(
        array: ArrayRef,
        expected_data_type: DataType,
        expected_values: Vec<T>,
    ) -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let mut exported = array.export_device_array_with_schema(&mut ctx).await?;

        let field = Field::try_from(&exported.schema)?;
        assert_eq!(field, Field::new("", expected_data_type, false));
        assert_eq!(
            exported.array.array.length,
            i64::try_from(expected_values.len())?
        );
        assert_eq!(exported.array.array.null_count, 0);
        assert_eq!(exported.array.array.n_buffers, 2);
        assert_eq!(exported.array.array.n_children, 0);
        assert!(exported.array.array.release.is_some());
        assert_eq!(exported.array.device_type, ARROW_DEVICE_CUDA);

        let private_data = unsafe { &*exported.array.array.private_data.cast::<PrivateData>() };
        let value_buffer = private_data.buffers[1]
            .as_ref()
            .vortex_expect("value buffer should be present");
        assert_eq!(value_buffer.len(), expected_values.len() * size_of::<T>());
        assert_exported_decimal_values(value_buffer, &expected_values);

        unsafe { release_exported_array(&raw mut exported.array.array) };
        Ok(())
    }

    #[rstest]
    #[case::i8(
        DecimalArray::from_iter([1i8, -2, 3], DecimalDType::new(2, 1)).into_array(),
        DataType::Decimal32(2, 1),
        vec![1i32, -2, 3]
    )]
    #[case::i16(
        DecimalArray::from_iter([100i16, -200, 300], DecimalDType::new(4, 2)).into_array(),
        DataType::Decimal32(4, 2),
        vec![100i32, -200, 300]
    )]
    #[case::i32(
        DecimalArray::from_iter([10_000i32, -20_000, 30_000], DecimalDType::new(9, 2)).into_array(),
        DataType::Decimal32(9, 2),
        vec![10_000i32, -20_000, 30_000]
    )]
    #[crate::test]
    async fn test_export_decimal32(
        #[case] array: ArrayRef,
        #[case] expected_data_type: DataType,
        #[case] expected_values: Vec<i32>,
    ) -> VortexResult<()> {
        assert_exported_decimal(array, expected_data_type, expected_values).await
    }

    #[rstest]
    #[case::i32(
        DecimalArray::from_iter([1_000_000i32, -2_000_000, 3_000_000], DecimalDType::new(10, 2)).into_array(),
        DataType::Decimal64(10, 2),
        vec![1_000_000i64, -2_000_000, 3_000_000]
    )]
    #[case::i32_boundary(
        DecimalArray::from_iter([i32::MIN, -1i32, 0, 1, i32::MAX], DecimalDType::new(10, 0)).into_array(),
        DataType::Decimal64(10, 0),
        vec![i32::MIN as i64, -1, 0, 1, i32::MAX as i64]
    )]
    #[case::i64(
        DecimalArray::from_iter([1_000_000i64, -2_000_000, 3_000_000], DecimalDType::new(18, 2)).into_array(),
        DataType::Decimal64(18, 2),
        vec![1_000_000i64, -2_000_000, 3_000_000]
    )]
    #[crate::test]
    async fn test_export_decimal64(
        #[case] array: ArrayRef,
        #[case] expected_data_type: DataType,
        #[case] expected_values: Vec<i64>,
    ) -> VortexResult<()> {
        assert_exported_decimal(array, expected_data_type, expected_values).await
    }

    #[rstest]
    #[case::i64_boundary(
        DecimalArray::from_iter([i64::MIN, -1i64, 0, 1, i64::MAX], DecimalDType::new(19, 0)).into_array(),
        DataType::Decimal128(19, 0),
        vec![i64::MIN as i128, -1, 0, 1, i64::MAX as i128]
    )]
    #[case::i128(
        DecimalArray::from_iter([1i128, -2, 3], DecimalDType::new(38, 2)).into_array(),
        DataType::Decimal128(38, 2),
        vec![1i128, -2, 3]
    )]
    #[crate::test]
    async fn test_export_decimal128(
        #[case] array: ArrayRef,
        #[case] expected_data_type: DataType,
        #[case] expected_values: Vec<i128>,
    ) -> VortexResult<()> {
        assert_exported_decimal(array, expected_data_type, expected_values).await
    }

    #[crate::test]
    async fn test_export_empty_decimal() -> VortexResult<()> {
        assert_exported_decimal(
            DecimalArray::new(
                Buffer::<i32>::empty(),
                DecimalDType::new(9, 2),
                Validity::NonNullable,
            )
            .into_array(),
            DataType::Decimal32(9, 2),
            Vec::<i32>::new(),
        )
        .await
    }

    #[crate::test]
    async fn test_export_empty_decimal_widening() -> VortexResult<()> {
        assert_exported_decimal(
            DecimalArray::new(
                Buffer::<i8>::empty(),
                DecimalDType::new(9, 2),
                Validity::NonNullable,
            )
            .into_array(),
            DataType::Decimal32(9, 2),
            Vec::<i32>::new(),
        )
        .await
    }

    #[crate::test]
    async fn test_export_decimal_narrowing_errors() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");
        let array = DecimalArray::from_iter([i256::from_parts(0, 1)], DecimalDType::new(38, 0))
            .into_array();

        let err = array
            .export_device_array_with_schema(&mut ctx)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("narrowing would require"));
        Ok(())
    }

    #[crate::test]
    async fn test_export_decimal_narrowing_from_arrow_import() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");
        let array = DecimalArray::from_iter([0i128, 1, -2], DecimalDType::new(10, 2)).into_array();

        let err = array
            .export_device_array_with_schema(&mut ctx)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("narrowing would require"));
        Ok(())
    }

    #[rstest]
    #[case::i64(
        DecimalArray::from_iter([i64::MIN, -1i64, 1, i64::MAX], DecimalDType::new(39, 0)).into_array(),
        DataType::Decimal256(39, 0),
        vec![i256::from_i128(i64::MIN as i128), i256::from_i128(-1), i256::from_i128(1), i256::from_i128(i64::MAX as i128)]
    )]
    #[case::i128(
        DecimalArray::from_iter([1i128, -2, 3], DecimalDType::new(39, 2)).into_array(),
        DataType::Decimal256(39, 2),
        vec![i256::from_i128(1), i256::from_i128(-2), i256::from_i128(3)]
    )]
    #[case::i256(
        DecimalArray::from_iter(
            [i256::from_i128(10), i256::from_i128(-20), i256::from_i128(30)],
            DecimalDType::new(76, 2),
        )
        .into_array(),
        DataType::Decimal256(76, 2),
        vec![i256::from_i128(10), i256::from_i128(-20), i256::from_i128(30)]
    )]
    #[crate::test]
    async fn test_export_decimal256(
        #[case] array: ArrayRef,
        #[case] expected_data_type: DataType,
        #[case] expected_values: Vec<i256>,
    ) -> VortexResult<()> {
        assert_exported_decimal(array, expected_data_type, expected_values).await
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
    async fn test_export_binary_inline_outline_values() -> VortexResult<()> {
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
        assert_eq!(field, Field::new("", DataType::Binary, true));
        assert_binary_layout(
            &exported.array.array,
            5,
            1,
            &[0, 0, 3, 3, 8, i32::try_from(8 + out_of_line.len())?],
            &[b"\x00\xff\xfe".as_slice(), b"short", out_of_line].concat(),
        )?;
        assert_eq!(exported.array.device_type, ARROW_DEVICE_CUDA);

        unsafe { release_exported_array(&raw mut exported.array.array) };
        Ok(())
    }

    #[crate::test]
    async fn test_export_binary_empty_and_all_null() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let empty = VarBinViewArray::from_iter_nullable_bin(std::iter::empty::<Option<&[u8]>>())
            .into_array();
        let mut exported = empty.export_device_array_with_schema(&mut ctx).await?;
        let field = Field::try_from(&exported.schema)?;
        assert_eq!(field, Field::new("", DataType::Binary, true));
        assert_binary_layout(&exported.array.array, 0, 0, &[0], b"")?;
        assert_eq!(exported.array.device_type, ARROW_DEVICE_CUDA);
        unsafe { release_exported_array(&raw mut exported.array.array) };

        let all_null =
            VarBinViewArray::from_iter_nullable_bin([None::<&[u8]>, None::<&[u8]>]).into_array();
        let mut exported = all_null.export_device_array_with_schema(&mut ctx).await?;
        let field = Field::try_from(&exported.schema)?;
        assert_eq!(field, Field::new("", DataType::Binary, true));
        assert_binary_layout(&exported.array.array, 2, 2, &[0, 0, 0], b"")?;
        assert_eq!(exported.array.device_type, ARROW_DEVICE_CUDA);
        unsafe { release_exported_array(&raw mut exported.array.array) };

        Ok(())
    }

    #[crate::test]
    async fn test_export_binary_invalid_data_buffer_ref_errors() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let view = BinaryView::make_view(b"this references a missing data buffer", 0, 0);
        let array = VarBinViewArray::new_handle(
            BufferHandle::new_host(Buffer::from_iter([view]).into_byte_buffer()),
            Arc::from([]),
            DType::Binary(Nullability::NonNullable),
            Validity::NonNullable,
        )
        .into_array();

        let err = array
            .export_device_array_with_schema(&mut ctx)
            .await
            .expect_err("missing binary data buffer should fail");
        assert!(
            err.to_string()
                .contains("a view references an invalid data buffer")
        );

        Ok(())
    }

    #[crate::test]
    async fn test_export_binary_i32_offset_overflow_errors() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let view = BinaryView::new_ref(i32::MAX as u32 + 1, [0; 4], 0, 0);
        let array = VarBinViewArray::new_handle(
            BufferHandle::new_host(Buffer::from_iter([view]).into_byte_buffer()),
            Arc::from([]),
            DType::Binary(Nullability::NonNullable),
            Validity::NonNullable,
        )
        .into_array();

        let err = array
            .export_device_array_with_schema(&mut ctx)
            .await
            .expect_err("oversized binary value should fail Arrow Binary export");
        assert!(
            err.to_string()
                .contains("offsets exceed i32 range required by Arrow Binary")
        );

        Ok(())
    }

    #[crate::test]
    async fn test_export_list() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let array = ListArray::try_new(
            PrimitiveArray::from_iter(0i32..5).into_array(),
            PrimitiveArray::from_iter([0i32, 2, 2, 5]).into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let mut exported = array.export_device_array_with_schema(&mut ctx).await?;

        let field = Field::try_from(&exported.schema)?;
        assert_eq!(
            field,
            Field::new_list(
                "",
                Field::new(Field::LIST_FIELD_DEFAULT_NAME, DataType::Int32, false),
                false,
            )
        );
        assert_eq!(exported.array.array.length, 3);
        assert_eq!(exported.array.array.null_count, 0);
        assert_eq!(exported.array.array.n_buffers, 2);
        let buffers = unsafe { std::slice::from_raw_parts(exported.array.array.buffers, 2) };
        assert!(buffers[0].is_null());
        assert!(!buffers[1].is_null());
        assert_eq!(exported.array.array.n_children, 1);
        let children = unsafe { std::slice::from_raw_parts(exported.array.array.children, 1) };
        let elements = unsafe { &*children[0] };
        assert_eq!(elements.length, 5);
        assert_eq!(elements.n_buffers, 2);
        assert_eq!(exported.array.device_type, ARROW_DEVICE_CUDA);

        unsafe { release_exported_array(&raw mut exported.array.array) };
        Ok(())
    }

    #[crate::test]
    async fn test_export_host_contiguous_list_view() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let array = ListViewArray::new(
            PrimitiveArray::from_iter(0i32..5).into_array(),
            PrimitiveArray::from_iter([0i32, 2, 2]).into_array(),
            PrimitiveArray::from_iter([2i32, 0, 3]).into_array(),
            Validity::NonNullable,
        )
        .into_array();
        let mut exported = array.export_device_array_with_schema(&mut ctx).await?;

        assert_eq!(exported.array.array.length, 3);
        assert_eq!(exported.array.array.n_buffers, 2);
        assert_eq!(
            private_data_buffer_i32_values(&exported.array.array, 1)?,
            [0, 2, 2, 5]
        );
        assert_eq!(exported.array.device_type, ARROW_DEVICE_CUDA);

        unsafe { release_exported_array(&raw mut exported.array.array) };
        Ok(())
    }

    #[crate::test]
    async fn test_export_host_non_contiguous_nested_list_view_falls_back_to_cpu() -> VortexResult<()>
    {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let elements = StructArray::new(
            FieldNames::from_iter(["x"]),
            vec![PrimitiveArray::from_iter(0i32..4).into_array()],
            4,
            Validity::NonNullable,
        )
        .into_array();
        let array = ListViewArray::new(
            elements,
            PrimitiveArray::from_iter([0i32, 1]).into_array(),
            PrimitiveArray::from_iter([3i32, 2]).into_array(),
            Validity::NonNullable,
        )
        .into_array();
        let mut exported = array.export_device_array_with_schema(&mut ctx).await?;

        assert_eq!(exported.array.array.length, 2);
        assert_eq!(
            private_data_buffer_i32_values(&exported.array.array, 1)?,
            [0, 3, 5]
        );
        let list_children = unsafe { std::slice::from_raw_parts(exported.array.array.children, 1) };
        let struct_child = unsafe { &*list_children[0] };
        assert_eq!(struct_child.length, 5);
        let struct_children = unsafe { std::slice::from_raw_parts(struct_child.children, 1) };
        let field_child = unsafe { &*struct_children[0] };
        assert_eq!(
            private_data_buffer_i32_values(field_child, 1)?,
            [0, 1, 2, 1, 2]
        );
        assert_eq!(exported.array.device_type, ARROW_DEVICE_CUDA);

        unsafe { release_exported_array(&raw mut exported.array.array) };
        Ok(())
    }

    #[crate::test]
    async fn test_export_host_non_contiguous_dictionary_list_view_schema_matches_rebuilt_child()
    -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let elements = DictArray::try_new(
            PrimitiveArray::from_option_iter([Some(0u8), None, Some(1), Some(2)]).into_array(),
            PrimitiveArray::from_iter([10i32, 20, 30]).into_array(),
        )?
        .into_array();
        let array = ListViewArray::new(
            elements,
            PrimitiveArray::from_iter([0i32, 1]).into_array(),
            PrimitiveArray::from_iter([3i32, 2]).into_array(),
            Validity::NonNullable,
        )
        .into_array();
        let mut exported = array.export_device_array_with_schema(&mut ctx).await?;

        let field = Field::try_from(&exported.schema)?;
        assert_eq!(
            field,
            Field::new_list(
                "",
                Field::new(
                    Field::LIST_FIELD_DEFAULT_NAME,
                    DataType::Dictionary(
                        Box::new(DataType::Int64),
                        Box::new(DataType::Dictionary(
                            Box::new(DataType::Int16),
                            Box::new(DataType::Int32),
                        )),
                    ),
                    true,
                ),
                false,
            )
        );
        assert_eq!(
            private_data_buffer_i32_values(&exported.array.array, 1)?,
            [0, 3, 5]
        );
        let list_children = unsafe { std::slice::from_raw_parts(exported.array.array.children, 1) };
        let dict_child = unsafe { &*list_children[0] };
        assert!(!dict_child.dictionary.is_null());
        assert_eq!(dict_child.length, 5);
        assert_eq!(dict_child.n_buffers, 2);
        let nested_dict = unsafe { &*dict_child.dictionary };
        assert!(!nested_dict.dictionary.is_null());

        unsafe { release_exported_array(&raw mut exported.array.array) };
        Ok(())
    }

    // Regression test: with an average list size >= 128 the host list-view rebuild picks its
    // list-by-list strategy, which may canonicalize Dict elements. The schema must describe the
    // rebuilt child layout.
    #[crate::test]
    async fn test_export_host_large_lists_dictionary_list_view_schema_matches_rebuilt_child()
    -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let elements = DictArray::try_new(
            PrimitiveArray::from_option_iter(
                (0..256u32).map(|i| (i % 5 != 0).then_some((i % 3) as u8)),
            )
            .into_array(),
            PrimitiveArray::from_iter([10i32, 20, 30]).into_array(),
        )?
        .into_array();
        let array = ListViewArray::new(
            elements,
            PrimitiveArray::from_iter([128i32, 0]).into_array(),
            PrimitiveArray::from_iter([128i32, 128]).into_array(),
            Validity::NonNullable,
        )
        .into_array();
        let mut exported = array.export_device_array_with_schema(&mut ctx).await?;

        let field = Field::try_from(&exported.schema)?;
        assert_eq!(
            field,
            Field::new_list(
                "",
                Field::new(Field::LIST_FIELD_DEFAULT_NAME, DataType::Int32, true),
                false,
            )
        );
        assert_eq!(
            private_data_buffer_i32_values(&exported.array.array, 1)?,
            [0, 128, 256]
        );
        let list_children = unsafe { std::slice::from_raw_parts(exported.array.array.children, 1) };
        let child = unsafe { &*list_children[0] };
        assert!(child.dictionary.is_null());
        assert_eq!(child.length, 256);
        assert_eq!(child.n_buffers, 2);

        unsafe { release_exported_array(&raw mut exported.array.array) };
        Ok(())
    }

    #[rstest]
    #[case::i32_i32(PType::I32, PType::I32)]
    #[case::u32_u16(PType::U32, PType::U16)]
    #[case::i64_u8(PType::I64, PType::U8)]
    #[case::u64_i16(PType::U64, PType::I16)]
    #[crate::test]
    async fn test_export_device_contiguous_list_view(
        #[case] offsets_ptype: PType,
        #[case] sizes_ptype: PType,
    ) -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let elements = primitive_i32_on_device(0..5, &mut ctx).await?;
        let offsets = integer_array_on_device(offsets_ptype, &[0, 2, 2], &mut ctx).await?;
        let sizes = integer_array_on_device(sizes_ptype, &[2, 0, 3], &mut ctx).await?;
        let array =
            ListViewArray::new(elements, offsets, sizes, Validity::NonNullable).into_array();
        let mut exported = array.export_device_array_with_schema(&mut ctx).await?;

        let field = Field::try_from(&exported.schema)?;
        assert_eq!(
            field,
            Field::new_list(
                "",
                Field::new(Field::LIST_FIELD_DEFAULT_NAME, DataType::Int32, false),
                false,
            )
        );
        assert_eq!(exported.array.array.length, 3);
        assert_eq!(exported.array.array.n_buffers, 2);
        assert_eq!(
            private_data_buffer_i32_values(&exported.array.array, 1)?,
            [0, 2, 2, 5]
        );
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
        DataType::Binary
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
        let is_binary = expected_data_type == DataType::Binary;
        assert_eq!(field, Field::new("", expected_data_type, false));
        if is_binary {
            assert_binary_layout(
                &exported.array.array,
                3,
                0,
                &[0, 6, 36, 67],
                b"inlinefirst value stored out-of-linesecond value stored out-of-line",
            )?;
        } else {
            assert_varbinview_layout(&exported.array.array, 3, 0, &expected_data_buffer_lengths)?;
        }
        assert_eq!(exported.array.device_type, ARROW_DEVICE_CUDA);

        unsafe { release_exported_array(&raw mut exported.array.array) };
        Ok(())
    }

    #[rstest]
    #[case::i64(PrimitiveArray::from_iter([0i64, 2, 2, 5]).into_array())]
    #[case::u64(PrimitiveArray::from_iter([0u64, 2, 2, 5]).into_array())]
    #[crate::test]
    async fn test_export_list_with_non_i32_offsets(#[case] offsets: ArrayRef) -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let array = ListArray::try_new(
            PrimitiveArray::from_iter(0i32..5).into_array(),
            offsets,
            Validity::NonNullable,
        )?
        .into_array();
        let mut exported = array.export_device_array_with_schema(&mut ctx).await?;

        assert_eq!(exported.array.array.length, 3);
        assert_eq!(exported.array.array.n_buffers, 2);
        assert_eq!(
            private_data_buffer_i32_values(&exported.array.array, 1)?,
            [0, 2, 2, 5]
        );

        unsafe { release_exported_array(&raw mut exported.array.array) };
        Ok(())
    }

    #[rstest]
    #[case::i32_i32(PType::I32, PType::I32)]
    #[case::u32_u16(PType::U32, PType::U16)]
    #[case::i64_u8(PType::I64, PType::U8)]
    #[case::u64_i16(PType::U64, PType::I16)]
    #[crate::test]
    async fn test_export_device_non_contiguous_primitive_list_view(
        #[case] offsets_ptype: PType,
        #[case] sizes_ptype: PType,
    ) -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let elements = primitive_i32_on_device([10, 11, 12, 13, 14], &mut ctx).await?;
        let offsets = integer_array_on_device(offsets_ptype, &[3, 0, 2], &mut ctx).await?;
        let sizes = integer_array_on_device(sizes_ptype, &[2, 2, 1], &mut ctx).await?;
        let array =
            ListViewArray::new(elements, offsets, sizes, Validity::NonNullable).into_array();
        let mut exported = array.export_device_array_with_schema(&mut ctx).await?;

        assert_eq!(exported.array.array.length, 3);
        assert_eq!(
            private_data_buffer_i32_values(&exported.array.array, 1)?,
            [0, 2, 4, 5]
        );
        let children = unsafe { std::slice::from_raw_parts(exported.array.array.children, 1) };
        let elements = unsafe { &*children[0] };
        assert_eq!(
            private_data_buffer_i32_values(elements, 1)?,
            [13, 14, 10, 11, 12]
        );
        assert_eq!(exported.array.device_type, ARROW_DEVICE_CUDA);

        unsafe { release_exported_array(&raw mut exported.array.array) };
        Ok(())
    }

    #[crate::test]
    async fn test_export_device_non_contiguous_dictionary_list_view() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let codes = primitive_on_device([0u8, 1, 2, 0, 1], &mut ctx).await?;
        let values = PrimitiveArray::from_iter([10i32, 20, 30]).into_array();
        let elements = DictArray::try_new(codes, values)?.into_array();
        let offsets = primitive_i32_on_device([3, 0, 2], &mut ctx).await?;
        let sizes = primitive_i32_on_device([2, 2, 1], &mut ctx).await?;
        let array =
            ListViewArray::new(elements, offsets, sizes, Validity::NonNullable).into_array();
        let mut exported = array.export_device_array_with_schema(&mut ctx).await?;

        let field = Field::try_from(&exported.schema)?;
        assert_eq!(
            field,
            Field::new_list(
                "",
                Field::new(
                    Field::LIST_FIELD_DEFAULT_NAME,
                    DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Int32)),
                    false,
                ),
                false,
            )
        );
        assert_eq!(exported.array.array.length, 3);
        assert_eq!(
            private_data_buffer_i32_values(&exported.array.array, 1)?,
            [0, 2, 4, 5]
        );
        let children = unsafe { std::slice::from_raw_parts(exported.array.array.children, 1) };
        let elements = unsafe { &*children[0] };
        assert!(!elements.dictionary.is_null());
        assert_eq!(
            private_data_buffer_i16_values(elements, 1)?,
            [0, 1, 0, 1, 2]
        );
        let dictionary = unsafe { &*elements.dictionary };
        assert_eq!(private_data_buffer_i32_values(dictionary, 1)?, [10, 20, 30]);
        assert_eq!(exported.array.device_type, ARROW_DEVICE_CUDA);

        unsafe { release_exported_array(&raw mut exported.array.array) };
        Ok(())
    }

    #[crate::test]
    async fn test_export_device_non_contiguous_dictionary_list_view_nullable_values()
    -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let codes = primitive_on_device([0u8, 1, 2, 0, 1], &mut ctx).await?;
        let values = PrimitiveArray::from_option_iter([Some(10i32), None, Some(30)]).into_array();
        let elements = DictArray::try_new(codes, values)?.into_array();
        let offsets = primitive_i32_on_device([3, 0, 2], &mut ctx).await?;
        let sizes = primitive_i32_on_device([2, 2, 1], &mut ctx).await?;
        let array =
            ListViewArray::new(elements, offsets, sizes, Validity::NonNullable).into_array();
        let mut exported = array.export_device_array_with_schema(&mut ctx).await?;

        let children = unsafe { std::slice::from_raw_parts(exported.array.array.children, 1) };
        let elements = unsafe { &*children[0] };
        assert_eq!(elements.null_count, 0);
        assert_eq!(
            private_data_buffer_i16_values(elements, 1)?,
            [0, 1, 0, 1, 2]
        );
        let dictionary = unsafe { &*elements.dictionary };
        assert_eq!(dictionary.null_count, 1);

        unsafe { release_exported_array(&raw mut exported.array.array) };
        Ok(())
    }

    #[crate::test]
    async fn test_export_device_non_contiguous_dictionary_list_view_nullable_codes_errors()
    -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let codes = PrimitiveArray::from_option_iter([Some(0u8), None, Some(2), Some(0), Some(1)]);
        let codes_handle = ctx.ensure_on_device(codes.buffer_handle().clone()).await?;
        let codes = PrimitiveArray::from_buffer_handle(codes_handle, PType::U8, codes.validity()?)
            .into_array();
        let values =
            PrimitiveArray::from_option_iter([Some(10i32), Some(20), Some(30)]).into_array();
        let elements = DictArray::try_new(codes, values)?.into_array();
        let offsets = primitive_i32_on_device([3, 0, 2], &mut ctx).await?;
        let sizes = primitive_i32_on_device([2, 2, 1], &mut ctx).await?;
        let array =
            ListViewArray::new(elements, offsets, sizes, Validity::NonNullable).into_array();
        let err = match array.export_device_array(&mut ctx).await {
            Ok(mut exported) => {
                unsafe { release_exported_array(&raw mut exported.array) };
                vortex_bail!("nullable dictionary codes should be unsupported")
            }
            Err(err) => err,
        };

        assert!(
            err.to_string().contains("nullable dictionary codes"),
            "unexpected error: {err}"
        );
        Ok(())
    }

    #[rstest]
    #[case::out_of_bounds(&[3], &[2], "offsets/sizes are invalid")]
    #[case::negative_offset(&[-1], &[1], "offsets exceed i32 range")]
    #[crate::test]
    async fn test_export_device_invalid_list_view_returns_error(
        #[case] offsets_values: &[i64],
        #[case] sizes_values: &[i64],
        #[case] expected_error: &str,
    ) -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let elements = primitive_i32_on_device(0..4, &mut ctx).await?;
        let offsets = integer_array_on_device(PType::I32, offsets_values, &mut ctx).await?;
        let sizes = integer_array_on_device(PType::I32, sizes_values, &mut ctx).await?;
        let array = unsafe {
            ListViewArray::new_unchecked(elements, offsets, sizes, Validity::NonNullable)
        }
        .into_array();
        let err = match array.export_device_array(&mut ctx).await {
            Ok(mut exported) => {
                unsafe { release_exported_array(&raw mut exported.array) };
                vortex_bail!("invalid device list view should be unsupported")
            }
            Err(err) => err,
        };

        assert!(
            err.to_string().contains(expected_error),
            "unexpected error: {err}"
        );
        Ok(())
    }

    #[crate::test]
    async fn test_export_device_non_contiguous_nested_list_view_returns_error() -> VortexResult<()>
    {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let field = primitive_i32_on_device(0..4, &mut ctx).await?;
        let elements = StructArray::new(
            FieldNames::from_iter(["x"]),
            vec![field],
            4,
            Validity::NonNullable,
        )
        .into_array();
        let offsets = primitive_i32_on_device([0, 1], &mut ctx).await?;
        let sizes = primitive_i32_on_device([3, 2], &mut ctx).await?;
        let array =
            ListViewArray::new(elements, offsets, sizes, Validity::NonNullable).into_array();
        let err = match array.export_device_array(&mut ctx).await {
            Ok(mut exported) => {
                unsafe { release_exported_array(&raw mut exported.array) };
                vortex_bail!("non-contiguous nested list view should be unsupported")
            }
            Err(err) => err,
        };

        assert!(
            err.to_string()
                .contains("GPU child rebuild only supports primitive children"),
            "unexpected error: {err}"
        );
        Ok(())
    }

    #[crate::test]
    async fn test_export_device_non_contiguous_nullable_primitive_list_view_returns_error()
    -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let elements = nullable_primitive_i32_on_device(
            [Some(10), None, Some(12), Some(13), Some(14)],
            &mut ctx,
        )
        .await?;
        let offsets = primitive_i32_on_device([3, 0, 2], &mut ctx).await?;
        let sizes = primitive_i32_on_device([2, 2, 1], &mut ctx).await?;
        let array =
            ListViewArray::new(elements, offsets, sizes, Validity::NonNullable).into_array();
        let err = match array.export_device_array(&mut ctx).await {
            Ok(mut exported) => {
                unsafe { release_exported_array(&raw mut exported.array) };
                vortex_bail!("non-contiguous nullable primitive list view should be unsupported")
            }
            Err(err) => err,
        };

        assert!(
            err.to_string()
                .contains("GPU child validity rebuild is not implemented"),
            "unexpected error: {err}"
        );
        Ok(())
    }

    #[crate::test]
    async fn test_export_fixed_size_list_as_list() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let array = FixedSizeListArray::new(
            PrimitiveArray::from_iter(0i32..6).into_array(),
            2,
            Validity::NonNullable,
            3,
        )
        .into_array();
        let mut exported = array.export_device_array_with_schema(&mut ctx).await?;

        let field = Field::try_from(&exported.schema)?;
        assert_eq!(
            field,
            Field::new_list(
                "",
                Field::new(Field::LIST_FIELD_DEFAULT_NAME, DataType::Int32, false),
                false,
            )
        );
        assert_eq!(exported.array.array.length, 3);
        assert_eq!(exported.array.array.null_count, 0);
        assert_eq!(exported.array.array.n_buffers, 2);
        let buffers = unsafe { std::slice::from_raw_parts(exported.array.array.buffers, 2) };
        assert!(buffers[0].is_null());
        assert!(!buffers[1].is_null());
        assert_eq!(
            private_data_buffer_i32_values(&exported.array.array, 1)?,
            [0, 2, 4, 6]
        );
        assert_eq!(exported.array.array.n_children, 1);
        let children = unsafe { std::slice::from_raw_parts(exported.array.array.children, 1) };
        let elements = unsafe { &*children[0] };
        assert_eq!(elements.length, 6);
        assert_eq!(elements.n_buffers, 2);
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

        let utf8 = VarBinViewArray::from_iter_nullable_str([
            Some("skip this out-of-line value before the slice"),
            Some("hello"),
            None,
            Some("this out-of-line value remains in the slice"),
        ])
        .into_array()
        .slice(1..4)?;
        let mut exported = utf8.export_device_array_with_schema(&mut ctx).await?;
        let field = Field::try_from(&exported.schema)?;
        assert_eq!(field, Field::new("", DataType::Utf8View, true));
        assert_varbinview_shape(&exported.array.array, 3, 1)?;
        assert_eq!(exported.array.device_type, ARROW_DEVICE_CUDA);

        let private_data = unsafe { &*exported.array.array.private_data.cast::<PrivateData>() };
        let null_buffer = private_data.buffers[0]
            .as_ref()
            .vortex_expect("sliced null buffer should be present");
        let null_bits = BitBuffer::new(null_buffer.to_host_sync(), 3)
            .iter()
            .collect::<Vec<_>>();
        assert_eq!(null_bits, [true, false, true]);
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
        assert_eq!(field, Field::new("", DataType::Binary, true));
        let sliced_out_of_line = b"this out-of-line binary value remains in the slice";
        assert_binary_layout(
            &exported.array.array,
            3,
            1,
            &[0, 0, 2, i32::try_from(2 + sliced_out_of_line.len())?],
            &[b"\x00\xff".as_slice(), sliced_out_of_line].concat(),
        )?;
        assert_eq!(exported.array.device_type, ARROW_DEVICE_CUDA);
        unsafe { release_exported_array(&raw mut exported.array.array) };

        Ok(())
    }

    #[rstest::rstest]
    #[case::input_ahead_of_arrow(5, 3, 9)]
    #[case::arrow_ahead_of_input(3, 70, 9)]
    #[case::equal_offsets(7, 7, 9)]
    #[case::byte_aligned_input(0, 9, 9)]
    #[case::word_aligned_offsets(64, 128, 130)]
    #[case::multi_word(13, 0, 301)]
    #[crate::test]
    async fn test_repack_arrow_validity_buffer_offsets(
        #[case] input_offset: usize,
        #[case] arrow_offset: usize,
        #[case] len: usize,
    ) -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let logical_bits = (0..len).map(|idx| idx % 3 != 0).collect::<Vec<_>>();
        // All-true filler before the slice would leak into the output if offsets were mishandled.
        let source = BitBuffer::from_iter(
            std::iter::repeat_n(true, input_offset).chain(logical_bits.iter().copied()),
        );
        let sliced = source.slice(input_offset..input_offset + len);
        // BitBuffer rebases whole bytes into the backing buffer, keeping the bit offset below 8.
        let (input_offset, _, input_buffer) = sliced.into_inner();

        let input_buffer = ctx
            .ensure_on_device(BufferHandle::new_host(input_buffer))
            .await?;
        let output_bits = len + arrow_offset;
        let output =
            repack_arrow_validity_buffer(&input_buffer, input_offset, len, arrow_offset, &mut ctx)?;
        ctx.synchronize_stream()?;

        let actual = BitBuffer::new(output.to_host_sync(), output_bits)
            .iter()
            .collect::<Vec<_>>();
        let expected = std::iter::repeat_n(false, arrow_offset)
            .chain(logical_bits)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);

        Ok(())
    }

    #[crate::test]
    async fn test_repack_arrow_validity_buffer_zeroes_padding() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let len = 9;
        let arrow_offset = 3;
        let source = BitBuffer::from_iter(std::iter::repeat_n(true, len));
        let (input_offset, _, input_buffer) = source.into_inner();
        let input_buffer = ctx
            .ensure_on_device(BufferHandle::new_host(input_buffer))
            .await?;
        let output_bytes = (len + arrow_offset).div_ceil(8);

        let output =
            repack_arrow_validity_buffer(&input_buffer, input_offset, len, arrow_offset, &mut ctx)?;
        ctx.synchronize_stream()?;

        assert_eq!(output.len(), output_bytes);
        let backing = cuda_backing_allocation(&output)?;
        let backing_bytes = backing.to_host_sync();
        assert_eq!(
            backing_bytes.len(),
            output_bytes.next_multiple_of(size_of::<u64>())
        );
        assert!(backing_bytes[output_bytes..].iter().all(|byte| *byte == 0));

        Ok(())
    }

    #[crate::test]
    async fn test_export_all_false_validity_buffer_is_zeroed_on_device() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let len = 4;
        let arrow_offset = 5;
        let (buffer, null_count) = export_arrow_validity_buffer(
            Validity::from(BitBuffer::from_iter(std::iter::repeat_n(false, len))),
            len,
            arrow_offset,
            &mut ctx,
        )
        .await?;
        ctx.synchronize_stream()?;

        assert_eq!(null_count, i64::try_from(len)?);
        let buffer = buffer.vortex_expect("all-false validity should export a null buffer");
        let bytes = buffer.to_host_sync();
        assert_eq!(bytes.len(), (len + arrow_offset).div_ceil(8));
        assert!(bytes.iter().all(|byte| *byte == 0));

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

        let private_data = unsafe { &*decimal.array.private_data.cast::<PrivateData>() };
        let value_buffer = private_data.buffers[1]
            .as_ref()
            .vortex_expect("value buffer should be present");
        assert_eq!(value_buffer.len(), 3 * size_of::<i64>());
        assert_exported_decimal_values(value_buffer, &[100i64, 0, 300]);

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
    async fn test_export_nested_struct_decimal_with_schema() -> VortexResult<()> {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let nested = StructArray::new(
            FieldNames::from_iter(["amount"]),
            vec![
                DecimalArray::from_iter([100i32, -200, 300], DecimalDType::new(9, 2)).into_array(),
            ],
            3,
            Validity::NonNullable,
        )
        .into_array();
        let array = StructArray::new(
            FieldNames::from_iter(["nested"]),
            vec![nested],
            3,
            Validity::NonNullable,
        )
        .into_array();
        let mut exported = array.export_device_array_with_schema(&mut ctx).await?;

        let schema = Schema::try_from(&exported.schema)?;
        assert_eq!(
            schema,
            Schema::new(vec![Field::new(
                "nested",
                DataType::Struct(Fields::from(vec![Field::new(
                    "amount",
                    DataType::Decimal32(9, 2),
                    false,
                )])),
                false,
            )])
        );

        let children = unsafe {
            std::slice::from_raw_parts(
                exported.array.array.children,
                usize::try_from(exported.array.array.n_children)?,
            )
        };
        let nested_child = unsafe { &*children[0] };
        let nested_children = unsafe {
            std::slice::from_raw_parts(
                nested_child.children,
                usize::try_from(nested_child.n_children)?,
            )
        };
        let decimal_child = unsafe { &*nested_children[0] };
        let private_data = unsafe { &*decimal_child.private_data.cast::<PrivateData>() };
        let value_buffer = private_data.buffers[1]
            .as_ref()
            .vortex_expect("value buffer should be present");
        assert_eq!(value_buffer.len(), 3 * size_of::<i32>());

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
