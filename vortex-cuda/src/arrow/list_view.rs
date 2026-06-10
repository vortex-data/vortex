// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA Arrow Device export helpers for Vortex `ListViewArray`.

use std::sync::Arc;

use cudarc::driver::CudaSlice;
use cudarc::driver::DeviceRepr;
use cudarc::driver::PushKernelArg;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::arrays::Dict;
use vortex::array::arrays::DictArray;
use vortex::array::arrays::ListViewArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::dict::DictOwnedExt;
use vortex::array::arrays::listview::ListViewDataParts;
use vortex::array::arrays::primitive::PrimitiveDataParts;
use vortex::array::buffer::BufferHandle;
use vortex::array::match_each_integer_ptype;
use vortex::buffer::Buffer;
use vortex::dtype::NativePType;
use vortex::dtype::PType;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;

use super::ArrowArray;
use super::SyncEvent;
use super::canonical::ListChildExport;
use super::canonical::export_arrow_validity_buffer;
use super::canonical::export_list_layout;
use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::CudaExecutionCtx;
use crate::cub::exclusive_sum_i32;
use crate::executor::CudaArrayExt;

/// Export a Vortex list-view as Arrow `List` using device kernels.
///
/// Reuses contiguous children; rebuilds non-contiguous primitive or dictionary-code children.
pub(super) async fn export_device_list_view(
    array: ListViewArray,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<(ArrowArray, SyncEvent)> {
    let len = array.len();
    let ListViewDataParts {
        elements,
        offsets,
        sizes,
        validity,
        ..
    } = array.into_data_parts();

    let (validity_buffer, null_count) = export_arrow_validity_buffer(validity, len, 0, ctx).await?;

    let (offsets_ptype, offsets_buffer) =
        primitive_device_buffer(offsets, "list offsets", ctx).await?;
    let (sizes_ptype, sizes_buffer) = primitive_device_buffer(sizes, "list sizes", ctx).await?;

    match export_device_list_view_offsets(
        offsets_ptype,
        offsets_buffer.clone(),
        sizes_ptype,
        sizes_buffer.clone(),
        len,
        ctx,
    )
    .await?
    {
        DeviceListViewOffsets::Contiguous(offsets_buffer) => {
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
        DeviceListViewOffsets::RequiresRebuild => match elements.try_downcast::<Dict>() {
            Ok(dict) => {
                export_rebuilt_dict_list_view(
                    dict,
                    offsets_ptype,
                    offsets_buffer,
                    sizes_ptype,
                    sizes_buffer,
                    len,
                    validity_buffer,
                    null_count,
                    ctx,
                )
                .await
            }
            Err(elements) => {
                export_rebuilt_primitive_list_view(
                    elements,
                    offsets_ptype,
                    offsets_buffer,
                    sizes_ptype,
                    sizes_buffer,
                    len,
                    validity_buffer,
                    null_count,
                    ctx,
                )
                .await
            }
        },
    }
}

enum DeviceListViewOffsets {
    Contiguous(BufferHandle),
    RequiresRebuild,
}

/// Build cuDF-supported `i32` Arrow `List` offsets from list-view offset/size device buffers.
#[expect(clippy::cognitive_complexity)]
async fn export_device_list_view_offsets(
    offsets_ptype: PType,
    offsets_buffer: BufferHandle,
    sizes_ptype: PType,
    sizes_buffer: BufferHandle,
    len: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<DeviceListViewOffsets> {
    if len == 0 {
        let offsets = ctx
            .ensure_on_device(BufferHandle::new_host(
                Buffer::from(vec![0i32]).into_byte_buffer(),
            ))
            .await?;
        return Ok(DeviceListViewOffsets::Contiguous(offsets));
    }

    match_each_integer_ptype!(offsets_ptype, |O| {
        match_each_integer_ptype!(sizes_ptype, |S| {
            export_device_list_view_offsets_typed::<O, S>(offsets_buffer, sizes_buffer, len, ctx)
                .await
        })
    })
}

/// Rebuild primitive list-view offsets and values for concrete offset and size types.
async fn rebuild_primitive_list_view_typed<O, S>(
    offsets: BufferHandle,
    sizes: BufferHandle,
    values: BufferHandle,
    elements_len: usize,
    list_len: usize,
    values_ptype: PType,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<(BufferHandle, BufferHandle)>
where
    O: NativePType + DeviceRepr + Send + Sync + 'static,
    S: NativePType + DeviceRepr + Send + Sync + 'static,
{
    let status = new_list_view_status(ctx).await?;
    let scan_len = list_len + 1;

    let scan_input = init_list_view_rebuild_scan::<S>(&sizes, &status, list_len, ctx)?;
    let output_offsets = BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(
        exclusive_sum_i32(&scan_input, scan_len, ctx)?,
    )));

    validate_list_view_rebuild_offsets::<S>(&sizes, &output_offsets, &status, list_len, ctx)?;
    check_list_view_rebuild_status(&status).await?;

    let total_values = total_values_from_offsets(&output_offsets, list_len).await?;
    let value_width = values_ptype.byte_width();
    let output_values_bytes = total_values
        .checked_mul(value_width)
        .ok_or_else(|| vortex_err!("rebuilt list child byte length overflow"))?;

    let output_values = gather_rebuilt_primitive_values::<O, S>(
        &offsets,
        &sizes,
        &values,
        &output_offsets,
        output_values_bytes,
        elements_len,
        list_len,
        value_width,
        &status,
        ctx,
    )?;
    check_list_view_rebuild_status(&status).await?;

    let values = BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(output_values)))
        .slice(0..output_values_bytes);
    Ok((output_offsets, values))
}

/// Allocate the device status word used by list-view rebuild kernels.
async fn new_list_view_status(ctx: &mut CudaExecutionCtx) -> VortexResult<BufferHandle> {
    ctx.ensure_on_device(BufferHandle::new_host(
        Buffer::from(vec![0u32]).into_byte_buffer(),
    ))
    .await
}

/// Convert the list-view rebuild status word into a Vortex error.
async fn check_list_view_rebuild_status(status: &BufferHandle) -> VortexResult<()> {
    match Buffer::<u32>::from_byte_buffer(status.try_to_host()?.await?)[0] {
        0 => Ok(()),
        1 => vortex_bail!(
            "cannot export device-resident ListViewArray as Arrow List: offsets/sizes are invalid for the child elements"
        ),
        2 => vortex_bail!(
            "cannot export device-resident ListViewArray as Arrow List: offsets exceed i32 range required by cuDF"
        ),
        status => vortex_bail!("unexpected list-view rebuild status {status}"),
    }
}

/// Initialize the exclusive-scan input for rebuilt Arrow List offsets.
fn init_list_view_rebuild_scan<S>(
    sizes: &BufferHandle,
    status: &BufferHandle,
    list_len: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<CudaSlice<i32>>
where
    S: NativePType + DeviceRepr + Send + Sync + 'static,
{
    let scan_len = list_len + 1;
    let sizes_view = sizes.cuda_view::<S>()?;
    let status_view = status.cuda_view::<u32>()?;
    let list_len_u64 = list_len as u64;
    let scan_len_u64 = scan_len as u64;
    let scan_input = ctx.device_alloc::<i32>(scan_len)?;
    let init_kernel = ctx
        .load_function_with_suffixes("list_view", &["rebuild_init_scan", &S::PTYPE.to_string()])?;

    ctx.launch_kernel(&init_kernel, scan_len, |args| {
        args.arg(&sizes_view)
            .arg(&scan_input)
            .arg(&status_view)
            .arg(&list_len_u64)
            .arg(&scan_len_u64);
    })?;

    Ok(scan_input)
}

/// Validate rebuilt Arrow List offsets and flag invalid views on device.
fn validate_list_view_rebuild_offsets<S>(
    sizes: &BufferHandle,
    output_offsets: &BufferHandle,
    status: &BufferHandle,
    list_len: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<()>
where
    S: NativePType + DeviceRepr + Send + Sync + 'static,
{
    let sizes_view = sizes.cuda_view::<S>()?;
    let output_offsets_view = output_offsets.cuda_view::<i32>()?;
    let status_view = status.cuda_view::<u32>()?;
    let list_len_u64 = list_len as u64;
    let validate_kernel = ctx.load_function_with_suffixes(
        "list_view",
        &["rebuild_validate_offsets", &S::PTYPE.to_string()],
    )?;

    ctx.launch_kernel(&validate_kernel, list_len, |args| {
        args.arg(&sizes_view)
            .arg(&output_offsets_view)
            .arg(&status_view)
            .arg(&list_len_u64);
    })
}

/// Read the final rebuilt offset to determine the output child length.
async fn total_values_from_offsets(
    output_offsets: &BufferHandle,
    list_len: usize,
) -> VortexResult<usize> {
    let total_values = Buffer::<i32>::from_byte_buffer(
        output_offsets
            .slice_typed::<i32>(list_len..list_len + 1)
            .try_to_host()?
            .await?,
    )[0];

    usize::try_from(total_values).map_err(Into::into)
}

/// Gather primitive child bytes from each list-view range into contiguous list order.
#[expect(clippy::too_many_arguments)]
fn gather_rebuilt_primitive_values<O, S>(
    offsets: &BufferHandle,
    sizes: &BufferHandle,
    values: &BufferHandle,
    output_offsets: &BufferHandle,
    output_values_bytes: usize,
    elements_len: usize,
    list_len: usize,
    value_width: usize,
    status: &BufferHandle,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<CudaSlice<u8>>
where
    O: NativePType + DeviceRepr + Send + Sync + 'static,
    S: NativePType + DeviceRepr + Send + Sync + 'static,
{
    let offsets_view = offsets.cuda_view::<O>()?;
    let sizes_view = sizes.cuda_view::<S>()?;
    let values_view = values.cuda_view::<u8>()?;
    let output_offsets_view = output_offsets.cuda_view::<i32>()?;
    let output_values = ctx.device_alloc::<u8>(output_values_bytes.max(1))?;
    let status_view = status.cuda_view::<u32>()?;
    let list_len_u64 = list_len as u64;
    let elements_len_u64 = elements_len as u64;
    let value_width_u64 = value_width as u64;
    let rebuild_kernel = ctx.load_function_with_suffixes(
        "list_view",
        &[
            "rebuild_primitive",
            &O::PTYPE.to_string(),
            &S::PTYPE.to_string(),
        ],
    )?;

    ctx.launch_kernel(&rebuild_kernel, list_len, |args| {
        args.arg(&offsets_view)
            .arg(&sizes_view)
            .arg(&output_offsets_view)
            .arg(&values_view)
            .arg(&output_values)
            .arg(&status_view)
            .arg(&list_len_u64)
            .arg(&elements_len_u64)
            .arg(&value_width_u64);
    })?;

    Ok(output_values)
}

/// Rebuild non-contiguous dictionary list-view codes while reusing the dictionary values.
#[expect(clippy::too_many_arguments)]
async fn export_rebuilt_dict_list_view(
    dict: DictArray,
    offsets_ptype: PType,
    offsets_buffer: BufferHandle,
    sizes_ptype: PType,
    sizes_buffer: BufferHandle,
    len: usize,
    validity_buffer: Option<BufferHandle>,
    null_count: i64,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<(ArrowArray, SyncEvent)> {
    let parts = dict.into_parts();
    let canonical_codes = parts.codes.execute_cuda(ctx).await?;
    let Canonical::Primitive(codes) = canonical_codes else {
        vortex_bail!(
            "cannot export non-contiguous device-resident ListViewArray with dictionary codes of {}: GPU child rebuild only supports primitive dictionary codes",
            canonical_codes.dtype()
        );
    };

    let (offsets_buffer, rebuilt_codes) = rebuild_primitive_list_view_child(
        codes,
        "dictionary codes",
        offsets_ptype,
        offsets_buffer,
        sizes_ptype,
        sizes_buffer,
        len,
        ctx,
    )
    .await?;
    let rebuilt_dict = DictArray::try_new(rebuilt_codes.into_array(), parts.values)?.into_array();

    export_list_layout(
        rebuilt_dict,
        len,
        validity_buffer,
        null_count,
        offsets_buffer,
        ListChildExport::PreserveConcreteLayout,
        ctx,
    )
    .await
}

/// Rebuild a non-contiguous primitive list-view child and export it as an Arrow List.
#[expect(clippy::too_many_arguments)]
async fn export_rebuilt_primitive_list_view(
    elements: ArrayRef,
    offsets_ptype: PType,
    offsets_buffer: BufferHandle,
    sizes_ptype: PType,
    sizes_buffer: BufferHandle,
    len: usize,
    validity_buffer: Option<BufferHandle>,
    null_count: i64,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<(ArrowArray, SyncEvent)> {
    let canonical_elements = elements.execute_cuda(ctx).await?;
    let Canonical::Primitive(elements) = canonical_elements else {
        vortex_bail!(
            "cannot export non-contiguous device-resident ListViewArray with {} child: GPU child rebuild only supports primitive children",
            canonical_elements.dtype()
        );
    };

    let (offsets_buffer, rebuilt_elements) = rebuild_primitive_list_view_child(
        elements,
        "primitive child",
        offsets_ptype,
        offsets_buffer,
        sizes_ptype,
        sizes_buffer,
        len,
        ctx,
    )
    .await?;

    export_list_layout(
        rebuilt_elements.into_array(),
        len,
        validity_buffer,
        null_count,
        offsets_buffer,
        ListChildExport::PreserveConcreteLayout,
        ctx,
    )
    .await
}

/// Gather a non-contiguous primitive child into list order and return new offsets and values.
#[expect(clippy::cognitive_complexity, clippy::too_many_arguments)]
async fn rebuild_primitive_list_view_child(
    elements: PrimitiveArray,
    child_name: &str,
    offsets_ptype: PType,
    offsets_buffer: BufferHandle,
    sizes_ptype: PType,
    sizes_buffer: BufferHandle,
    len: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<(BufferHandle, PrimitiveArray)> {
    let elements_len = elements.len();
    let PrimitiveDataParts {
        ptype,
        buffer,
        validity,
        ..
    } = elements.into_data_parts();

    vortex_ensure!(
        validity.execute_no_nulls(elements_len, ctx.execution_ctx())?,
        "cannot export non-contiguous device-resident ListViewArray with nullable {child_name}: GPU child validity rebuild is not implemented"
    );

    let values_buffer = ctx.ensure_on_device(buffer).await?;
    let (offsets_buffer, values_buffer) = match_each_integer_ptype!(offsets_ptype, |O| {
        match_each_integer_ptype!(sizes_ptype, |S| {
            rebuild_primitive_list_view_typed::<O, S>(
                offsets_buffer,
                sizes_buffer,
                values_buffer,
                elements_len,
                len,
                ptype,
                ctx,
            )
            .await
        })
    })?;

    Ok((
        offsets_buffer,
        PrimitiveArray::from_buffer_handle(values_buffer, ptype, validity),
    ))
}

/// Execute an integer array on CUDA and return its primitive type and device buffer.
async fn primitive_device_buffer(
    array: ArrayRef,
    name: &str,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<(PType, BufferHandle)> {
    let canonical = array.execute_cuda(ctx).await?;
    let Canonical::Primitive(primitive) = canonical else {
        vortex_bail!("{name} must be primitive, got {}", canonical.dtype());
    };

    let PrimitiveDataParts { ptype, buffer, .. } = primitive.into_data_parts();
    vortex_ensure!(ptype.is_int(), "{name} must have integer type, got {ptype}");

    Ok((ptype, ctx.ensure_on_device(buffer).await?))
}

/// Compute Arrow List offsets and report whether child values must be rebuilt.
async fn export_device_list_view_offsets_typed<O, S>(
    offsets: BufferHandle,
    sizes: BufferHandle,
    len: usize,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<DeviceListViewOffsets>
where
    O: NativePType + DeviceRepr + Send + Sync + 'static,
    S: NativePType + DeviceRepr + Send + Sync + 'static,
{
    let output_len = len + 1;
    let output = ctx.device_alloc::<i32>(output_len)?;

    let status = ctx
        .ensure_on_device(BufferHandle::new_host(
            Buffer::from(vec![0u32]).into_byte_buffer(),
        ))
        .await?;

    let offsets_view = offsets.cuda_view::<O>()?;
    let sizes_view = sizes.cuda_view::<S>()?;
    let status_view = status.cuda_view::<u32>()?;
    let list_len_u64 = len as u64;

    let kernel = ctx.load_function_with_suffixes(
        "list_view",
        &["offsets", &O::PTYPE.to_string(), &S::PTYPE.to_string()],
    )?;
    ctx.launch_kernel(&kernel, len, |args| {
        args.arg(&offsets_view)
            .arg(&sizes_view)
            .arg(&output)
            .arg(&status_view)
            .arg(&list_len_u64);
    })?;

    match Buffer::<u32>::from_byte_buffer(status.try_to_host()?.await?)[0] {
        0 => Ok(DeviceListViewOffsets::Contiguous(BufferHandle::new_device(
            Arc::new(CudaDeviceBuffer::new(output)),
        ))),
        1 => Ok(DeviceListViewOffsets::RequiresRebuild),
        2 => vortex_bail!(
            "cannot export device-resident ListViewArray as Arrow List: offsets exceed i32 range required by cuDF"
        ),
        status => vortex_bail!("unexpected list-view offsets kernel status {status}"),
    }
}
