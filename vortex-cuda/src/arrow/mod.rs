// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module implements the Arrow C Device data interface extension for sharing GPU-resident
//! data.
//!
//! This is an extension to the Arrow C Data Interface.
//!
//! More documentation at <https://arrow.apache.org/docs/format/CDeviceDataInterface.html>

mod canonical;
mod list_view;

use std::ffi::c_void;
use std::fmt::Debug;
use std::ptr;
use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use arrow_schema::ffi::FFI_ArrowSchema;
use async_trait::async_trait;
pub(crate) use canonical::CanonicalDeviceArrayExport;
use cudarc::driver::CudaEvent;
use cudarc::driver::CudaStream;
use cudarc::runtime::sys::cudaEvent_t;
use vortex::array::ArrayRef;
use vortex::array::arrays::Dict;
use vortex::array::arrays::FixedSizeList;
use vortex::array::arrays::List;
use vortex::array::arrays::ListView;
use vortex::array::arrays::Struct;
use vortex::array::arrays::dict::DictArraySlotsExt;
use vortex::array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex::array::arrays::list::ListArrayExt;
use vortex::array::arrays::listview::ListViewArrayExt;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::array::arrow::ArrowSessionExt;
use vortex::array::buffer::BufferHandle;
use vortex::dtype::DType;
use vortex::dtype::DecimalDType;
use vortex::dtype::DecimalType;
use vortex::dtype::PType;
use vortex::dtype::StructFields;
use vortex::error::VortexResult;
use vortex::error::vortex_err;

use crate::CudaBufferExt;
use crate::CudaExecutionCtx;

mod arrow_c_abi {
    #![allow(dead_code)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(non_upper_case_globals)]
    #![allow(clippy::absolute_paths)]

    include!(concat!(env!("OUT_DIR"), "/arrow_c_abi.rs"));
}

pub use arrow_c_abi::ArrowArray;
pub use arrow_c_abi::ArrowDeviceArray;
pub use arrow_c_abi::ArrowDeviceType;

#[cfg(feature = "_test-harness")]
#[doc(hidden)]
pub mod test_harness {
    use vortex::array::buffer::BufferHandle;
    use vortex::error::VortexResult;

    use crate::CudaExecutionCtx;
    use crate::arrow::canonical::repack_arrow_validity_buffer as repack_arrow_validity_buffer_impl;

    /// Repack a validity bitmap into Arrow's byte-addressed bitmap layout on the active stream.
    pub fn repack_arrow_validity_buffer(
        input_buffer: &BufferHandle,
        input_offset: usize,
        len: usize,
        arrow_offset: usize,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<BufferHandle> {
        repack_arrow_validity_buffer_impl(input_buffer, input_offset, len, arrow_offset, ctx)
    }
}

/// CUDA device memory.
pub const ARROW_DEVICE_CUDA: ArrowDeviceType = arrow_c_abi::ARROW_DEVICE_CUDA as ArrowDeviceType;

/// A pointer to a device-specific synchronization event, or null if synchronization is not needed.
pub type SyncEvent = *mut c_void;

impl ArrowArray {
    pub fn empty() -> Self {
        Self {
            length: 0,
            null_count: 0,
            offset: 0,
            n_buffers: 0,
            n_children: 0,
            buffers: ptr::null_mut(),
            children: ptr::null_mut(),
            dictionary: ptr::null_mut(),
            release: None,
            private_data: ptr::null_mut(),
        }
    }
}

unsafe impl Send for ArrowArray {}
unsafe impl Sync for ArrowArray {}

pub(crate) struct PrivateData {
    /// Hold a reference to the CudaStream so that it stays alive even after CudaExecutionCtx
    /// has been dropped.
    pub(crate) cuda_stream: Arc<CudaStream>,
    /// The single boxed slice which owns all buffers that the Rust code allocated on the device.
    #[allow(dead_code, reason = "buffers are retained for deferred drop")]
    pub(crate) buffers: Box<[Option<BufferHandle>]>,
    /// Boxed slice of buffer pointers. We return a pointer to the start of this allocation over
    /// the interface, so we hold it here so the Box contents are not freed.
    pub(crate) buffer_ptrs: Box<[*const c_void]>,
    pub(crate) export_event: CudaEvent,
    pub(crate) export_event_ptr: cudaEvent_t,
    pub(crate) children: Box<[*mut ArrowArray]>,
    pub(crate) dictionary: *mut ArrowArray,
}

impl PrivateData {
    /// Create private data for arrays that own buffers and child arrays but no dictionary.
    pub(crate) fn new(
        buffers: Vec<Option<BufferHandle>>,
        children: Vec<ArrowArray>,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Box<Self>> {
        Self::new_with_dictionary(buffers, children, None, ctx)
    }

    /// Create private data and optionally own an Arrow dictionary child.
    pub(crate) fn new_with_dictionary(
        buffers: Vec<Option<BufferHandle>>,
        children: Vec<ArrowArray>,
        dictionary: Option<ArrowArray>,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Box<Self>> {
        let buffers = buffers.into_boxed_slice();
        let buffer_ptrs: Box<[*const c_void]> = buffers
            .iter()
            .map(|buf| {
                match buf {
                    None => {
                        // null pointer
                        Ok(ptr::null())
                    }
                    Some(handle) => usize::try_from(handle.cuda_device_ptr()?)
                        .map(|ptr| ptr as *const c_void)
                        .map_err(|_| vortex_err!("CUDA device pointer does not fit in usize")),
                }
            })
            .collect::<VortexResult<Vec<_>>>()?
            .into_boxed_slice();

        let children = children
            .into_iter()
            .map(|array| Box::into_raw(Box::new(array)))
            .collect::<Box<[_]>>();

        let export_event = ctx
            .stream()
            .record_event(None)
            .map_err(|_| vortex_err!("failed to create CUDA export event"))?;

        let dictionary = dictionary
            .map(|array| Box::into_raw(Box::new(array)))
            .unwrap_or(ptr::null_mut());

        Ok(Box::new(Self {
            buffers,
            buffer_ptrs,
            cuda_stream: Arc::clone(ctx.stream()),
            children,
            dictionary,
            export_event_ptr: export_event.cu_event().cast(),
            export_event,
        }))
    }

    /// Return a stable pointer to the recorded CUDA export event handle.
    pub(crate) fn sync_event(&mut self) -> SyncEvent {
        (&raw mut self.export_event_ptr).cast()
    }
}

/// A Vortex array exported as an Arrow schema and Arrow Device array pair.
#[derive(Debug)]
pub struct ArrowDeviceArrayWithSchema {
    /// The Arrow C Data schema describing [`Self::array`].
    ///
    /// For top-level Vortex struct arrays this is an Arrow schema (a struct with one child per
    /// field). For top-level non-struct arrays this is a single Arrow field schema matching the
    /// column-shaped device array.
    pub schema: FFI_ArrowSchema,
    /// The Arrow C Device array containing the exported device-resident buffers.
    pub array: ArrowDeviceArray,
}

#[async_trait]
pub trait DeviceArrayExt {
    /// Export this array as an Arrow C Device array.
    ///
    /// The returned array owns any device buffers allocated during export. Call the embedded
    /// Arrow release callback when the consumer is done with the array.
    ///
    /// Arrow arrays are not self-describing, so callers that use this method directly must provide
    /// a matching schema out-of-band. Prefer [`Self::export_device_array_with_schema`] unless a
    /// consumer already has the CUDA export schema.
    async fn export_device_array(
        self,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<ArrowDeviceArray>;

    /// Export this array as an Arrow C Device array together with its matching Arrow C schema.
    ///
    /// Arrow arrays are not self-describing: consumers need both the [`ArrowDeviceArray`] and an
    /// Arrow schema to interpret the buffer layout. This helper derives the schema that matches the
    /// CUDA device export layout and returns it alongside the device array.
    ///
    /// Top-level struct arrays are exported as table-like Arrow schemas and struct-shaped device
    /// arrays. Top-level non-struct arrays are exported as column-shaped field schemas and
    /// column-shaped device arrays; this method does not wrap them in a single-field struct.
    ///
    /// Decimal exports use the Arrow decimal width implied by precision; storage wider than that
    /// width is rejected rather than narrowed on device.
    async fn export_device_array_with_schema(
        self,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<ArrowDeviceArrayWithSchema>;
}

#[async_trait]
impl DeviceArrayExt for ArrayRef {
    async fn export_device_array(
        self,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<ArrowDeviceArray> {
        let exporter = Arc::clone(ctx.exporter());
        exporter.export_device_array(self, ctx).await
    }

    async fn export_device_array_with_schema(
        self,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<ArrowDeviceArrayWithSchema> {
        let exporter = Arc::clone(ctx.exporter());
        exporter.export_device_array_with_schema(self, ctx).await
    }
}

/// Build the Arrow C schema that describes the exported device array.
pub(crate) fn arrow_schema_for_array(
    array: &ArrayRef,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<FFI_ArrowSchema> {
    if let Some(struct_array) = array.as_opt::<Struct>() {
        return Ok(FFI_ArrowSchema::try_from(Schema::new(
            arrow_device_export_struct_fields_for_array(
                struct_array.names().iter(),
                struct_array.iter_unmasked_fields(),
                ctx,
            )?,
        ))?);
    }

    Ok(FFI_ArrowSchema::try_from(
        arrow_device_export_field_for_array("", array, ctx)?,
    )?)
}

/// Build struct fields from a logical dtype when no concrete child arrays are available.
fn arrow_device_export_struct_fields(
    struct_dtype: &StructFields,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Vec<Field>> {
    let mut fields = Vec::with_capacity(struct_dtype.nfields());
    for (field_name, field_dtype) in struct_dtype.names().iter().zip(struct_dtype.fields()) {
        fields.push(arrow_device_export_field(
            field_name.as_ref(),
            &field_dtype,
            ctx,
        )?);
    }
    Ok(fields)
}

/// Build struct fields from concrete arrays so nested encodings like dictionaries are preserved.
fn arrow_device_export_struct_fields_for_array<'a>(
    names: impl IntoIterator<Item = &'a vortex::dtype::FieldName>,
    fields: impl IntoIterator<Item = &'a ArrayRef>,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Vec<Field>> {
    names
        .into_iter()
        .zip(fields)
        .map(|(name, array)| arrow_device_export_field_for_array(name.as_ref(), array, ctx))
        .collect()
}

/// Build the Arrow field matching how this concrete array will be exported.
fn arrow_device_export_field_for_array(
    name: impl AsRef<str>,
    array: &ArrayRef,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Field> {
    let name = name.as_ref();

    if let Some(dict) = array.as_opt::<Dict>() {
        let codes_dtype = arrow_device_export_dictionary_codes_dtype(dict.codes().dtype())?;
        let codes_type = arrow_device_export_field("", &codes_dtype, ctx)?
            .data_type()
            .clone();
        let values_type = arrow_device_export_field_for_array("", dict.values(), ctx)?
            .data_type()
            .clone();
        return Ok(Field::new(
            name,
            DataType::Dictionary(Box::new(codes_type), Box::new(values_type)),
            array.dtype().is_nullable(),
        ));
    }

    if let Some(struct_array) = array.as_opt::<Struct>() {
        return Ok(Field::new(
            name,
            DataType::Struct(
                arrow_device_export_struct_fields_for_array(
                    struct_array.names().iter(),
                    struct_array.iter_unmasked_fields(),
                    ctx,
                )?
                .into(),
            ),
            array.dtype().is_nullable(),
        ));
    }

    if let Some(list) = array.as_opt::<List>() {
        let element = arrow_device_export_field_for_array(
            Field::LIST_FIELD_DEFAULT_NAME,
            list.elements(),
            ctx,
        )?;
        return Ok(Field::new_list(name, element, array.dtype().is_nullable()));
    }

    if let Some(list) = array.as_opt::<ListView>() {
        let element = arrow_device_export_field_for_array(
            Field::LIST_FIELD_DEFAULT_NAME,
            list.elements(),
            ctx,
        )?;
        return Ok(Field::new_list(name, element, array.dtype().is_nullable()));
    }

    if let Some(list) = array.as_opt::<FixedSizeList>() {
        let element = arrow_device_export_field_for_array(
            Field::LIST_FIELD_DEFAULT_NAME,
            list.elements(),
            ctx,
        )?;
        return Ok(Field::new_list(name, element, array.dtype().is_nullable()));
    }

    arrow_device_export_field(name, &arrow_device_export_dtype(array.dtype()), ctx)
}

/// Return the signed dictionary code dtype used by Arrow Device export.
fn arrow_device_export_dictionary_codes_dtype(codes_dtype: &DType) -> VortexResult<DType> {
    // cuDF's Arrow Device importer only accepts signed dictionary indices.
    let ptype = match codes_dtype.as_ptype() {
        PType::U8 => PType::I16,
        PType::U16 => PType::I32,
        PType::U32 | PType::U64 => PType::I64,
        ptype @ (PType::I8 | PType::I16 | PType::I32 | PType::I64) => ptype,
        ptype => return Err(vortex_err!("dictionary codes must be integer, got {ptype}")),
    };

    Ok(DType::Primitive(ptype, codes_dtype.nullability()))
}

/// Build the Arrow field for a dtype-only export schema fallback.
fn arrow_device_export_field(
    name: impl AsRef<str>,
    dtype: &DType,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Field> {
    let field = ctx
        .execution_ctx()
        .session()
        .arrow()
        .to_arrow_field(name.as_ref(), dtype)?;

    let data_type = match dtype {
        DType::Binary(_) => DataType::Binary,
        DType::Decimal(decimal_dtype, _) => arrow_device_export_decimal_data_type(*decimal_dtype),
        DType::Struct(struct_dtype, _) => {
            DataType::Struct(arrow_device_export_struct_fields(struct_dtype, ctx)?.into())
        }
        _ => return Ok(field),
    };

    Ok(
        Field::new(field.name().clone(), data_type, field.is_nullable())
            .with_metadata(field.metadata().clone()),
    )
}

/// Return the Arrow decimal type with the device-export physical width.
fn arrow_device_export_decimal_data_type(decimal_dtype: DecimalDType) -> DataType {
    match cuda_decimal_value_type(decimal_dtype) {
        DecimalType::I32 => DataType::Decimal32(decimal_dtype.precision(), decimal_dtype.scale()),
        DecimalType::I64 => DataType::Decimal64(decimal_dtype.precision(), decimal_dtype.scale()),
        DecimalType::I128 => DataType::Decimal128(decimal_dtype.precision(), decimal_dtype.scale()),
        DecimalType::I256 => DataType::Decimal256(decimal_dtype.precision(), decimal_dtype.scale()),
        decimal_type => unreachable!("unsupported decimal value type {decimal_type}"),
    }
}

/// Return the decimal storage width Arrow expects for a precision.
pub(crate) fn cuda_decimal_value_type(decimal_dtype: DecimalDType) -> DecimalType {
    match decimal_dtype.precision() {
        1..=9 => DecimalType::I32,
        10..=18 => DecimalType::I64,
        19..=38 => DecimalType::I128,
        39..=76 => DecimalType::I256,
        p => unreachable!("precision {p} is invalid for a DecimalDType"),
    }
}

/// Adapt Vortex logical dtypes to the Arrow Device layout this exporter emits.
fn arrow_device_export_dtype(dtype: &DType) -> DType {
    match dtype {
        DType::List(element, nullability) => {
            DType::List(Arc::new(arrow_device_export_dtype(element)), *nullability)
        }
        DType::FixedSizeList(element, _, nullability) => {
            DType::List(Arc::new(arrow_device_export_dtype(element)), *nullability)
        }
        DType::Struct(fields, nullability) => DType::Struct(
            StructFields::new(
                fields.names().clone(),
                fields
                    .fields()
                    .map(|dtype| arrow_device_export_dtype(&dtype))
                    .collect(),
            ),
            *nullability,
        ),
        dtype => dtype.clone(),
    }
}

/// A type that can convert a Vortex array into an [`ArrowDeviceArray`].
#[async_trait]
pub trait ExportDeviceArray: Debug + Send + Sync + 'static {
    /// Export a Vortex array as an [`ArrowDeviceArray`].
    ///
    /// The Arrow Device Array is part of the Arrow C Device data interface extension to the Arrow
    /// specification. It enables passing Vortex arrays to other processes that consume Arrow
    /// arrays, such as cudf.
    ///
    /// See <https://arrow.apache.org/docs/format/CDeviceDataInterface.html>.
    async fn export_device_array(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<ArrowDeviceArray>;

    /// Export a Vortex array as an [`ArrowDeviceArray`] with a matching Arrow C schema.
    async fn export_device_array_with_schema(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<ArrowDeviceArrayWithSchema> {
        let schema = arrow_schema_for_array(&array, ctx)?;
        let array = self.export_device_array(array, ctx).await?;
        Ok(ArrowDeviceArrayWithSchema { schema, array })
    }
}
