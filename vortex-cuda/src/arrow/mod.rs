// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module implements the Arrow C Device data interface extension for sharing GPU-resident
//! data.
//!
//! This is an extension to the Arrow C Data Interface.
//!
//! More documentation at <https://arrow.apache.org/docs/format/CDeviceDataInterface.html>

mod canonical;

use std::ffi::c_void;
use std::fmt::Debug;
use std::ptr;
use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Fields;
use arrow_schema::Schema;
use arrow_schema::ffi::FFI_ArrowSchema;
use async_trait::async_trait;
pub(crate) use canonical::CanonicalDeviceArrayExport;
use cudarc::driver::CudaEvent;
use cudarc::driver::CudaStream;
use cudarc::runtime::sys::cudaEvent_t;
use vortex::array::ArrayRef;
use vortex::array::arrow::ArrowSessionExt;
use vortex::array::buffer::BufferHandle;
use vortex::array::validity::Validity;
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
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

#[expect(
    unused,
    reason = "cuda_stream and cuda_buffers need to have deferred drop"
)]
pub(crate) struct PrivateData {
    /// Hold a reference to the CudaStream so that it stays alive even after CudaExecutionCtx
    /// has been dropped.
    pub(crate) cuda_stream: Arc<CudaStream>,
    /// The single boxed slice which owns all buffers that the Rust code allocated on the device.
    pub(crate) buffers: Box<[Option<BufferHandle>]>,
    /// Boxed slice of buffer pointers. We return a pointer to the start of this allocation over
    /// the interface, so we hold it here so the Box contents are not freed.
    pub(crate) buffer_ptrs: Box<[*const c_void]>,
    pub(crate) cuda_event: CudaEvent,
    pub(crate) cuda_event_ptr: cudaEvent_t,
    pub(crate) children: Box<[*mut ArrowArray]>,
}

impl PrivateData {
    pub(crate) fn new(
        buffers: Vec<Option<BufferHandle>>,
        children: Vec<ArrowArray>,
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

        // generate the synchronization event
        let cuda_event = ctx
            .stream()
            .record_event(None)
            .map_err(|_| vortex_err!("failed to create cudaEvent_t"))?;

        Ok(Box::new(Self {
            buffers,
            buffer_ptrs,
            cuda_stream: Arc::clone(ctx.stream()),
            children,
            cuda_event_ptr: cuda_event.cu_event().cast(),
            cuda_event,
        }))
    }

    pub(crate) fn sync_event(&mut self) -> SyncEvent {
        (&raw mut self.cuda_event_ptr).cast()
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
    async fn export_device_array(
        self,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<ArrowDeviceArray>;

    /// Export this array as an Arrow C Device array together with its matching Arrow C schema.
    ///
    /// Arrow arrays are not self-describing: consumers need both the [`ArrowDeviceArray`] and an
    /// Arrow schema to interpret the buffer layout. This helper derives the schema from the
    /// Vortex dtype using the session's Arrow conversion rules, adapts it to the physical layout
    /// produced by the device exporter, and returns it alongside the device array.
    ///
    /// Top-level struct arrays are exported as table-like Arrow schemas and struct-shaped device
    /// arrays. Top-level non-struct arrays are exported as column-shaped field schemas and
    /// column-shaped device arrays; this method does not wrap them in a single-field struct.
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
        let schema = arrow_schema_for_array(&self, ctx)?;
        let array = self.export_device_array(ctx).await?;
        Ok(ArrowDeviceArrayWithSchema { schema, array })
    }
}

/// Build the Arrow C schema that describes the device array exported for `array`.
///
/// Top-level Vortex structs are represented as Arrow schemas, which is the shape expected for
/// table-like consumers. Non-struct arrays are represented as a single Arrow field schema, matching
/// the column-shaped [`ArrowDeviceArray`] returned by [`DeviceArrayExt::export_device_array`].
fn arrow_schema_for_array(
    array: &ArrayRef,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<FFI_ArrowSchema> {
    let arrow = ctx.execution_ctx().session().arrow();
    match array.dtype() {
        DType::Struct(..) => Ok(FFI_ArrowSchema::try_from(&device_array_schema(
            &arrow.to_arrow_schema(array.dtype())?,
        ))?),
        _ => Ok(FFI_ArrowSchema::try_from(&device_array_field(
            &arrow.to_arrow_field("", array.dtype())?,
        ))?),
    }
}

/// Convert an Arrow schema to the form described by this device exporter.
///
/// The session-level Arrow conversion describes Vortex arrays as Arrow logical types. This pass is
/// the single place where those types are adapted to the physical layouts emitted by the CUDA
/// `ArrowDeviceArray` exporter, while preserving field names, nullability, and metadata.
fn device_array_schema(schema: &Schema) -> Schema {
    Schema::new_with_metadata(
        schema
            .fields()
            .iter()
            .map(|field| device_array_field(field.as_ref()))
            .collect::<Vec<_>>(),
        schema.metadata().clone(),
    )
}

/// Convert a single Arrow field to the form described by this device exporter.
fn device_array_field(field: &Field) -> Field {
    field
        .clone()
        .with_data_type(device_array_data_type(field.data_type()))
}

/// Convert an Arrow logical dtype to the form described by this device exporter.
///
/// This currently preserves Arrow view types such as `Utf8View` and `BinaryView`, because the CUDA
/// exporter emits Arrow view-layout buffers for Vortex string/binary arrays. Struct children are
/// adapted recursively.
fn device_array_data_type(data_type: &DataType) -> DataType {
    match data_type {
        DataType::Struct(fields) => DataType::Struct(Fields::from(
            fields
                .iter()
                .map(|field| device_array_field(field.as_ref()))
                .collect::<Vec<_>>(),
        )),
        _ => data_type.clone(),
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
}

/// Check that the validity buffer is empty and does not need to be copied over the device boundary.
pub(crate) fn check_validity_empty(validity: &Validity) -> VortexResult<()> {
    if let Validity::AllInvalid | Validity::Array(_) = validity {
        vortex_bail!("Exporting array with non-trivial validity not supported yet")
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::utf8_view(DataType::Utf8View)]
    #[case::binary_view(DataType::BinaryView)]
    #[case::unchanged_leaf(DataType::UInt32)]
    fn device_array_data_type_preserves_leafs(#[case] input: DataType) {
        assert_eq!(device_array_data_type(&input), input);
    }

    #[test]
    fn device_array_data_type_recurses_into_struct_fields() {
        let data_type = DataType::Struct(Fields::from(vec![
            Field::new("utf8", DataType::Utf8View, true),
            Field::new("binary", DataType::BinaryView, false),
            Field::new("int", DataType::Int32, false),
        ]));

        assert_eq!(
            device_array_data_type(&data_type),
            DataType::Struct(Fields::from(vec![
                Field::new("utf8", DataType::Utf8View, true),
                Field::new("binary", DataType::BinaryView, false),
                Field::new("int", DataType::Int32, false),
            ]))
        );
    }

    #[test]
    fn device_array_field_preserves_name_nullability_and_metadata() {
        let field = Field::new("strings", DataType::Utf8View, true).with_metadata(
            [("key".to_string(), "value".to_string())]
                .into_iter()
                .collect(),
        );

        assert_eq!(
            device_array_field(&field),
            Field::new("strings", DataType::Utf8View, true).with_metadata(field.metadata().clone())
        );
    }

    #[test]
    fn device_array_schema_preserves_schema_metadata() {
        let schema = Schema::new_with_metadata(
            vec![
                Field::new("strings", DataType::Utf8View, false),
                Field::new("binary", DataType::BinaryView, true),
            ],
            [("schema-key".to_string(), "schema-value".to_string())]
                .into_iter()
                .collect(),
        );

        assert_eq!(
            device_array_schema(&schema),
            Schema::new_with_metadata(
                vec![
                    Field::new("strings", DataType::Utf8View, false),
                    Field::new("binary", DataType::BinaryView, true),
                ],
                schema.metadata().clone(),
            )
        );
    }
}
