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

use std::ffi::CString;
use std::ffi::c_char;
use std::ffi::c_int;
use std::ffi::c_void;
use std::fmt::Debug;
use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;
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
use vortex::array::stream::SendableArrayStream;
use vortex::dtype::DType;
use vortex::dtype::DecimalDType;
use vortex::dtype::DecimalType;
use vortex::dtype::PType;
use vortex::dtype::StructFields;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::session::VortexSession;

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
pub use arrow_c_abi::ArrowDeviceArrayStream;
pub use arrow_c_abi::ArrowDeviceType;
use arrow_c_abi::ArrowSchema;

#[cfg(feature = "_test-harness")]
#[doc(hidden)]
pub mod test_harness {
    pub use crate::arrow::canonical::count_arrow_validity_nulls;
    pub use crate::arrow::canonical::repack_arrow_validity_buffer;
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

impl ArrowDeviceArray {
    /// A zeroed device array: an empty Arrow array with no device. Used as a
    /// callback output placeholder and as the basis for the end-of-stream
    /// marker.
    fn empty() -> Self {
        Self {
            array: ArrowArray::empty(),
            device_id: 0,
            device_type: 0,
            sync_event: ptr::null_mut(),
            reserved: Default::default(),
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

/// POSIX EIO for Arrow stream producer/export failures.
const LIBC_EIO: c_int = 5;

/// POSIX EINVAL for invalid Arrow stream callback arguments or released streams.
const LIBC_EINVAL: c_int = 22;

#[derive(Debug, PartialEq)]
enum ArrowDeviceStreamSchema {
    Schema(Schema),
    Field(Field),
}

impl ArrowDeviceStreamSchema {
    /// Convert an Arrow C schema into the stream schema shape for `dtype`.
    fn from_ffi(schema: &FFI_ArrowSchema, dtype: &DType) -> VortexResult<Self> {
        if matches!(dtype, DType::Struct(..)) {
            Ok(Self::Schema(Schema::try_from(schema)?))
        } else {
            Ok(Self::Field(Field::try_from(schema)?))
        }
    }

    /// Convert a Vortex dtype into a stream schema when no stream array is available.
    ///
    /// This uses only the logical dtype, so it can differ from a non-empty stream's first-array
    /// schema for encodings the dtype does not capture: a dictionary column reports a plain field
    /// here but `DataType::Dictionary` once a concrete array is seen.
    fn from_dtype(dtype: &DType, ctx: &mut CudaExecutionCtx) -> VortexResult<Self> {
        let dtype = arrow_device_export_dtype(dtype);
        if let DType::Struct(struct_dtype, _) = &dtype {
            Ok(Self::Schema(Schema::new(
                arrow_device_export_struct_fields(struct_dtype, ctx)?,
            )))
        } else {
            Ok(Self::Field(arrow_device_export_field("", &dtype, ctx)?))
        }
    }

    /// Export this stream schema as an owned Arrow C schema.
    fn to_ffi(&self) -> VortexResult<FFI_ArrowSchema> {
        match self {
            Self::Schema(schema) => Ok(FFI_ArrowSchema::try_from(schema)?),
            Self::Field(field) => Ok(FFI_ArrowSchema::try_from(field)?),
        }
    }
}

type ArrayStreamIterator = Box<dyn Iterator<Item = VortexResult<ArrayRef>>>;

struct DeviceArrayStreamPrivateData {
    array_iter: ArrayStreamIterator,
    ctx: CudaExecutionCtx,
    /// Runtime used by Arrow stream callbacks to pull from `array_iter` and block on per-array
    /// CUDA exports. It must match the runtime that owns the underlying Vortex scan tasks so those
    /// tasks are polled while callbacks are producing arrays.
    runtime: CurrentThreadRuntime,
    dtype: DType,
    schema: Option<ArrowDeviceStreamSchema>,
    pending_array: Option<ArrowDeviceArray>,
    device_id: i64,
    last_error: Option<CString>,
}

impl DeviceArrayStreamPrivateData {
    /// Clear the last stream error before a new callback invocation.
    fn clear_error(&mut self) {
        self.last_error = None;
    }

    /// Store the last stream error and return the requested Arrow callback error code.
    ///
    /// Interior NUL bytes are replaced so `get_last_error` is never null while a non-zero status
    /// is reported.
    fn set_error(&mut self, error: impl ToString, code: c_int) -> c_int {
        let message = error.to_string().replace('\0', " ");
        self.last_error = Some(CString::new(message).unwrap_or_default());
        code
    }

    /// Return the stream schema, exporting the first stream array to derive it if needed.
    ///
    /// A first array is held in `pending_array` so the following `get_next` returns it.
    fn get_or_init_schema(&mut self) -> VortexResult<&ArrowDeviceStreamSchema> {
        if self.schema.is_none() {
            match self.array_iter.next() {
                Some(array) => self.pending_array = Some(self.export_stream_array(array?)?),
                None => {
                    self.schema = Some(ArrowDeviceStreamSchema::from_dtype(
                        &self.dtype,
                        &mut self.ctx,
                    )?);
                }
            }
        }

        self.schema
            .as_ref()
            .ok_or_else(|| vortex_err!("ArrowDeviceArrayStream schema was not initialized"))
    }

    /// Export and return the next Arrow device array, or `None` at end of stream.
    fn next_array(&mut self) -> VortexResult<Option<ArrowDeviceArray>> {
        if let Some(array) = self.pending_array.take() {
            return Ok(Some(array));
        }

        match self.array_iter.next() {
            Some(array) => self.export_stream_array(array?).map(Some),
            None => Ok(None),
        }
    }

    /// Export one array from the Vortex stream, validating it against the device stream.
    fn export_stream_array(&mut self, array: ArrayRef) -> VortexResult<ArrowDeviceArray> {
        vortex_ensure!(
            array.dtype() == &self.dtype,
            "stream array dtype changed from {} to {}",
            self.dtype,
            array.dtype()
        );

        let ArrowDeviceArrayWithSchema {
            schema: mut ffi_schema,
            array: mut device_array,
        } = self
            .runtime
            .block_on(array.export_device_array_with_schema(&mut self.ctx))?;

        // Release the schema we no longer need, and on failure release the array we will not
        // return.
        let checked = self.check_stream_array(&ffi_schema, &device_array);
        release_schema(&mut ffi_schema);
        let exported_schema = match checked {
            Ok(exported_schema) => exported_schema,
            Err(error) => {
                release_device_array(&mut device_array);
                return Err(error);
            }
        };
        if self.schema.is_none() {
            self.schema = Some(exported_schema);
        }
        Ok(device_array)
    }

    /// Check that a freshly exported device array matches the stream schema and CUDA device.
    fn check_stream_array(
        &self,
        ffi_schema: &FFI_ArrowSchema,
        device_array: &ArrowDeviceArray,
    ) -> VortexResult<ArrowDeviceStreamSchema> {
        vortex_ensure!(
            device_array.device_type == ARROW_DEVICE_CUDA,
            "stream array exported on non-CUDA device type {}",
            device_array.device_type
        );
        vortex_ensure!(
            device_array.device_id == self.device_id,
            "stream array moved from CUDA device {} to {}",
            self.device_id,
            device_array.device_id
        );

        let exported_schema = ArrowDeviceStreamSchema::from_ffi(ffi_schema, &self.dtype)?;
        if let Some(stream_schema) = &self.schema {
            vortex_ensure!(
                stream_schema == &exported_schema,
                "stream array Arrow schema changed from {:?} to {:?}; an Arrow C device stream \
                 requires every array to share one schema, so chunks must not vary their \
                 encoding (for example a dictionary-encoded chunk among plain chunks)",
                stream_schema,
                exported_schema
            );
        }
        Ok(exported_schema)
    }
}

impl Drop for DeviceArrayStreamPrivateData {
    /// Release a first stream array if `get_schema` exported it and `get_next` never returned it.
    fn drop(&mut self) {
        if let Some(mut array) = self.pending_array.take() {
            release_device_array(&mut array);
        }
    }
}

/// Extension trait for exporting a Vortex array stream as an Arrow Device stream.
pub trait DeviceArrayStreamExt {
    /// Export this stream as an [`ArrowDeviceArrayStream`].
    ///
    /// Arrays are exported by reusing one [`CudaExecutionCtx`], and every produced
    /// [`ArrowDeviceArray`] must remain on the CUDA device captured at stream construction. The
    /// returned [`ArrowDeviceArrayStream`] owns the Vortex stream and must be released through its
    /// embedded `release` callback.
    ///
    /// The Arrow Device stream contract requires all arrays to share the schema reported by
    /// `get_schema`. The schema is derived from the first array, or from the logical dtype
    /// for an empty stream. Chunks that export to different Arrow types are rejected mid-stream.
    ///
    /// Drive the returned stream from one thread. `runtime` must be the runtime that owns the
    /// underlying scan tasks and per-array exports.
    fn export_device_array_stream(
        self,
        session: &VortexSession,
        runtime: &CurrentThreadRuntime,
    ) -> VortexResult<ArrowDeviceArrayStream>;
}

impl DeviceArrayStreamExt for SendableArrayStream {
    /// Drive this stream on `runtime` and export it.
    fn export_device_array_stream(
        self,
        session: &VortexSession,
        runtime: &CurrentThreadRuntime,
    ) -> VortexResult<ArrowDeviceArrayStream> {
        let dtype = self.dtype().clone();
        let ctx = crate::CudaSession::create_execution_ctx(session)?;
        let array_iter = Box::new(runtime.block_on_stream(self));
        Ok(device_array_stream(array_iter, dtype, ctx, runtime.clone()))
    }
}

/// Build the Arrow Device stream that owns `array_iter` and exports its arrays through `ctx`.
fn device_array_stream(
    array_iter: ArrayStreamIterator,
    dtype: DType,
    ctx: CudaExecutionCtx,
    runtime: CurrentThreadRuntime,
) -> ArrowDeviceArrayStream {
    let private_data = Box::new(DeviceArrayStreamPrivateData {
        device_id: ctx.stream().context().ordinal() as i64,
        array_iter,
        ctx,
        runtime,
        dtype,
        schema: None,
        pending_array: None,
        last_error: None,
    });

    ArrowDeviceArrayStream {
        device_type: ARROW_DEVICE_CUDA,
        get_schema: Some(device_stream_get_schema),
        get_next: Some(device_stream_get_next),
        get_last_error: Some(device_stream_get_last_error),
        release: Some(device_stream_release),
        private_data: Box::into_raw(private_data).cast(),
    }
}

/// Returns the stream state stored in `private_data`.
unsafe fn device_stream_private_data<'a>(
    stream: *mut ArrowDeviceArrayStream,
) -> Option<&'a mut DeviceArrayStreamPrivateData> {
    let stream = unsafe { stream.as_mut()? };
    unsafe {
        stream
            .private_data
            .cast::<DeviceArrayStreamPrivateData>()
            .as_mut()
    }
}

/// Create the Arrow end-of-stream marker for the stream's CUDA device.
fn released_device_array(device_id: i64) -> ArrowDeviceArray {
    ArrowDeviceArray {
        device_id,
        device_type: ARROW_DEVICE_CUDA,
        ..ArrowDeviceArray::empty()
    }
}

/// Release an Arrow C schema if it is live.
pub fn release_schema(schema: &mut FFI_ArrowSchema) {
    if let Some(release) = schema.release {
        unsafe { release(schema) };
    }
}

/// Release an Arrow device array if it is live.
pub fn release_device_array(array: &mut ArrowDeviceArray) {
    if let Some(release) = array.array.release {
        unsafe { release(&raw mut array.array) };
    }
}

/// Runs an Arrow stream callback body.
///
/// Returns an Arrow callback status code and stores failures in `last_error`.
fn device_stream_callback(
    state: &mut DeviceArrayStreamPrivateData,
    panic_message: &'static str,
    callback: impl FnOnce(&mut DeviceArrayStreamPrivateData) -> VortexResult<()>,
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| callback(state)));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(err)) => state.set_error(err, LIBC_EIO),
        Err(_) => state.set_error(panic_message, LIBC_EIO),
    }
}

/// Write the stream's Arrow schema, initializing it from the first stream array if unset.
unsafe extern "C" fn device_stream_get_schema(
    stream: *mut ArrowDeviceArrayStream,
    out: *mut ArrowSchema,
) -> c_int {
    let Some(state) = (unsafe { device_stream_private_data(stream) }) else {
        return LIBC_EINVAL;
    };
    state.clear_error();

    if out.is_null() {
        return state.set_error("null ArrowSchema output", LIBC_EINVAL);
    }

    fn body(state: &mut DeviceArrayStreamPrivateData, out: *mut ArrowSchema) -> VortexResult<()> {
        let schema = state.get_or_init_schema()?.to_ffi()?;
        unsafe { ptr::write(out.cast::<FFI_ArrowSchema>(), schema) };
        Ok(())
    }

    device_stream_callback(
        state,
        "panic in ArrowDeviceArrayStream::get_schema",
        |state| body(state, out),
    )
}

/// Write the next exported Arrow device array, or a released array at end of stream.
unsafe extern "C" fn device_stream_get_next(
    stream: *mut ArrowDeviceArrayStream,
    out: *mut ArrowDeviceArray,
) -> c_int {
    let Some(state) = (unsafe { device_stream_private_data(stream) }) else {
        return LIBC_EINVAL;
    };
    state.clear_error();

    if out.is_null() {
        return state.set_error("null ArrowDeviceArray output", LIBC_EINVAL);
    }

    // Keep the fallible part in a local function so `device_stream_callback` handles callback
    // status and error reporting consistently.
    fn body(
        state: &mut DeviceArrayStreamPrivateData,
        out: *mut ArrowDeviceArray,
    ) -> VortexResult<()> {
        let array = state
            .next_array()?
            .unwrap_or_else(|| released_device_array(state.device_id));
        unsafe { ptr::write(out, array) };
        Ok(())
    }

    device_stream_callback(
        state,
        "panic in ArrowDeviceArrayStream::get_next",
        |state| body(state, out),
    )
}

/// Return the most recent callback error message, or null if no error is stored.
unsafe extern "C" fn device_stream_get_last_error(
    stream: *mut ArrowDeviceArrayStream,
) -> *const c_char {
    let Some(state) = (unsafe { device_stream_private_data(stream) }) else {
        return ptr::null();
    };

    state
        .last_error
        .as_ref()
        .map_or(ptr::null(), |error| error.as_ptr())
}

/// Free the stream state and null its callbacks. The null `release` makes a second call a no-op.
unsafe extern "C" fn device_stream_release(stream: *mut ArrowDeviceArrayStream) {
    let Some(stream_ref) = (unsafe { stream.as_mut() }) else {
        return;
    };
    if stream_ref.release.is_none() {
        return;
    }

    stream_ref.get_schema = None;
    stream_ref.get_next = None;
    stream_ref.get_last_error = None;
    stream_ref.release = None;

    let private_data = std::mem::replace(&mut stream_ref.private_data, ptr::null_mut());
    if !private_data.is_null() {
        drop(catch_unwind(AssertUnwindSafe(|| unsafe {
            drop(Box::from_raw(
                private_data.cast::<DeviceArrayStreamPrivateData>(),
            ));
        })));
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

#[cfg(test)]
mod tests {
    use std::ffi::CStr;
    use std::ptr;

    use arrow_schema::DataType;
    use arrow_schema::ffi::FFI_ArrowSchema;
    use futures::stream;
    use vortex::VortexSessionDefault;
    use vortex::array::ArrayRef;
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::stream::ArrayStreamAdapter;
    use vortex::array::stream::ArrayStreamExt;
    use vortex::dtype::DType;
    use vortex::dtype::Nullability;
    use vortex::dtype::PType;
    use vortex::error::VortexResult;
    use vortex::error::vortex_err;
    use vortex::io::runtime::current::CurrentThreadRuntime;
    use vortex::session::VortexSession;
    use vortex_cuda_macros::test as cuda_test;

    use crate::CudaSession;
    use crate::arrow::ARROW_DEVICE_CUDA;
    use crate::arrow::ArrowDeviceArray;
    use crate::arrow::ArrowDeviceArrayStream;
    use crate::arrow::ArrowSchema;
    use crate::arrow::DeviceArrayStreamExt;
    use crate::arrow::LIBC_EINVAL;
    use crate::arrow::release_device_array;
    use crate::arrow::release_schema;

    fn last_error(stream: &mut ArrowDeviceArrayStream) -> VortexResult<String> {
        let get_last_error = stream
            .get_last_error
            .ok_or_else(|| vortex_err!("stream missing get_last_error callback"))?;
        let error = unsafe { get_last_error(stream as *mut ArrowDeviceArrayStream) };
        Ok(if error.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned()
        })
    }

    #[cuda_test]
    fn test_export_device_array_stream_schema_next_eos_release() -> VortexResult<()> {
        let runtime = CurrentThreadRuntime::new();
        let session = VortexSession::default().with::<CudaSession>();
        let array = PrimitiveArray::from_iter(0u32..5).into_array();
        let stream = array.to_array_stream().boxed();
        let mut device_stream = stream.export_device_array_stream(&session, &runtime)?;
        assert_eq!(device_stream.device_type, ARROW_DEVICE_CUDA);

        let mut schema = FFI_ArrowSchema::empty();
        let get_schema = device_stream
            .get_schema
            .ok_or_else(|| vortex_err!("stream missing get_schema callback"))?;
        let status = unsafe {
            get_schema(
                &raw mut device_stream,
                (&raw mut schema).cast::<ArrowSchema>(),
            )
        };
        assert_eq!(status, 0);
        let field = arrow_schema::Field::try_from(&schema)?;
        assert_eq!(field.data_type(), &DataType::UInt32);

        let get_next = device_stream
            .get_next
            .ok_or_else(|| vortex_err!("stream missing get_next callback"))?;
        let mut first_array = ArrowDeviceArray::empty();
        let status = unsafe { get_next(&raw mut device_stream, &raw mut first_array) };
        assert_eq!(status, 0);
        assert_eq!(first_array.device_type, ARROW_DEVICE_CUDA);
        assert_eq!(first_array.array.length, 5);
        assert!(first_array.array.release.is_some());

        let mut eos = ArrowDeviceArray::empty();
        let status = unsafe { get_next(&raw mut device_stream, &raw mut eos) };
        assert_eq!(status, 0);
        assert_eq!(eos.device_type, ARROW_DEVICE_CUDA);
        assert!(eos.array.release.is_none());

        unsafe {
            release_device_array(&mut first_array);
            release_schema(&mut schema);
            let release = device_stream
                .release
                .ok_or_else(|| vortex_err!("stream missing release callback"))?;
            release(&raw mut device_stream);
            release(&raw mut device_stream);
        }
        assert!(device_stream.release.is_none());
        Ok(())
    }

    #[cuda_test]
    fn test_export_device_array_stream_empty_stream_schema_and_eos() -> VortexResult<()> {
        let runtime = CurrentThreadRuntime::new();
        let session = VortexSession::default().with::<CudaSession>();
        let dtype = DType::Primitive(PType::U32, Nullability::NonNullable);
        let stream =
            ArrayStreamAdapter::new(dtype, stream::empty::<VortexResult<ArrayRef>>()).boxed();
        let mut device_stream = stream.export_device_array_stream(&session, &runtime)?;

        let get_schema = device_stream
            .get_schema
            .ok_or_else(|| vortex_err!("stream missing get_schema callback"))?;
        let mut schema = FFI_ArrowSchema::empty();
        let status = unsafe {
            get_schema(
                &raw mut device_stream,
                (&raw mut schema).cast::<ArrowSchema>(),
            )
        };
        assert_eq!(status, 0);
        let field = arrow_schema::Field::try_from(&schema)?;
        assert_eq!(field.data_type(), &DataType::UInt32);

        let get_next = device_stream
            .get_next
            .ok_or_else(|| vortex_err!("stream missing get_next callback"))?;
        let mut eos = ArrowDeviceArray::empty();
        let status = unsafe { get_next(&raw mut device_stream, &raw mut eos) };
        assert_eq!(status, 0);
        assert!(eos.array.release.is_none());

        unsafe {
            release_schema(&mut schema);
            let release = device_stream
                .release
                .ok_or_else(|| vortex_err!("stream missing release callback"))?;
            release(&raw mut device_stream);
        }
        Ok(())
    }

    #[cuda_test]
    fn test_export_device_array_stream_null_outputs_report_error() -> VortexResult<()> {
        let runtime = CurrentThreadRuntime::new();
        let session = VortexSession::default().with::<CudaSession>();
        let array = PrimitiveArray::from_iter(0u32..5).into_array();
        let stream = array.to_array_stream().boxed();
        let mut device_stream = stream.export_device_array_stream(&session, &runtime)?;

        let get_schema = device_stream
            .get_schema
            .ok_or_else(|| vortex_err!("stream missing get_schema callback"))?;
        let status = unsafe { get_schema(&raw mut device_stream, ptr::null_mut()) };
        assert_eq!(status, LIBC_EINVAL);
        assert_eq!(last_error(&mut device_stream)?, "null ArrowSchema output");

        let get_next = device_stream
            .get_next
            .ok_or_else(|| vortex_err!("stream missing get_next callback"))?;
        let status = unsafe { get_next(&raw mut device_stream, ptr::null_mut()) };
        assert_eq!(status, LIBC_EINVAL);
        assert_eq!(
            last_error(&mut device_stream)?,
            "null ArrowDeviceArray output"
        );

        unsafe {
            let release = device_stream
                .release
                .ok_or_else(|| vortex_err!("stream missing release callback"))?;
            release(&raw mut device_stream);
        }
        Ok(())
    }
}
