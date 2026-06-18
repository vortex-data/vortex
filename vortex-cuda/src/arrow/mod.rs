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
use std::sync::LazyLock;

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
use vortex::io::runtime::current::CurrentThreadWorkerPool;
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

// POSIX EIO for Arrow stream producer/export failures.
const ARROW_STREAM_EIO: c_int = 5;
// POSIX EINVAL for invalid Arrow stream callback arguments or released streams.
const ARROW_STREAM_EINVAL: c_int = 22;

static DEVICE_STREAM_RUNTIME: LazyLock<CurrentThreadRuntime> =
    LazyLock::new(CurrentThreadRuntime::new);
static DEVICE_STREAM_WORKER_POOL: LazyLock<CurrentThreadWorkerPool> = LazyLock::new(|| {
    let pool = DEVICE_STREAM_RUNTIME.new_pool();
    pool.set_workers_to_available_parallelism();
    pool
});

/// Return the shared runtime used to drive Vortex streams for Arrow Device export.
fn device_stream_runtime() -> &'static CurrentThreadRuntime {
    LazyLock::force(&DEVICE_STREAM_WORKER_POOL);
    &DEVICE_STREAM_RUNTIME
}

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

    /// Convert a Vortex dtype into a stream schema when no batch is available.
    ///
    /// This uses only the logical dtype, so it can differ from a non-empty stream's first-batch
    /// schema for encodings the dtype does not capture: a dictionary column reports a plain field
    /// here but `DataType::Dictionary` once a concrete batch is seen. This is harmless because an
    /// empty stream carries no data.
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

    /// Store the last stream error and return the Arrow callback error code.
    ///
    /// Interior NUL bytes are replaced so `get_last_error` is never null while a non-zero status
    /// is reported.
    fn set_error(&mut self, error: impl ToString) -> c_int {
        let message = error.to_string().replace('\0', " ");
        self.last_error = Some(CString::new(message).unwrap_or_default());
        ARROW_STREAM_EIO
    }

    /// Return the stream schema, exporting the first batch to derive it if needed.
    ///
    /// A first batch is held in `pending_array` so the following `get_next` returns it.
    fn ensure_schema(&mut self) -> VortexResult<&ArrowDeviceStreamSchema> {
        if self.schema.is_none() {
            match self.array_iter.next() {
                Some(array) => self.pending_array = Some(self.export_batch(array?)?),
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

    /// Export and return the next device batch, or `None` at end of stream.
    fn next_array(&mut self) -> VortexResult<Option<ArrowDeviceArray>> {
        if let Some(array) = self.pending_array.take() {
            return Ok(Some(array));
        }

        match self.array_iter.next() {
            Some(array) => self.export_batch(array?).map(Some),
            None => Ok(None),
        }
    }

    /// Export one Vortex batch as a device array, validating it against the stream.
    fn export_batch(&mut self, array: ArrayRef) -> VortexResult<ArrowDeviceArray> {
        vortex_ensure!(
            array.dtype() == &self.dtype,
            "stream batch dtype changed from {} to {}",
            self.dtype,
            array.dtype()
        );

        let ArrowDeviceArrayWithSchema {
            mut schema,
            mut array,
        } = device_stream_runtime()
            .block_on(array.export_device_array_with_schema(&mut self.ctx))?;

        // Release the schema we no longer need, and on any failure the array we will not return.
        let checked = self.check_batch(&schema, &array);
        release_schema(&mut schema);
        if let Err(error) = checked {
            release_device_array(&mut array);
            return Err(error);
        }
        Ok(array)
    }

    /// Check that a freshly exported batch matches the stream schema and CUDA device.
    ///
    /// The caller still owns `schema` and `array` and is responsible for releasing them on error.
    /// This method only commits `self.schema` after the batch is accepted, so a rejected first batch
    /// never becomes the schema later reported by `get_schema`.
    fn check_batch(
        &mut self,
        schema: &FFI_ArrowSchema,
        array: &ArrowDeviceArray,
    ) -> VortexResult<()> {
        vortex_ensure!(
            array.device_type == ARROW_DEVICE_CUDA,
            "stream batch exported on non-CUDA device type {}",
            array.device_type
        );
        vortex_ensure!(
            array.device_id == self.device_id,
            "stream batch moved from CUDA device {} to {}",
            self.device_id,
            array.device_id
        );

        // Commit the schema only after the batch is fully accepted, so a rejected first batch
        // never becomes the schema later reported by `get_schema`.
        let batch_schema = ArrowDeviceStreamSchema::from_ffi(schema, &self.dtype)?;
        match &self.schema {
            Some(stream_schema) => {
                vortex_ensure!(
                    stream_schema == &batch_schema,
                    "stream batch Arrow schema changed from {:?} to {:?}",
                    stream_schema,
                    batch_schema
                );
            }
            None => self.schema = Some(batch_schema),
        }
        Ok(())
    }
}

impl Drop for DeviceArrayStreamPrivateData {
    /// Release a first batch if `get_schema` exported it and `get_next` never returned it.
    fn drop(&mut self) {
        if let Some(mut array) = self.pending_array.take() {
            release_device_array(&mut array);
        }
    }
}

/// Extension trait for exporting a Vortex array stream as an Arrow C Device stream.
pub trait DeviceArrayStreamExt {
    /// Export this stream as an [`ArrowDeviceArrayStream`].
    ///
    /// Batches are exported through one persistent [`CudaExecutionCtx`]. The stream records that
    /// context's CUDA device at construction time, and each `get_next` verifies that the produced
    /// [`ArrowDeviceArray`] is CUDA-resident on that same device. The returned C stream owns the
    /// Vortex stream and must be released through its embedded `release` callback.
    ///
    /// Per the Arrow C stream contract, drive the returned stream from a single thread; its
    /// callbacks must not be invoked concurrently.
    fn export_device_array_stream(
        self,
        session: &VortexSession,
    ) -> VortexResult<ArrowDeviceArrayStream>;
}

impl DeviceArrayStreamExt for SendableArrayStream {
    /// Drive this stream on the shared Arrow Device stream runtime and export it.
    fn export_device_array_stream(
        self,
        session: &VortexSession,
    ) -> VortexResult<ArrowDeviceArrayStream> {
        let dtype = self.dtype().clone();
        let ctx = crate::CudaSession::create_execution_ctx(session)?;
        let array_iter = Box::new(device_stream_runtime().block_on_stream(self));
        Ok(device_array_stream(array_iter, dtype, ctx))
    }
}

/// Build the Arrow C Device stream that owns `array_iter` and exports its batches through `ctx`.
fn device_array_stream(
    array_iter: ArrayStreamIterator,
    dtype: DType,
    ctx: CudaExecutionCtx,
) -> ArrowDeviceArrayStream {
    let private_data = Box::new(DeviceArrayStreamPrivateData {
        device_id: ctx.stream().context().ordinal() as i64,
        array_iter,
        ctx,
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

/// Return the private stream state for a live Arrow device stream.
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
        array: ArrowArray::empty(),
        device_id,
        device_type: ARROW_DEVICE_CUDA,
        sync_event: ptr::null_mut(),
        reserved: Default::default(),
    }
}

/// Release an Arrow C schema if it is live.
fn release_schema(schema: &mut FFI_ArrowSchema) {
    if let Some(release) = schema.release {
        unsafe { release(schema) };
    }
}

/// Release an Arrow device array if it is live.
fn release_device_array(array: &mut ArrowDeviceArray) {
    if let Some(release) = array.array.release {
        unsafe { release(&raw mut array.array) };
    }
}

/// Run a stream callback body and convert errors or panics into Arrow status codes.
fn device_stream_callback(
    state: &mut DeviceArrayStreamPrivateData,
    panic_message: &'static str,
    callback: impl FnOnce(&mut DeviceArrayStreamPrivateData) -> VortexResult<()>,
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| callback(state)));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(err)) => state.set_error(err),
        Err(_) => state.set_error(panic_message),
    }
}

/// Write the stream's Arrow schema, initializing it from the first batch if necessary.
unsafe extern "C" fn device_stream_get_schema(
    stream: *mut ArrowDeviceArrayStream,
    out: *mut ArrowSchema,
) -> c_int {
    let Some(state) = (unsafe { device_stream_private_data(stream) }) else {
        return ARROW_STREAM_EINVAL;
    };
    state.clear_error();

    if out.is_null() {
        return state.set_error("null ArrowSchema output");
    }

    fn body(state: &mut DeviceArrayStreamPrivateData, out: *mut ArrowSchema) -> VortexResult<()> {
        let schema = state.ensure_schema()?.to_ffi()?;
        unsafe { ptr::write(out.cast::<FFI_ArrowSchema>(), schema) };
        Ok(())
    }

    device_stream_callback(
        state,
        "panic in ArrowDeviceArrayStream::get_schema",
        |state| body(state, out),
    )
}

/// Write the next exported Arrow device batch, or a released array at end of stream.
unsafe extern "C" fn device_stream_get_next(
    stream: *mut ArrowDeviceArrayStream,
    out: *mut ArrowDeviceArray,
) -> c_int {
    let Some(state) = (unsafe { device_stream_private_data(stream) }) else {
        return ARROW_STREAM_EINVAL;
    };
    state.clear_error();

    if out.is_null() {
        return state.set_error("null ArrowDeviceArray output");
    }

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

    if !stream_ref.private_data.is_null() {
        unsafe {
            drop(Box::from_raw(
                stream_ref
                    .private_data
                    .cast::<DeviceArrayStreamPrivateData>(),
            ));
        }
        stream_ref.private_data = ptr::null_mut();
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
    use arrow_schema::DataType;
    use arrow_schema::ffi::FFI_ArrowSchema;
    use vortex::VortexSessionDefault;
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::stream::ArrayStreamExt;
    use vortex::error::VortexResult;
    use vortex::error::vortex_err;
    use vortex::session::VortexSession;
    use vortex_cuda_macros::test as cuda_test;

    use crate::CudaSession;
    use crate::arrow::ARROW_DEVICE_CUDA;
    use crate::arrow::ArrowArray;
    use crate::arrow::ArrowDeviceArray;
    use crate::arrow::ArrowSchema;
    use crate::arrow::DeviceArrayStreamExt;

    /// Release an Arrow C schema in stream tests if it is live.
    unsafe fn release_schema(schema: &mut FFI_ArrowSchema) {
        if let Some(release) = schema.release {
            unsafe { release(schema) };
        }
    }

    /// Release an Arrow device array in stream tests if it is live.
    unsafe fn release_device_array(array: &mut ArrowDeviceArray) {
        if let Some(release) = array.array.release {
            unsafe { release(&raw mut array.array) };
        }
    }

    /// Create a zeroed placeholder Arrow device array for callback outputs.
    fn empty_device_array() -> ArrowDeviceArray {
        ArrowDeviceArray {
            array: ArrowArray::empty(),
            device_id: 0,
            device_type: 0,
            sync_event: std::ptr::null_mut(),
            reserved: [0; 3],
        }
    }

    /// Verify schema, batch, EOS, and idempotent release stream behavior.
    #[cuda_test]
    fn test_export_device_array_stream_schema_next_eos_release() -> VortexResult<()> {
        let session = VortexSession::default().with_some(CudaSession::try_default()?);
        let array = PrimitiveArray::from_iter(0u32..5).into_array();
        let stream = array.to_array_stream().boxed();
        let mut device_stream = stream.export_device_array_stream(&session)?;
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
        let mut first_batch = empty_device_array();
        let status = unsafe { get_next(&raw mut device_stream, &raw mut first_batch) };
        assert_eq!(status, 0);
        assert_eq!(first_batch.device_type, ARROW_DEVICE_CUDA);
        assert_eq!(first_batch.array.length, 5);
        assert!(first_batch.array.release.is_some());

        let mut eos = empty_device_array();
        let status = unsafe { get_next(&raw mut device_stream, &raw mut eos) };
        assert_eq!(status, 0);
        assert_eq!(eos.device_type, ARROW_DEVICE_CUDA);
        assert!(eos.array.release.is_none());

        unsafe {
            release_device_array(&mut first_batch);
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
}
