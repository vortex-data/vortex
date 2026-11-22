// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! JNI bindings for the Vortex file writer.
//!
//! Writes go through an in-flight queue of at most [`WRITE_CHANNEL_CAPACITY`] pending
//! batches on the same thread that drives the current-thread runtime.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use arrow_array::RecordBatch;
use arrow_array::StructArray;
use arrow_array::ffi::FFI_ArrowArray;
use arrow_array::ffi::FFI_ArrowSchema;
use arrow_data::layout;
use arrow_schema::DataType;
use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use async_fs::File;
use futures::SinkExt;
use futures::channel::mpsc;
use jni::Env;
use jni::EnvUnowned;
use jni::objects::JClass;
use jni::objects::JObject;
use jni::objects::JString;
use jni::objects::JValue;
use jni::sys::JNI_FALSE;
use jni::sys::JNI_TRUE;
use jni::sys::jboolean;
use jni::sys::jlong;
use jni::sys::jobject;
use object_store::ObjectStore;
use object_store::path::Path as ObjectStorePath;
use vortex::array::ArrayRef;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::buffer::ByteBuffer;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::dtype::DType;
use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex::file::CountingVortexWrite;
use vortex::file::DeferredMetadata;
use vortex::file::VortexWriteOptions;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex::file::WriteSummary;
use vortex::file::multi::parse_uri_or_path;
use vortex::io::VortexWrite;
use vortex::io::compat::Compat;
use vortex::io::object_store::ObjectStoreWrite;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::Task;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::BufferedBytesTracker;
use vortex::session::VortexSession;
use vortex::utils::aliases::hash_map::HashMap;
use vortex_arrow::ArrowSessionExt;

use crate::RUNTIME;
use crate::buffer::exported_buffer_addresses;
use crate::buffer::import_buffers;
use crate::errors::JNIError;
use crate::errors::try_or_throw;
use crate::file::extract_metadata;
use crate::file::extract_properties;
use crate::io::JavaWrite;
use crate::object_store::make_object_store;
use crate::session::session_ref;
use crate::statistics::checked_jlong;
use crate::statistics::column_statistics_to_java;
use crate::statistics::footer_column_statistics;

/// Capacity of the in-flight write queue. Small on purpose so that back-pressure from
/// the writer is felt on the Java thread producing batches.
const WRITE_CHANNEL_CAPACITY: usize = 4;

enum ResolvedStore {
    ObjectStore(Arc<dyn ObjectStore>, ObjectStorePath),
    Path(PathBuf),
}

fn resolve_store(
    url_or_path: &str,
    properties: &HashMap<String, String>,
) -> VortexResult<ResolvedStore> {
    let url = parse_uri_or_path(url_or_path)?;
    if url.scheme() == "file" {
        let path = url
            .to_file_path()
            .map_err(|_| vortex_err!("invalid file URL: {url_or_path}"))?;
        Ok(ResolvedStore::Path(path))
    } else {
        let (store, path) = make_object_store(&url, properties)?;
        Ok(ResolvedStore::ObjectStore(store, path))
    }
}

/// Write options for `session`: its defaults, or the compact strategy, which trades read throughput
/// for file size as Python's `VortexWriteOptions.compact()` does.
fn session_write_options(session: &VortexSession, compact: bool) -> VortexWriteOptions {
    let options = session.write_options();
    if !compact {
        return options;
    }
    let compressor = BtrBlocksCompressorBuilder::from_session(session).with_compact();
    options.with_strategy(
        WriteStrategyBuilder::from_session(session)
            .with_btrblocks_builder(compressor)
            .build(),
    )
}

/// Native writer holding a write-task handle and a sender that Java pushes batches into.
pub struct NativeWriter {
    handle: Option<Task<VortexResult<WriteSummary>>>,
    session: VortexSession,
    arrow_schema: SchemaRef,
    write_schema: DType,
    bytes_written: Arc<AtomicU64>,
    buffered_bytes: BufferedBytesTracker,
    deferred_metadata: DeferredMetadata,
    sender: mpsc::Sender<VortexResult<ArrayRef>>,
}

impl NativeWriter {
    /// Spawn the write task into `write` and return the writer that sends batches to it.
    fn start<W: VortexWrite + Unpin + Send + 'static>(
        session: &VortexSession,
        arrow_schema: SchemaRef,
        write_schema: DType,
        write_options: VortexWriteOptions,
        write: W,
    ) -> Self {
        let (sender, rx) = mpsc::channel(WRITE_CHANNEL_CAPACITY);
        let stream = ArrayStreamAdapter::new(write_schema.clone(), rx);

        let buffered_bytes = write_options.buffered_bytes_tracker();
        let deferred_metadata = write_options.deferred_metadata();

        let mut write = CountingVortexWrite::new(write);
        let bytes_written = write.counter();
        let handle = session.handle().spawn(async move {
            let summary = write_options.write(&mut write, stream).await?;
            write.shutdown().await?;
            Ok(summary)
        });

        Self {
            handle: Some(handle),
            session: session.clone(),
            arrow_schema,
            write_schema,
            bytes_written,
            buffered_bytes,
            deferred_metadata,
            sender,
        }
    }

    pub fn into_raw(self: Box<Self>) -> jlong {
        Box::into_raw(self) as jlong
    }

    /// SAFETY: pointer must have been returned by [`Self::into_raw`].
    pub unsafe fn from_raw(pointer: jlong) -> Box<Self> {
        unsafe { Box::from_raw(pointer as *mut Self) }
    }

    /// SAFETY: pointer must have been returned by [`Self::into_raw`].
    pub unsafe fn from_ptr<'a>(pointer: jlong) -> &'a Self {
        debug_assert!(pointer != 0, "null writer pointer");
        unsafe { &*(pointer as *const Self) }
    }

    fn write_record_batch(&self, batch: RecordBatch) -> VortexResult<()> {
        let vortex_batch = self
            .session
            .arrow()
            .from_arrow_record_batch(batch, self.arrow_schema.as_ref())?;
        if !vortex_batch.dtype().eq(&self.write_schema) {
            return Err(vortex_err!(
                "write schema mismatch: expected {}, got {}",
                self.write_schema,
                vortex_batch.dtype()
            ));
        }
        let mut sender = self.sender.clone();
        RUNTIME
            .block_on(async move { sender.send(Ok(vortex_batch)).await })
            .map_err(|e| vortex_err!("failed to send batch: {e}"))
    }

    fn bytes_written(&self) -> u64 {
        self.bytes_written.load(Ordering::Relaxed)
    }

    fn buffered_bytes(&self) -> u64 {
        self.buffered_bytes.buffered_bytes()
    }

    /// Close the input and wait for the write task. The task adds `metadata` to the footer.
    fn close(mut self, metadata: HashMap<String, ByteBuffer>) -> VortexResult<WriteSummary> {
        // The write task reads the metadata after the input ends, so add it first.
        self.deferred_metadata.extend(metadata);

        self.sender.disconnect();
        let handle = self
            .handle
            .take()
            .ok_or_else(|| vortex_err!("writer already closed"))?;
        RUNTIME.block_on(handle)
    }
}

fn write_summary_to_java<'local>(
    env: &mut Env<'local>,
    summary: &WriteSummary,
) -> Result<JObject<'local>, JNIError> {
    let statistics = footer_column_statistics(summary.footer());
    let columns = column_statistics_to_java(env, summary.footer(), statistics)?;

    Ok(env.new_object(
        jni::jni_str!("dev/vortex/api/VortexWriteSummary"),
        jni::jni_sig!("(JJ[Ldev/vortex/api/VortexColumnStatistics;)V"),
        &[
            JValue::Long(checked_jlong(summary.size(), "file size")?),
            JValue::Long(checked_jlong(summary.row_count(), "row count")?),
            JValue::Object(columns.as_ref()),
        ],
    )?)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeWriter_create(
    mut env: EnvUnowned,
    _class: JClass,
    session_ptr: jlong,
    uri: JString,
    arrow_schema_addr: jlong,
    options: JObject,
    metadata: JObject,
    compact: jboolean,
) -> jlong {
    try_or_throw(&mut env, |env| {
        if session_ptr == 0 {
            throw_runtime!("null session pointer");
        }
        if arrow_schema_addr == 0 {
            throw_runtime!("null arrow schema address");
        }
        let session = unsafe { session_ref(session_ptr) };

        // Owned, so dropping it releases the exporter's copy as the C data interface requires.
        let ffi_schema =
            unsafe { FFI_ArrowSchema::from_raw(arrow_schema_addr as *mut FFI_ArrowSchema) };
        let arrow_schema = Arc::new(Schema::try_from(&ffi_schema)?);
        let write_schema = session.arrow().from_arrow_schema(arrow_schema.as_ref())?;

        let file_path: String = uri.try_to_string(env)?;
        let properties: HashMap<String, String> = extract_properties(env, &options)?;
        let metadata = extract_metadata(env, &metadata)?;
        let resolved = resolve_store(&file_path, &properties)?;
        let write_options =
            session_write_options(session, compact).with_metadata_segments(metadata);
        // The same check runs inside `write`, but only once the write task is under way, where
        // it would surface as an opaque send failure on the first batch.
        write_options.validate_metadata()?;

        let writer = match resolved {
            ResolvedStore::Path(path) => {
                let file = RUNTIME.block_on(async {
                    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                        async_fs::create_dir_all(parent).await?;
                    }
                    Ok::<_, VortexError>(File::create(path).await?)
                })?;
                NativeWriter::start(session, arrow_schema, write_schema, write_options, file)
            }
            ResolvedStore::ObjectStore(store, path) => {
                let object_write =
                    RUNTIME.block_on(ObjectStoreWrite::new(Arc::new(Compat::new(store)), &path))?;
                NativeWriter::start(
                    session,
                    arrow_schema,
                    write_schema,
                    write_options,
                    object_write,
                )
            }
        };

        Ok(Box::new(writer).into_raw())
    })
}

/// Create a writer that streams the file into a caller-provided
/// `dev.vortex.io.NativeWritable` instead of a native storage client.
///
/// Bytes are pushed through blocking `write`/`flush` upcalls on the runtime thread
/// driving the write task. The Java caller owns the underlying stream and must close
/// it after `NativeWriter.close` returns; the native side only flushes.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeWriter_createStream(
    mut env: EnvUnowned,
    _class: JClass,
    session_ptr: jlong,
    writable: JObject,
    arrow_schema_addr: jlong,
    metadata: JObject,
    compact: jboolean,
) -> jlong {
    try_or_throw(&mut env, |env| {
        if session_ptr == 0 {
            throw_runtime!("null session pointer");
        }
        if arrow_schema_addr == 0 {
            throw_runtime!("null arrow schema address");
        }
        if writable.is_null() {
            throw_runtime!("null writable");
        }
        let session = unsafe { session_ref(session_ptr) };

        // Owned, so dropping it releases the exporter's copy as the C data interface requires.
        let ffi_schema =
            unsafe { FFI_ArrowSchema::from_raw(arrow_schema_addr as *mut FFI_ArrowSchema) };
        let arrow_schema = Arc::new(Schema::try_from(&ffi_schema)?);
        let write_schema = session.arrow().from_arrow_schema(arrow_schema.as_ref())?;

        let metadata = extract_metadata(env, &metadata)?;
        let vm = env.get_java_vm()?;
        let writable = Arc::new(env.new_global_ref(&writable)?);
        let write_options =
            session_write_options(session, compact).with_metadata_segments(metadata);
        // See the note in `create`: validate before the write task can start.
        write_options.validate_metadata()?;

        let writer = NativeWriter::start(
            session,
            arrow_schema,
            write_schema,
            write_options,
            JavaWrite::new(vm, writable),
        );

        Ok(Box::new(writer).into_raw())
    })
}

/// Write a batch to the Vortex file directly from Arrow C Data Interface pointers.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeWriter_writeBatch(
    mut env: EnvUnowned,
    _class: JClass,
    writer_ptr: jlong,
    arrow_array_addr: jlong,
    arrow_schema_addr: jlong,
) -> jboolean {
    if writer_ptr <= 0 {
        return JNI_FALSE;
    }

    try_or_throw(&mut env, |_env| {
        let writer = unsafe { NativeWriter::from_ptr(writer_ptr) };

        let ffi_array =
            unsafe { FFI_ArrowArray::from_raw(arrow_array_addr as *mut FFI_ArrowArray) };
        // Take the schema as well as the array. As the consumer, native code must release both, and dropping the
        // owned schema does that; borrowing it would leak the exporter's copy on every batch.
        let ffi_schema =
            unsafe { FFI_ArrowSchema::from_raw(arrow_schema_addr as *mut FFI_ArrowSchema) };

        let array_data = unsafe { arrow_array::ffi::from_ffi(ffi_array, &ffi_schema) }
            .map_err(|e| JNIError::Vortex(vortex_err!("failed to import Arrow FFI data: {e}")))?;

        let batch = RecordBatch::from(StructArray::from(array_data));
        writer.write_record_batch(batch)?;
        Ok(JNI_TRUE)
    })
}

/// Check that the buffer, child and dictionary counts of `array` agree with `data_type`.
///
/// The Arrow import does not validate, so a mismatch can panic or read out of bounds. This check
/// cannot find a different value width, for example `Int64` in the place of `Int32`.
fn check_ffi_layout(array: &FFI_ArrowArray, data_type: &DataType) -> VortexResult<()> {
    let expected = layout(data_type);
    let fixed_buffers = expected.buffers.len() + usize::from(expected.can_contain_null_mask);

    // View types have a variable count of data buffers.
    if expected.variadic {
        vortex_ensure!(
            array.num_buffers() > fixed_buffers,
            "Arrow array for {data_type} has {} buffers, expected more than {fixed_buffers}",
            array.num_buffers()
        );
    } else {
        vortex_ensure!(
            array.num_buffers() == fixed_buffers,
            "Arrow array for {data_type} has {} buffers, expected {fixed_buffers}",
            array.num_buffers()
        );
    }

    let children: Vec<&DataType> = match data_type {
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::FixedSizeList(field, _)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::Map(field, _) => vec![field.data_type()],
        DataType::Struct(fields) => fields.iter().map(|field| field.data_type()).collect(),
        DataType::Union(fields, _) => fields.iter().map(|(_, field)| field.data_type()).collect(),
        DataType::RunEndEncoded(run_ends, values) => {
            vec![run_ends.data_type(), values.data_type()]
        }
        _ => Vec::new(),
    };
    vortex_ensure!(
        array.num_children() == children.len(),
        "Arrow array for {data_type} has {} children, expected {}",
        array.num_children(),
        children.len()
    );

    for (index, child_type) in children.into_iter().enumerate() {
        check_ffi_layout(array.child(index), child_type)?;
    }

    match (data_type, array.dictionary()) {
        (DataType::Dictionary(_, value_type), Some(dictionary)) => {
            check_ffi_layout(dictionary, value_type)
        }
        (DataType::Dictionary(..), None) => {
            Err(vortex_err!("Arrow array for {data_type} has no dictionary"))
        }
        (_, Some(_)) => Err(vortex_err!(
            "Arrow array for {data_type} has a dictionary, expected none"
        )),
        (_, None) => Ok(()),
    }
}

/// Write a batch from an Arrow C Data Interface array that agrees with the writer schema.
///
/// Native code takes the array and releases it, also when the call fails.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeWriter_writeArray(
    mut env: EnvUnowned,
    _class: JClass,
    writer_ptr: jlong,
    arrow_array_addr: jlong,
) -> jboolean {
    if writer_ptr <= 0 {
        return JNI_FALSE;
    }

    try_or_throw(&mut env, |_env| {
        if arrow_array_addr == 0 {
            throw_runtime!("null arrow array address");
        }
        let writer = unsafe { NativeWriter::from_ptr(writer_ptr) };

        // Take the array first, so that each path below releases it.
        let ffi_array =
            unsafe { FFI_ArrowArray::from_raw(arrow_array_addr as *mut FFI_ArrowArray) };
        let dtype = DataType::Struct(writer.arrow_schema.fields().clone());
        check_ffi_layout(&ffi_array, &dtype)?;

        // SAFETY: `check_ffi_layout` passed, and the caller guarantees the value types.
        let array_data = unsafe { arrow_array::ffi::from_ffi_and_data_type(ffi_array, dtype) }
            .map_err(|e| JNIError::Vortex(vortex_err!("failed to import Arrow FFI data: {e}")))?;

        let batch = RecordBatch::from(StructArray::from(array_data));
        writer.write_record_batch(batch)?;
        Ok(JNI_TRUE)
    })
}

/// Import a Java Arrow batch with per-allocation ownership instead of a shared batch owner.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeWriter_writeBatchBuffers(
    mut env: EnvUnowned,
    _class: JClass,
    writer_ptr: jlong,
    arrow_array_addr: jlong,
    buffers: JObject,
) -> jboolean {
    if writer_ptr <= 0 {
        return JNI_FALSE;
    }
    try_or_throw(&mut env, |env| {
        // SAFETY: Java validates the root schema against the writer's schema and holds the
        // writer monitor for this call. Moving the array clears its release callback.
        let writer = unsafe { NativeWriter::from_ptr(writer_ptr) };
        let array = unsafe { FFI_ArrowArray::from_raw(arrow_array_addr as *mut FFI_ArrowArray) };
        let exported_addresses = exported_buffer_addresses(&array);
        let dtype = DataType::Struct(writer.arrow_schema.fields().clone());
        // SAFETY: the Java export keeps all buffers alive through the import, and the
        // caller validated its schema before exporting the array.
        let data = unsafe { arrow_array::ffi::from_ffi_and_data_type(array, dtype) }
            .map_err(|e| JNIError::Vortex(vortex_err!("failed to import Arrow FFI data: {e}")))?;
        let data = import_buffers(env, data, &exported_addresses, buffers)?;
        writer.write_record_batch(RecordBatch::from(StructArray::from(data)))?;
        Ok(JNI_TRUE)
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeWriter_bytesWritten(
    mut env: EnvUnowned,
    _class: JClass,
    writer_ptr: jlong,
) -> jlong {
    if writer_ptr <= 0 {
        return -1;
    }

    try_or_throw(&mut env, |_env| {
        let writer = unsafe { NativeWriter::from_ptr(writer_ptr) };
        Ok(checked_jlong(writer.bytes_written(), "bytes written")?)
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeWriter_bufferedBytes(
    mut env: EnvUnowned,
    _class: JClass,
    writer_ptr: jlong,
) -> jlong {
    if writer_ptr <= 0 {
        return -1;
    }

    try_or_throw(&mut env, |_env| {
        let writer = unsafe { NativeWriter::from_ptr(writer_ptr) };
        Ok(checked_jlong(writer.buffered_bytes(), "buffered bytes")?)
    })
}

/// Finish the writer and return its summary.
///
/// `metadata` is a nullable `Map<String, byte[]>` that merges with the create-time segments.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeWriter_finish(
    mut env: EnvUnowned,
    _class: JClass,
    writer_ptr: jlong,
    metadata: JObject,
) -> jobject {
    if writer_ptr <= 0 {
        return JObject::null().into_raw();
    }
    let writer = unsafe { NativeWriter::from_raw(writer_ptr) };
    try_or_throw(&mut env, |env| {
        let metadata = extract_metadata(env, &metadata)?;
        let summary = writer.close(metadata)?;
        Ok(write_summary_to_java(env, &summary)?.into_raw())
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeWriter_close(
    mut env: EnvUnowned,
    _class: JClass,
    writer_ptr: jlong,
) {
    if writer_ptr <= 0 {
        return;
    }
    let writer = unsafe { NativeWriter::from_raw(writer_ptr) };
    try_or_throw(&mut env, |_env| {
        writer.close(HashMap::new())?;
        Ok(())
    });
}
