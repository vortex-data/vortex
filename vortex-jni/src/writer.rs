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
use vortex::aggregate_fn::AggregateFnVTableExt;
use vortex::aggregate_fn::fns::is_sorted::IsSorted;
use vortex::aggregate_fn::fns::is_sorted::IsSortedOptions;
use vortex::array::ArrayRef;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::buffer::ByteBuffer;
use vortex::dtype::DType;
use vortex::editions::ComponentKind;
use vortex::editions::EditionSessionExt;
use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::file::CountingVortexWrite;
use vortex::file::VortexWriteOptions;
use vortex::file::WriteOptionsSessionExt;
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

/// Native writer holding a write-task handle and a sender that Java pushes batches into.
pub struct NativeWriter {
    handle: Option<Task<VortexResult<WriteSummary>>>,
    session: VortexSession,
    arrow_schema: SchemaRef,
    write_schema: DType,
    bytes_written: Arc<AtomicU64>,
    buffered_bytes: BufferedBytesTracker,
    sender: mpsc::Sender<VortexResult<ArrayRef>>,
}

impl NativeWriter {
    pub fn new(
        session: VortexSession,
        arrow_schema: SchemaRef,
        write_schema: DType,
        bytes_written: Arc<AtomicU64>,
        buffered_bytes: BufferedBytesTracker,
        handle: Task<VortexResult<WriteSummary>>,
        sender: mpsc::Sender<VortexResult<ArrayRef>>,
    ) -> Self {
        Self {
            handle: Some(handle),
            session,
            arrow_schema,
            write_schema,
            bytes_written,
            buffered_bytes,
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

    fn close(mut self) -> VortexResult<WriteSummary> {
        self.sender.disconnect();
        let handle = self
            .handle
            .take()
            .ok_or_else(|| vortex_err!("writer already closed"))?;
        RUNTIME.block_on(handle)
    }
}

/// The options of every JNI write. Besides the default file statistics, a writer records whether
/// each column is sorted across the file, which lets readers report the order to query engines,
/// when the enabled editions permit that aggregate.
fn jni_write_options(
    session: &VortexSession,
    metadata: HashMap<String, ByteBuffer>,
) -> VortexWriteOptions {
    let is_sorted = IsSorted.bind(IsSortedOptions { strict: false });
    let options = session.write_options();
    let options = if session
        .enabled_component_ids(ComponentKind::Aggregate)
        .contains(&is_sorted.id())
    {
        options
            .with_additional_file_statistics(vec![is_sorted])
            // `IsSorted` has no legacy statistics slot, so only the nested statistics keep it.
            .include_nested_statistics()
    } else {
        options
    };

    options.with_metadata_segments(metadata)
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
        let (tx, rx) = mpsc::channel(WRITE_CHANNEL_CAPACITY);
        let stream = ArrayStreamAdapter::new(write_schema.clone(), rx);
        let write_options = jni_write_options(session, metadata);
        // The same check runs inside `write`, but only once the write task is under way, where
        // it would surface as an opaque send failure on the first batch.
        write_options.validate_metadata()?;
        let buffered_bytes = write_options.buffered_bytes_tracker();

        let (bytes_written, handle) = match resolved {
            ResolvedStore::Path(path) => {
                let file = RUNTIME.block_on(async {
                    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                        async_fs::create_dir_all(parent).await?;
                    }
                    Ok::<_, VortexError>(File::create(path).await?)
                })?;
                let mut write = CountingVortexWrite::new(file);
                let bytes_written = write.counter();
                let handle = session.handle().spawn(async move {
                    let summary = write_options.write(&mut write, stream).await?;
                    write.shutdown().await?;
                    Ok(summary)
                });
                (bytes_written, handle)
            }
            ResolvedStore::ObjectStore(store, path) => {
                let object_write =
                    RUNTIME.block_on(ObjectStoreWrite::new(Arc::new(Compat::new(store)), &path))?;
                let mut write = CountingVortexWrite::new(object_write);
                let bytes_written = write.counter();
                let handle = session.handle().spawn(async move {
                    let summary = write_options.write(&mut write, stream).await?;
                    write.shutdown().await?;
                    Ok(summary)
                });
                (bytes_written, handle)
            }
        };

        Ok(Box::new(NativeWriter::new(
            session.clone(),
            arrow_schema,
            write_schema,
            bytes_written,
            buffered_bytes,
            handle,
            tx,
        ))
        .into_raw())
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
        let (tx, rx) = mpsc::channel(WRITE_CHANNEL_CAPACITY);
        let stream = ArrayStreamAdapter::new(write_schema.clone(), rx);
        let write_options = jni_write_options(session, metadata);
        // See the note in `create`: validate before the write task can start.
        write_options.validate_metadata()?;
        let buffered_bytes = write_options.buffered_bytes_tracker();

        let mut write = CountingVortexWrite::new(JavaWrite::new(vm, writable));
        let bytes_written = write.counter();
        let handle = session.handle().spawn(async move {
            let summary = write_options.write(&mut write, stream).await?;
            write.shutdown().await?;
            Ok(summary)
        });

        Ok(Box::new(NativeWriter::new(
            session.clone(),
            arrow_schema,
            write_schema,
            bytes_written,
            buffered_bytes,
            handle,
            tx,
        ))
        .into_raw())
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

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeWriter_finish(
    mut env: EnvUnowned,
    _class: JClass,
    writer_ptr: jlong,
) -> jobject {
    if writer_ptr <= 0 {
        return JObject::null().into_raw();
    }
    let writer = unsafe { NativeWriter::from_raw(writer_ptr) };
    try_or_throw(&mut env, |env| {
        let summary = writer.close()?;
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
        writer.close()?;
        Ok(())
    });
}
