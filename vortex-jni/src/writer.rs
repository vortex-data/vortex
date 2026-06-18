// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! JNI bindings for the Vortex file writer.
//!
//! Writes go through an in-flight queue of at most [`WRITE_CHANNEL_CAPACITY`] pending
//! batches on the same thread that drives the current-thread runtime.

use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::StructArray;
use arrow_array::ffi::FFI_ArrowArray;
use arrow_array::ffi::FFI_ArrowSchema;
use arrow_schema::SchemaRef;
use async_fs::File;
use futures::SinkExt;
use futures::channel::mpsc;
use jni::EnvUnowned;
use jni::objects::JClass;
use jni::objects::JObject;
use jni::objects::JString;
use jni::sys::JNI_FALSE;
use jni::sys::JNI_TRUE;
use jni::sys::jboolean;
use jni::sys::jlong;
use object_store::ObjectStore;
use object_store::path::Path as ObjectStorePath;
use url::Url;
use vortex::array::ArrayRef;
use vortex::array::VTable;
use vortex::array::arrow::ArrowSessionExt;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::dtype::DType;
use vortex::dtype::Field as DTypeField;
use vortex::dtype::FieldPath;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex::file::WriteSummary;
use vortex::io::VortexWrite;
use vortex::io::compat::Compat;
use vortex::io::object_store::ObjectStoreWrite;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::Task;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex::utils::aliases::hash_map::HashMap;
use vortex_parquet_variant::ParquetVariant;

use crate::RUNTIME;
use crate::dtype::import_arrow_schema;
use crate::errors::JNIError;
use crate::errors::try_or_throw;
use crate::file::extract_properties;
use crate::object_store::make_object_store;
use crate::session::session_ref;

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
    match Url::parse(url_or_path) {
        Ok(url) if url.scheme() == "file" => {
            let path = url
                .to_file_path()
                .map_err(|_| vortex_err!("invalid file URL: {url_or_path}"))?;
            Ok(ResolvedStore::Path(path))
        }
        Ok(url) => {
            let path = ObjectStorePath::from_url_path(url.path())
                .map_err(|_| vortex_err!("invalid object_store path: {}", url.path()))?;
            let store = make_object_store(&url, properties)?;
            Ok(ResolvedStore::ObjectStore(store, path))
        }
        Err(_) => Ok(ResolvedStore::Path(PathBuf::from(url_or_path))),
    }
}

fn write_options_for_schema(
    session: &VortexSession,
    write_schema: &DType,
) -> vortex::file::VortexWriteOptions {
    let variant_paths = variant_field_paths(write_schema);
    if variant_paths.is_empty() {
        return session.write_options();
    }

    let mut allowed = vortex::file::ALLOWED_ENCODINGS.clone();
    allowed.insert(ParquetVariant.id());

    let strategy = WriteStrategyBuilder::default().with_allow_encodings(allowed);

    session.write_options().with_strategy(strategy.build())
}

fn variant_field_paths(dtype: &DType) -> Vec<FieldPath> {
    let mut paths = Vec::new();
    collect_variant_field_paths(dtype, FieldPath::root(), &mut paths);
    paths
}

fn collect_variant_field_paths(dtype: &DType, path: FieldPath, paths: &mut Vec<FieldPath>) {
    match dtype {
        DType::Variant(_) => paths.push(path),
        DType::Struct(fields, _) => {
            for (name, field_dtype) in fields.names().iter().zip(fields.fields()) {
                collect_variant_field_paths(
                    &field_dtype,
                    path.clone().push(DTypeField::from(name.clone())),
                    paths,
                );
            }
        }
        _ => {}
    }
}

/// Native writer holding a write-task handle and a sender that Java pushes batches into.
pub struct NativeWriter {
    handle: Option<Task<VortexResult<WriteSummary>>>,
    session: VortexSession,
    arrow_schema: SchemaRef,
    write_schema: DType,
    sender: mpsc::Sender<VortexResult<ArrayRef>>,
}

impl NativeWriter {
    pub fn new(
        session: VortexSession,
        arrow_schema: SchemaRef,
        write_schema: DType,
        handle: Task<VortexResult<WriteSummary>>,
        sender: mpsc::Sender<VortexResult<ArrayRef>>,
    ) -> Self {
        Self {
            handle: Some(handle),
            session,
            arrow_schema,
            write_schema,
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

    fn close(mut self) -> VortexResult<()> {
        self.sender.disconnect();
        let handle = self
            .handle
            .take()
            .ok_or_else(|| vortex_err!("writer already closed"))?;
        RUNTIME.block_on(async {
            handle.await?;
            VortexResult::Ok(())
        })
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeWriter_create(
    mut env: EnvUnowned,
    _class: JClass,
    session_ptr: jlong,
    uri: JString,
    arrow_schema_addr: jlong,
    options: JObject,
) -> jlong {
    try_or_throw(&mut env, |env| {
        if session_ptr == 0 {
            throw_runtime!("null session pointer");
        }
        if arrow_schema_addr == 0 {
            throw_runtime!("null arrow schema address");
        }
        let session = unsafe { session_ref(session_ptr) };

        let arrow_schema = Arc::new(import_arrow_schema(arrow_schema_addr)?);
        let write_schema = session.arrow().from_arrow_schema(arrow_schema.as_ref())?;

        let file_path: String = uri.try_to_string(env)?;
        let properties: HashMap<String, String> = extract_properties(env, &options)?;
        let resolved = resolve_store(&file_path, &properties)?;
        let (tx, rx) = mpsc::channel(WRITE_CHANNEL_CAPACITY);
        let stream = ArrayStreamAdapter::new(write_schema.clone(), rx);
        let write_options = write_options_for_schema(session, &write_schema);

        let handle = session.handle().spawn(async move {
            match resolved {
                ResolvedStore::Path(path) => {
                    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                        async_fs::create_dir_all(parent).await?;
                    }
                    let mut file = File::create(path).await?;
                    let summary = write_options.write(&mut file, stream).await?;
                    file.shutdown().await?;
                    Ok(summary)
                }
                ResolvedStore::ObjectStore(store, path) => {
                    let mut write =
                        ObjectStoreWrite::new(Arc::new(Compat::new(store)), &path).await?;
                    let summary = write_options.write(&mut write, stream).await?;
                    write.shutdown().await?;
                    Ok(summary)
                }
            }
        });

        Ok(Box::new(NativeWriter::new(
            session.clone(),
            arrow_schema,
            write_schema,
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
        let ffi_schema = unsafe { &*(arrow_schema_addr as *const FFI_ArrowSchema) };

        let array_data = unsafe { arrow_array::ffi::from_ffi(ffi_array, ffi_schema) }
            .map_err(|e| JNIError::Vortex(vortex_err!("failed to import Arrow FFI data: {e}")))?;

        let batch = RecordBatch::from(StructArray::from(array_data));
        writer.write_record_batch(batch)?;
        Ok(JNI_TRUE)
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
