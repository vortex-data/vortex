// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io::Cursor;

use arrow_array::RecordBatch;
use arrow_ipc::reader::StreamReader;
use futures::SinkExt;
use futures::channel::mpsc;
use jni::JNIEnv;
use jni::objects::{JByteArray, JClass, JObject, JString};
use jni::sys::{JNI_FALSE, JNI_TRUE, jboolean, jlong};
use object_store::path::Path;
use tokio::task::JoinHandle;
use url::Url;
use vortex::arrow::FromArrowArray;
use vortex::dtype::DType;
use vortex::error::{VortexResult, vortex_bail, vortex_err};
use vortex::file::{VortexWriteOptions, WriteStrategyBuilder};
use vortex::stream::ArrayStreamAdapter;
use vortex::utils::aliases::hash_map::HashMap;
use vortex::{Array, ArrayRef};

use crate::errors::{JNIError, try_or_throw};
use crate::object_store::make_object_store;
use crate::{block_on, get_process_task_executor, spawn};

/// Native writer around a file writer.
pub struct NativeWriter {
    /// Handle to the write operation, launched onto the global runtime.
    /// It will unwrap to () if the write succeeded, or to a VortexError with the reason if it fails.
    handle: Option<JoinHandle<VortexResult<()>>>,
    /// Vortex schema for all batches.
    write_schema: DType,
    /// Ingest arrays into the handle.
    sender: mpsc::Sender<VortexResult<ArrayRef>>,
}

impl NativeWriter {
    /// Create a new writer which tracks a write task and a join handle instead.
    pub fn new(
        write_schema: DType,
        handle: JoinHandle<VortexResult<()>>,
        sender: mpsc::Sender<VortexResult<ArrayRef>>,
    ) -> Self {
        Self {
            handle: Some(handle),
            write_schema,
            sender,
        }
    }

    pub fn into_raw(self: Box<Self>) -> jlong {
        Box::into_raw(self) as jlong
    }

    pub unsafe fn from_raw(pointer: jlong) -> Box<Self> {
        unsafe { Box::from_raw(pointer as *mut Self) }
    }

    #[allow(clippy::expect_used)]
    pub unsafe fn from_ptr<'a>(pointer: jlong) -> &'a Self {
        unsafe {
            (pointer as *const Self)
                .as_ref()
                .expect("Pointer should never be null")
        }
    }

    /// Write an Arrow record batch to the writer stream.
    pub fn write_record_batch(&self, batch: RecordBatch) -> VortexResult<()> {
        // We do not allow top-level nulls
        let vortex_batch = ArrayRef::from_arrow(batch, false);

        // Validate schema conforms
        if !vortex_batch.dtype().eq(&self.write_schema) {
            vortex_bail!(
                "write_record_batch schema mismatch: expected {}, batch {}",
                self.write_schema,
                vortex_batch.dtype()
            );
        }

        let mut sender = self.sender.clone();

        block_on(
            "NativeWriter::write_batch",
            Box::pin(async move {
                sender
                    .send(Ok(vortex_batch))
                    .await
                    .map_err(|_| vortex_err!("write_record_batch: send failure"))
            }),
        )?;

        Ok(())
    }

    /// Close and block until all data is flushed and the write has completed.
    ///
    /// Flushes all external values
    pub fn close(mut self) -> VortexResult<()> {
        // Drop the writer.
        self.sender.disconnect();

        // Close the stream. This takes ownership of the inner and blocks on it.
        let handle = self.handle.take().ok_or_else(|| {
            vortex_err!("JoinHandle absent, closing an already closed NativeWriter")
        })?;

        // Join the write handle, which completes after all chunks have been flushed and the file
        // stream is closed.

        block_on("NativeWriter::close", handle)
            .map_err(|join| vortex_err!("NativeWriter::close: error joining write task: {join}"))?
    }
}

/// Create a new file writer at the provided URI with some configurable options.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeWriterMethods_create(
    mut env: JNIEnv,
    _class: JClass,
    uri: JString,
    dtype_ptr: jlong,
    options: JObject,
) -> jlong {
    try_or_throw(&mut env, |env| {
        let write_schema = unsafe { *Box::from_raw(dtype_ptr as *mut DType) };
        let file_path: String = env.get_string(&uri)?.into();

        let Ok(url) = Url::parse(&file_path) else {
            throw_runtime!("Invalid URL: {file_path}");
        };

        let mut properties: HashMap<String, String> = HashMap::new();

        if !options.is_null() {
            let opts = env.get_map(&options)?;
            let mut iterator = opts.iter(env)?;
            while let Some((key, val)) = iterator.next(env)? {
                let key = env.auto_local(key);
                let val = env.auto_local(val);
                let key_str = env.get_string(key.as_ref().into())?;
                let val_str = env.get_string(val.as_ref().into())?;
                properties.insert(key_str.into(), val_str.into());
            }
        }

        let start = std::time::Instant::now();
        let duration = std::time::Instant::now().duration_since(start);
        log::debug!("make_object_store latency = {duration:?}");

        // Peg a stream task upfront that all has the same schema.
        let path = Path::from_url_path(url.path())
            .map_err(|_| vortex_err!("invalid object_store Path {}", url.path()))?;

        // Create a new task to hold the sender
        let (tx, rx) = mpsc::channel(32);
        let w = ArrayStreamAdapter::new(write_schema.clone(), rx);

        let (store, _scheme) = make_object_store(&url, &properties)?;
        let write_handle = spawn(async move {
            VortexWriteOptions::default()
                .with_strategy(
                    WriteStrategyBuilder::new()
                        .with_executor(get_process_task_executor())
                        .build(),
                )
                .write_object_store(&store, &path, w)
                .await
        });

        Ok(Box::new(NativeWriter::new(write_schema, write_handle, tx)).into_raw())
    })
}

/// Writes a batch to the Vortex file
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeWriterMethods_writeBatch<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    writer_ptr: jlong,
    arrow_data: JByteArray<'local>,
) -> jboolean {
    // Validate pointer before using it
    if writer_ptr <= 0 {
        return JNI_FALSE;
    }

    try_or_throw(&mut env, |env| {
        // Get the writer
        let writer = unsafe { NativeWriter::from_ptr(writer_ptr) };

        // Get the Arrow IPC data bytes
        let data = env.convert_byte_array(&arrow_data)?;

        // Parse the Arrow IPC stream to extract RecordBatches
        let cursor = Cursor::new(data);
        let mut reader = StreamReader::try_new(cursor, None).map_err(|e| {
            JNIError::Vortex(vortex::error::vortex_err!(
                "Failed to parse Arrow IPC data: {}",
                e
            ))
        })?;

        // Read all batches from the IPC stream
        for batch_result in &mut reader {
            let batch = batch_result
                .map_err(|e| JNIError::Vortex(vortex_err!("Failed to read RecordBatch: {e}")))?;
            writer.write_record_batch(batch)?;
        }

        Ok(JNI_TRUE)
    })
}

/// Closes the writer
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeWriterMethods_close<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    writer_ptr: jlong,
) {
    let writer = unsafe { NativeWriter::from_raw(writer_ptr) };

    try_or_throw(&mut env, |_env| {
        writer.close()?;
        Ok(())
    });
}
