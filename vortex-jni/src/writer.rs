// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use jni::JNIEnv;
use jni::objects::{JByteArray, JClass, JMap, JString};
use jni::sys::{JNI_TRUE, jboolean, jlong};

use crate::errors::{JNIError, try_or_throw};

/// Wrapper for a Vortex file writer
/// This is a placeholder implementation that collects Arrow batches
/// TODO: Implement actual Vortex file writing when array conversion is available
pub struct WriterWrapper {
    path: String,
    schema: Arc<Schema>,
    batches: Vec<RecordBatch>,
}

impl WriterWrapper {
    pub fn new(path: String, schema: Arc<Schema>) -> Result<Self, JNIError> {
        Ok(WriterWrapper {
            path,
            schema,
            batches: Vec::new(),
        })
    }

    pub fn write_batch(&mut self, batch: RecordBatch) -> Result<(), JNIError> {
        self.batches.push(batch);
        Ok(())
    }

    pub fn close(self) -> Result<(), JNIError> {
        // TODO: Convert Arrow batches to Vortex and write to file
        // For now, create an empty file as a placeholder
        File::create(&self.path).map_err(|e| {
            JNIError::Vortex(vortex::error::vortex_err!("Failed to create file: {}", e))
        })?;

        Ok(())
    }
}

/// Creates a new Vortex writer
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeWriterMethods_create<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    file_path: JString<'local>,
    schema_json: JString<'local>,
    _options: JMap<'local, 'local, 'local>,
) -> jlong {
    try_or_throw(&mut env, |env| {
        // Get the file path
        let path: String = env.get_string(&file_path)?.into();

        // Parse the Arrow schema from JSON
        // For now, create a simple schema placeholder
        let _schema_str: String = env.get_string(&schema_json)?.into();
        let schema = Arc::new(Schema::empty());

        // Create the writer
        let wrapper = WriterWrapper::new(path, schema)?;

        // Return the pointer
        Ok(Box::into_raw(Box::new(wrapper)) as jlong)
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
    try_or_throw(&mut env, |env| {
        // Get the writer
        let wrapper = unsafe { &mut *(writer_ptr as *mut WriterWrapper) };

        // Get the Arrow data bytes
        let _data = env.convert_byte_array(&arrow_data)?;

        // TODO: Parse Arrow IPC format and convert to RecordBatch
        // For now, create a placeholder empty batch
        let schema = wrapper.schema.clone();
        let batch = RecordBatch::new_empty(schema);
        wrapper.write_batch(batch)?;

        Ok(JNI_TRUE)
    })
}

/// Closes the writer
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeWriterMethods_close<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    writer_ptr: jlong,
) -> jboolean {
    try_or_throw(&mut env, |_env| {
        // Take ownership of the writer and close it
        let wrapper = unsafe { Box::from_raw(writer_ptr as *mut WriterWrapper) };
        wrapper.close()?;

        Ok(JNI_TRUE)
    })
}
