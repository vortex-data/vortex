// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io::Cursor;

use arrow_array::RecordBatch;
use arrow_ipc::reader::StreamReader;
use jni::JNIEnv;
use jni::objects::{JByteArray, JClass, JObject, JString};
use jni::sys::{JNI_FALSE, JNI_TRUE, jboolean, jlong};
use vortex::arrays::ChunkedArray;
use vortex::arrow::FromArrowArray;
use vortex::file::VortexWriteOptions;
use vortex::{ArrayRef, IntoArray};

use crate::errors::{JNIError, try_or_throw};

/// Wrapper for a Vortex file writer that writes Arrow RecordBatches to Vortex format
pub struct WriterWrapper {
    path: String,
    arrays: Vec<ArrayRef>,
}

impl WriterWrapper {
    pub fn new(path: String) -> Result<Self, JNIError> {
        Ok(WriterWrapper {
            path,
            arrays: Vec::new(),
        })
    }

    pub fn write_batch(&mut self, batch: RecordBatch) -> Result<(), JNIError> {
        // Convert RecordBatch to Vortex array immediately to free Arrow memory
        // Use false for nullable since top-level structs representing rows should not be nullable
        let array = ArrayRef::from_arrow(batch, false);
        self.arrays.push(array);
        Ok(())
    }

    pub fn close(self) -> Result<(), JNIError> {
        // Write all accumulated Vortex arrays to file
        if !self.arrays.is_empty() {
            // Arrays are already converted, no need to convert again
            let arrays = self.arrays;

            // If we have multiple arrays, combine them into a ChunkedArray
            let final_array = if arrays.len() == 1 {
                arrays.into_iter().next().ok_or_else(|| {
                    JNIError::Vortex(vortex::error::vortex_err!("No arrays to write"))
                })?
            } else {
                // Create a ChunkedArray from multiple batches
                ChunkedArray::from_iter(arrays).into_array()
            };

            // Write the final array to file
            let path = self.path.clone();

            // Write using VortexWriteOptions with the array's stream
            crate::block_on("write_vortex", async move {
                // Create parent directories if they don't exist
                if let Some(parent) = std::path::Path::new(&path).parent() {
                    tokio::fs::create_dir_all(parent).await.map_err(|e| {
                        JNIError::Vortex(vortex::error::vortex_err!(
                            "Failed to create directories: {}",
                            e
                        ))
                    })?;
                }

                let file = tokio::fs::File::create(&path).await.map_err(|e| {
                    JNIError::Vortex(vortex::error::vortex_err!("Failed to create file: {}", e))
                })?;

                let result = VortexWriteOptions::default()
                    .write(file, final_array.to_array_stream())
                    .await
                    .map_err(JNIError::Vortex);

                // Ensure file is fully written to disk
                if result.is_ok() && let Ok(f) = tokio::fs::File::open(&path).await {
                    let _ = f.sync_all().await;
                }

                result
            })?;
        } else {
            // Write an empty Vortex file with proper format
            // We need to write at least a valid empty array to ensure the file is readable
            
            // Create an empty struct array with no fields (represents empty rows)
            use vortex::arrays::StructArray;
            
            let empty_struct = StructArray::new_with_len(0).into_array();
            
            // Write the empty array to file
            let path = self.path.clone();
            crate::block_on("write_empty_vortex", async move {
                // Create parent directories if they don't exist
                if let Some(parent) = std::path::Path::new(&path).parent() {
                    tokio::fs::create_dir_all(parent).await.map_err(|e| {
                        JNIError::Vortex(vortex::error::vortex_err!(
                            "Failed to create directories: {}",
                            e
                        ))
                    })?;
                }

                let file = tokio::fs::File::create(&path).await.map_err(|e| {
                    JNIError::Vortex(vortex::error::vortex_err!("Failed to create file: {}", e))
                })?;

                VortexWriteOptions::default()
                    .write(file, empty_struct.to_array_stream())
                    .await
                    .map_err(JNIError::Vortex)
            })?;
        }

        Ok(())
    }
}

/// Creates a new Vortex writer
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeWriterMethods_create<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    file_path: JString<'local>,
    _schema_json: JString<'local>,
    _options: JObject<'local>,
) -> jlong {
    try_or_throw(&mut env, |env| {
        // Get the file path
        let path: String = env.get_string(&file_path)?.into();

        // Note: schema_json parameter is kept for API compatibility but not used
        // The schema will be extracted from the IPC data itself

        // Create the writer
        let wrapper = WriterWrapper::new(path)?;

        // Return the pointer
        let ptr = Box::into_raw(Box::new(wrapper)) as jlong;
        Ok(ptr)
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
        let wrapper = unsafe { &mut *(writer_ptr as *mut WriterWrapper) };

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
            let batch = batch_result.map_err(|e| {
                JNIError::Vortex(vortex::error::vortex_err!(
                    "Failed to read RecordBatch: {}",
                    e
                ))
            })?;
            wrapper.write_batch(batch)?;
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
) -> jboolean {
    // Validate pointer before using it
    if writer_ptr <= 0 {
        return JNI_TRUE; // Return success for null/invalid pointers (already closed)
    }

    // Check if the pointer looks valid (aligned and in reasonable range)
    #[allow(clippy::cast_possible_truncation)]
    if (writer_ptr as usize) % 8 != 0 {
        return JNI_FALSE;
    }

    try_or_throw(&mut env, |_env| {
        // Take ownership of the writer and close it
        let wrapper = unsafe { Box::from_raw(writer_ptr as *mut WriterWrapper) };
        wrapper.close()?;

        Ok(JNI_TRUE)
    })
}
