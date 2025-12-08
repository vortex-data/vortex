// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::anyhow;
use arrow_array::RecordBatch;
use arrow_array::RecordBatchReader;
use arrow_cast::cast;
use arrow_schema::ArrowError;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

/// A streaming iterator that reads RecordBatches from multiple Parquet files sequentially.
/// Works equally well for single files and multiple files.
pub struct ParquetFilesIterator {
    files: Vec<PathBuf>,
    schema: Arc<Schema>,
    current_file_index: usize,
    current_reader: Option<Box<dyn RecordBatchReader + Send>>,
}

impl ParquetFilesIterator {
    pub fn new(files: Vec<PathBuf>, schema: Arc<Schema>) -> anyhow::Result<Self> {
        let mut iter = Self {
            files,
            schema,
            current_file_index: 0,
            current_reader: None,
        };
        iter.advance_to_next_file()
            .map_err(|e| anyhow!("Failed to open first Parquet file: {}", e))?;
        Ok(iter)
    }

    fn advance_to_next_file(&mut self) -> Result<(), ArrowError> {
        if self.current_file_index < self.files.len() {
            let file = File::open(&self.files[self.current_file_index]).map_err(|e| {
                ArrowError::IoError(format!("Failed to open Parquet file: {}", e), e)
            })?;
            let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
            self.current_reader = Some(Box::new(builder.build()?));
            self.current_file_index += 1;
        } else {
            self.current_reader = None;
        }
        Ok(())
    }
}

impl Iterator for ParquetFilesIterator {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(reader) = &mut self.current_reader {
                if let Some(result) = reader.next() {
                    return Some(result);
                }
                // Current file is exhausted, try next file
                if let Err(e) = self.advance_to_next_file() {
                    return Some(Err(e));
                }
            } else {
                // No more files
                return None;
            }
        }
    }
}

impl RecordBatchReader for ParquetFilesIterator {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

// Utf8View to Utf8 conversion utilities

/// Convert Utf8View fields in a schema to Utf8.
pub fn convert_utf8view_schema(schema: &Schema) -> Arc<Schema> {
    let new_fields: Vec<Field> = schema
        .fields()
        .iter()
        .map(|field| {
            let new_dtype = match field.data_type() {
                DataType::Utf8View => DataType::Utf8,
                dt => dt.clone(),
            };
            Field::new(field.name(), new_dtype, field.is_nullable())
        })
        .collect();
    Arc::new(Schema::new(new_fields))
}

/// Convert Utf8View arrays to Utf8 arrays in a RecordBatch.
pub fn convert_utf8view_batch(batch: RecordBatch) -> anyhow::Result<RecordBatch> {
    let schema = batch.schema();
    let mut new_columns = Vec::new();

    for (i, column) in batch.columns().iter().enumerate() {
        let field = schema.field(i);
        let new_column = if field.data_type() == &DataType::Utf8View {
            // Cast Utf8View to Utf8.
            cast(column, &DataType::Utf8)?
        } else {
            column.clone()
        };
        new_columns.push(new_column);
    }

    let new_schema = convert_utf8view_schema(&schema);
    Ok(RecordBatch::try_new(new_schema, new_columns)?)
}

/// A wrapper iterator that converts Utf8View columns to Utf8 during iteration.
pub struct ConvertingParquetFilesIterator {
    inner: ParquetFilesIterator,
    converted_schema: Arc<Schema>,
}

impl ConvertingParquetFilesIterator {
    pub fn new(inner: ParquetFilesIterator) -> Self {
        let converted_schema = convert_utf8view_schema(&inner.schema);
        Self {
            inner,
            converted_schema,
        }
    }
}

impl Iterator for ConvertingParquetFilesIterator {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|result| {
            result.and_then(|batch| {
                convert_utf8view_batch(batch)
                    .map_err(|e| ArrowError::ExternalError(e.to_string().into()))
            })
        })
    }
}

impl RecordBatchReader for ConvertingParquetFilesIterator {
    fn schema(&self) -> SchemaRef {
        self.converted_schema.clone()
    }
}
