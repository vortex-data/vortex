// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::fs::File;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::anyhow;
use arrow_cast::cast;
use lance::dataset::Dataset as LanceDataset;
use lance::dataset::WriteParams;
use lance::deps::arrow_array::RecordBatch;
use lance::deps::arrow_array::RecordBatchReader;
use lance::deps::arrow_schema::ArrowError;
use lance::deps::arrow_schema::DataType;
use lance::deps::arrow_schema::Field;
use lance::deps::arrow_schema::Schema;
use lance::deps::arrow_schema::SchemaRef;
use lance_encoding::version::LanceFileVersion;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tokio::fs::create_dir_all;
use tracing::info;
use vortex_bench::utils::file::idempotent_async;

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
        Arc::clone(&self.schema)
    }
}

/// Generic function to convert Parquet files to Lance format.
///
/// This function:
/// 1. Finds all Parquet files matching the optional prefix
/// 2. Creates a streaming iterator over them
/// 3. Writes them to a single Lance dataset
///
/// If `convert_utf8view` is true, any Utf8View columns will be converted to Utf8
/// (required for datasets like TPCH since Lance doesn't support Utf8View).
pub async fn convert_parquet_to_lance<'p>(
    parquet_dir: &'p Path,
    lance_dir: &'p Path,
    dataset_name: &str,
    file_prefix: Option<&str>,
    convert_utf8view: bool,
) -> anyhow::Result<()> {
    // let lance_dir = lance_dir.to_path_buf();
    // let parquet_dir = parquet_dir.to_path_buf();
    // let file_prefix = file_prefix.to_owned();
    // let dataset_name = dataset_name.to_string();

    let dataset_path = lance_dir.join(format!("{}.lance", dataset_name));

    // Use idempotent pattern to avoid reprocessing
    idempotent_async(
        dataset_path.as_path(),
        move |lance_path: PathBuf| async move {
            create_dir_all(&lance_dir).await?;

            // Collect all Parquet files in the directory
            let parquet_files: Vec<_> = fs::read_dir(parquet_dir)?
                .filter_map(|entry| entry.ok())
                .filter(|entry| {
                    let path = entry.path();
                    if path.extension().is_some_and(|e| e == "parquet") {
                        if let Some(prefix) = file_prefix {
                            // Check if file starts with the prefix
                            path.file_stem()
                                .and_then(|s| s.to_str())
                                .map(|s| s.starts_with(prefix))
                                .unwrap_or(false)
                        } else {
                            // No prefix filter, accept all parquet files
                            true
                        }
                    } else {
                        false
                    }
                })
                .map(|entry| entry.path())
                .collect();

            if parquet_files.is_empty() {
                anyhow::bail!(
                    "No Parquet files found{}in {}",
                    if let Some(p) = file_prefix {
                        format!(" with prefix '{}' ", p)
                    } else {
                        " ".to_string()
                    },
                    parquet_dir.display()
                );
            }

            info!(
                "Converting {} Parquet file(s) to Lance dataset '{}'",
                parquet_files.len(),
                dataset_name
            );

            // Get schema from the first Parquet file
            let first_file = File::open(&parquet_files[0])?;
            let first_builder = ParquetRecordBatchReaderBuilder::try_new(first_file)?;
            let schema = Arc::clone(first_builder.schema());

            // Create a streaming iterator that reads from all Parquet files
            let batch_iter = ParquetFilesIterator::new(parquet_files, schema)?;

            info!("Starting streaming write to Lance");

            // Write all batches to a single Lance dataset
            let lance_path_str = lance_path
                .to_str()
                .ok_or_else(|| anyhow!("Lance dataset path is not valid UTF-8"))?;

            // Use the converting iterator if needed
            if convert_utf8view {
                info!("Converting Utf8View columns to Utf8 for Lance compatibility");
                let converting_iter = ConvertingParquetFilesIterator::new(batch_iter);
                LanceDataset::write(
                    Box::new(converting_iter),
                    lance_path_str,
                    Some(WriteParams::with_storage_version(LanceFileVersion::V2_1)),
                )
                .await?;
            } else {
                LanceDataset::write(
                    Box::new(batch_iter),
                    lance_path_str,
                    Some(WriteParams::with_storage_version(LanceFileVersion::V2_1)),
                )
                .await?;
            }

            info!(
                "Successfully created Lance dataset '{}' at {}",
                dataset_name,
                lance_path.display()
            );

            anyhow::Ok(())
        },
    )
    .await?;

    Ok(())
}

// Utf8View to Utf8 conversion utilities
// Lance doesn't support Arrow's Utf8View type (variable-length string view optimization).
// We must convert Utf8View columns to regular Utf8 before writing to Lance.

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
            Arc::clone(column)
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
        Arc::clone(&self.converted_schema)
    }
}
