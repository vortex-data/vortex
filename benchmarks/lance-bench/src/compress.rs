// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::anyhow;
use anyhow::bail;
use async_trait::async_trait;
use futures::StreamExt;
use lance::dataset::Dataset;
use lance::dataset::WriteParams;
use lance::deps::arrow_array::RecordBatch;
use lance::deps::arrow_array::RecordBatchIterator;
use lance::deps::arrow_schema::SchemaRef;
use lance_encoding::version::LanceFileVersion;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tempfile::TempDir;
use vortex_bench::Format;
use vortex_bench::compress::Compressed;
use vortex_bench::compress::CompressedData;
use vortex_bench::compress::Compressor;
use vortex_bench::compress::Uncompressed;
use vortex_bench::compress::read_projection;

use crate::convert::convert_utf8view_batch;
use crate::convert::convert_utf8view_schema;

/// Read a Lance dataset and decompress it back into RecordBatches.
pub async fn lance_decompress_read(path: &str) -> anyhow::Result<usize> {
    // Open the Lance dataset from the filesystem path.
    let dataset = Dataset::open(path).await?;
    let mut scanner = dataset.scan();

    // Apply the fixed wide-table read projection, if any. Columns are named "0".."n-1".
    if let Some(cols) = read_projection(dataset.schema().fields.len()) {
        let names: Vec<String> = cols.iter().map(|i| i.to_string()).collect();
        scanner.project(&names)?;
    }

    // Convert to a stream of RecordBatches.
    let mut stream = scanner.try_into_stream().await?;
    let mut nbytes = 0;

    // Sum up the memory size of all decompressed batches.
    while let Some(batch_result) = stream.next().await {
        let batch = batch_result?;
        nbytes += batch.get_array_memory_size();
    }

    Ok(nbytes)
}

/// Calculate the total size of a Lance dataset on disk.
pub fn calculate_lance_size(dataset_path: &Path) -> anyhow::Result<u64> {
    let mut total_size = 0u64;

    // Walk the directory tree to sum up all file sizes.
    for entry in fs::read_dir(dataset_path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        let path = entry.path();

        // Only count files with the `.lance` extension.
        if metadata.is_file() && path.extension().and_then(|e| e.to_str()) == Some("lance") {
            total_size += metadata.len();
        }
    }

    // Lance creates a 'data' subdirectory for the actual data files.
    let data_path = dataset_path.join("data");
    if data_path.exists() {
        for entry in fs::read_dir(&data_path)? {
            let entry = entry?;
            let metadata = entry.metadata()?;
            let path = entry.path();

            // Only count files with the `.lance` extension.
            if metadata.is_file() && path.extension().and_then(|e| e.to_str()) == Some("lance") {
                total_size += metadata.len();
            }
        }
    }

    Ok(total_size)
}

/// Subdirectory of the temp directory holding the Lance dataset.
const DATASET_DIR: &str = "dataset";

/// Source batches in Lance's Arrow version, which differs from the workspace's.
struct LanceInput {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
}

/// Compressor implementation for Lance format.
///
/// Lance writes to the filesystem rather than in-memory buffers, so this implementation
/// uses temp directories. The compress method returns the total size of Lance files on disk.
pub struct LanceCompressor;

#[async_trait]
impl Compressor for LanceCompressor {
    fn format(&self) -> Format {
        Format::Lance
    }

    async fn load(&self, parquet_path: &Path) -> anyhow::Result<Uncompressed> {
        // Read the input parquet file
        let file = File::open(parquet_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let schema = Arc::clone(builder.schema());
        let reader = builder.build()?;
        let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>()?;

        // Convert Utf8View columns to Utf8 (Lance doesn't support Utf8View)
        let batches = batches
            .into_iter()
            .map(convert_utf8view_batch)
            .collect::<anyhow::Result<Vec<_>>>()?;
        let schema = convert_utf8view_schema(&schema);

        Ok(Uncompressed::Opaque(Box::new(LanceInput {
            schema,
            batches,
        })))
    }

    async fn compress(&self, input: &Uncompressed) -> anyhow::Result<Compressed> {
        let LanceInput { schema, batches } = input.opaque()?;
        // The Lance writer wants an owned reader; batch clones only bump buffer refcounts.
        let batches = batches.clone();

        // Create temp directory for Lance dataset
        let temp_dir = TempDir::new()?;
        let dataset_path = temp_dir.path().join(DATASET_DIR);
        fs::create_dir_all(&dataset_path)?;

        let start = Instant::now();

        // Write to Lance format
        let path_str = dataset_path
            .to_str()
            .ok_or_else(|| anyhow!("Failed to convert path to str"))?;
        let reader_iter = RecordBatchIterator::new(batches.into_iter().map(Ok), Arc::clone(schema));
        let write_params = WriteParams::with_storage_version(LanceFileVersion::V2_0);
        Dataset::write(reader_iter, path_str, Some(write_params)).await?;

        let elapsed = start.elapsed();

        // Calculate size of Lance files on disk
        let size = calculate_lance_size(&dataset_path)?;

        Ok(Compressed {
            data: CompressedData::Dir(temp_dir),
            size,
            elapsed,
        })
    }

    async fn decompress(&self, compressed: &Compressed) -> anyhow::Result<Duration> {
        let CompressedData::Dir(temp_dir) = &compressed.data else {
            bail!("Lance decompression expects a dataset directory");
        };
        let dataset_path = temp_dir.path().join(DATASET_DIR);
        let path_str = dataset_path
            .to_str()
            .ok_or_else(|| anyhow!("Failed to convert path to str"))?;

        let start = Instant::now();
        lance_decompress_read(path_str).await?;
        Ok(start.elapsed())
    }
}
