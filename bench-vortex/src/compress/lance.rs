// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Lance Benchmark Implementation
//
// Unlike Parquet and Vortex which write to in-memory buffers, Lance requires filesystem access.
// We attempted using Lance's `memory://` protocol but it doesn't seem to persist data between
// write and read.
//
// Therefore, Lance benchmarks include filesystem I/O that the other formats don't have. This is
// probably fine because everything should be cached in the OS.
//
// This might not be a perfect comparison, but given the relative speeds between Vortex, Lance, and
// Parquet, it probably doesn't matter that much. Please let us know if this can be improved!

use std::fs;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchIterator};
use arrow_schema::Schema;
use futures::StreamExt;
use lance::dataset::{Dataset, WriteParams};
use lance_encoding::version::LanceFileVersion;
use tempfile::TempDir;

use crate::utils::parquet_utils::{convert_utf8view_batch, convert_utf8view_schema};

/// Write pre-converted [`RecordBatch`]es to Lance format.
pub async fn lance_compress_write_only(
    batches: Vec<RecordBatch>,
    schema: Arc<Schema>,
    dataset_path: &Path,
) -> anyhow::Result<()> {
    let path = dataset_path.to_str().unwrap();
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
    // Lance v2.1 fails on CMSProvider dataset.
    let write_params = WriteParams::with_storage_version(LanceFileVersion::V2_0);
    Dataset::write(reader, path, Some(write_params)).await?;
    Ok(())
}

/// Read a Lance dataset and decompress it back into RecordBatches.
pub async fn lance_decompress_read(path: &str) -> anyhow::Result<usize> {
    // Open the Lance dataset from the filesystem path.
    let dataset = Dataset::open(path).await?;
    let scanner = dataset.scan();

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

/// Helper function for decompression benchmark setup.
/// Includes Utf8View conversion and creates dataset at a fixed path.
pub async fn lance_compress_write(
    batches: Vec<RecordBatch>,
    schema: Arc<Schema>,
    temp_dir: &TempDir,
) -> anyhow::Result<String> {
    // Convert Utf8View columns to Utf8 (Lance doesn't support Utf8View).
    let converted_batches: Vec<RecordBatch> = batches
        .into_iter()
        .map(convert_utf8view_batch)
        .collect::<anyhow::Result<Vec<_>>>()?;
    let converted_schema = convert_utf8view_schema(&schema);

    // Create a fixed subdirectory for decompression testing.
    let dataset_dir = temp_dir.path().join("dataset");
    fs::create_dir_all(&dataset_dir)?;
    let path = dataset_dir.to_str().unwrap();

    let reader = RecordBatchIterator::new(converted_batches.into_iter().map(Ok), converted_schema);
    // Lance v2.1 fails on CMSProvider dataset.
    let write_params = WriteParams::with_storage_version(LanceFileVersion::V2_0);
    Dataset::write(reader, path, Some(write_params)).await?;

    Ok(path.to_string())
}
