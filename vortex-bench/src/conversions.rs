// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::path::Path;
use std::path::PathBuf;

use futures::StreamExt;
use futures::TryStreamExt;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::arrow::async_reader::ParquetRecordBatchStream;
use sysinfo::System;
use tokio::fs::File;
use tokio::fs::OpenOptions;
use tokio::fs::create_dir_all;
use tokio::io::AsyncWriteExt;
use tracing::Instrument;
use tracing::info;
use tracing::trace;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrow::FromArrowArray;
use vortex::array::builders::builder_with_capacity;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::array::stream::ArrayStreamExt;
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::file::WriteOptionsSessionExt;
use vortex::session::VortexSession;

use crate::CompactionStrategy;
use crate::Format;
use crate::SESSION;
use crate::utils::file::idempotent_async;

/// Memory budget per concurrent conversion stream in GB. This is somewhat arbitary.
const MEMORY_PER_STREAM_GB: u64 = 4;

/// Minimum number of concurrent conversion streams.
const MIN_CONCURRENCY: u64 = 1;

/// Maximum number of concurrent conversion streams. This is somewhat arbitary.
const MAX_CONCURRENCY: u64 = 16;

/// Returns the available system memory in bytes.
fn available_memory_bytes() -> u64 {
    System::new_all().available_memory()
}

/// Calculate appropriate concurrency based on available memory.
fn calculate_concurrency() -> usize {
    let available_gb = available_memory_bytes() / (1024 * 1024 * 1024);
    let concurrency = (available_gb / MEMORY_PER_STREAM_GB).clamp(MIN_CONCURRENCY, MAX_CONCURRENCY);

    info!(
        "Available memory: {}GB, maximum concurrency is: {}",
        available_gb, concurrency
    );

    concurrency as usize
}

/// Read a Parquet file and return it as a Vortex [`ChunkedArray`].
///
/// Note: This loads the entire file into memory. For large files, use the streaming conversion like
/// in [`parquet_to_vortex_stream`] instead.
pub async fn parquet_to_vortex_chunks(parquet_path: PathBuf) -> anyhow::Result<ChunkedArray> {
    let file = File::open(parquet_path).await?;
    let builder = ParquetRecordBatchStreamBuilder::new(file).await?;
    let reader = builder.build()?;

    let chunks: Vec<ArrayRef> = parquet_to_vortex_stream(reader)
        .map(|r| r.map_err(anyhow::Error::from))
        .try_collect()
        .await?;

    Ok(ChunkedArray::from_iter(chunks))
}

/// Create a streaming Vortex array from a Parquet reader.
///
/// Streams record batches and converts them to Vortex arrays on-the-fly, avoiding loading the
/// entire file into memory.
pub fn parquet_to_vortex_stream(
    reader: ParquetRecordBatchStream<File>,
) -> impl futures::Stream<Item = VortexResult<ArrayRef>> {
    reader.map(move |result| {
        result.map_err(|e| vortex_err!(External: e)).and_then(|rb| {
            let chunk = ArrayRef::from_arrow(rb, false)?;
            let mut builder = builder_with_capacity(chunk.dtype(), chunk.len());

            // Canonicalize the chunk.
            chunk.append_to_builder(
                builder.as_mut(),
                &mut VortexSession::default().create_execution_ctx(),
            )?;

            Ok(builder.finish())
        })
    })
}

/// Convert a single Parquet file to Vortex format using streaming.
///
/// Streams data directly from Parquet to Vortex without loading the entire file into memory.
pub async fn convert_parquet_file_to_vortex(
    parquet_path: &Path,
    output_path: &Path,
    compaction: CompactionStrategy,
) -> anyhow::Result<()> {
    let file = File::open(parquet_path).await?;
    let builder = ParquetRecordBatchStreamBuilder::new(file).await?;
    let dtype = DType::from_arrow(builder.schema().as_ref());

    let stream = parquet_to_vortex_stream(builder.build()?);

    let mut output_file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(output_path)
        .await?;

    compaction
        .apply_options(SESSION.write_options())
        .write(
            &mut output_file,
            ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype, stream)),
        )
        .await?;

    Ok(())
}

/// Convert all Parquet files in a directory to Vortex format.
///
/// This function reads Parquet files from `{input_path}/parquet/` and writes Vortex files to
/// `{input_path}/vortex-file-compressed/` (for Default compaction) or
/// `{input_path}/vortex-compact/` (for Compact compaction).
///
/// The conversion is idempotent: existing Vortex files will not be regenerated.
pub async fn convert_parquet_directory_to_vortex(
    input_path: &Path,
    compaction: CompactionStrategy,
) -> anyhow::Result<()> {
    let (format, dir_name) = match compaction {
        CompactionStrategy::Compact => (Format::VortexCompact, Format::VortexCompact.name()),
        CompactionStrategy::Default => (Format::OnDiskVortex, Format::OnDiskVortex.name()),
    };

    let vortex_dir = input_path.join(dir_name);
    let parquet_path = input_path.join(Format::Parquet.name());
    create_dir_all(&vortex_dir).await?;

    let parquet_inputs = fs::read_dir(&parquet_path)?.collect::<std::io::Result<Vec<_>>>()?;
    trace!(
        "Found {} parquet files in {}",
        parquet_inputs.len(),
        parquet_path.to_str().unwrap()
    );

    let iter = parquet_inputs
        .iter()
        .filter(|entry| entry.path().extension().is_some_and(|e| e == "parquet"));

    let concurrency = calculate_concurrency();
    futures::stream::iter(iter)
        .map(|dir_entry| {
            let filename = {
                let mut temp = dir_entry.path();
                temp.set_extension("");
                temp.file_name().unwrap().to_str().unwrap().to_string()
            };
            let parquet_file_path = parquet_path.join(format!("{filename}.parquet"));
            let output_path = vortex_dir.join(format!("{filename}.{}", format.ext()));

            tokio::spawn(
                async move {
                    idempotent_async(output_path.as_path(), move |vtx_file| async move {
                        info!(
                            "Processing file '{filename}' with {:?} strategy",
                            compaction
                        );
                        convert_parquet_file_to_vortex(&parquet_file_path, &vtx_file, compaction)
                            .await
                    })
                    .await
                    .expect("Failed to write Vortex file")
                }
                .in_current_span(),
            )
        })
        .buffer_unordered(concurrency)
        .try_collect::<Vec<_>>()
        .await?;

    Ok(())
}

/// Convert a Parquet file to Vortex format with the specified compaction strategy.
///
/// Uses `idempotent_async` to skip conversion if the output file already exists.
pub async fn write_parquet_as_vortex(
    parquet_path: PathBuf,
    vortex_path: &str,
    compaction: CompactionStrategy,
) -> anyhow::Result<PathBuf> {
    idempotent_async(vortex_path, |output_fname| async move {
        let mut output_file = File::create(&output_fname).await?;
        let data = parquet_to_vortex_chunks(parquet_path).await?;
        let write_options = compaction.apply_options(SESSION.write_options());
        write_options
            .write(&mut output_file, data.into_array().to_array_stream())
            .await?;
        output_file.flush().await?;
        Ok(())
    })
    .await
}
