// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::fs::File;
use std::path::Path;
use std::path::PathBuf;

use arrow_array::RecordBatchReader;
use futures::StreamExt;
use futures::TryStreamExt;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tokio::fs::OpenOptions;
use tokio::fs::create_dir_all;
use tracing::Instrument;
use tracing::info;
use tracing::trace;
use vortex::array::ArrayRef;
use vortex::array::arrow::FromArrowArray;
use vortex::array::iter::ArrayIteratorAdapter;
use vortex::array::iter::ArrayIteratorExt;
use vortex::array::stream::ArrayStream;
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType;
use vortex::error::VortexError;
use vortex::file::WriteOptionsSessionExt;

use crate::CompactionStrategy;
use crate::Format;
use crate::SESSION;
use crate::utils::file::idempotent_async;

/// Read a Parquet file and return it as a Vortex ArrayStream.
pub fn parquet_to_vortex(parquet_path: PathBuf) -> anyhow::Result<impl ArrayStream> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(parquet_path)?)?.build()?;

    let array_iter = ArrayIteratorAdapter::new(
        DType::from_arrow(reader.schema()),
        reader.map(|br| {
            br.map_err(VortexError::from)
                .map(|b| ArrayRef::from_arrow(b, false))
        }),
    );

    Ok(array_iter.into_array_stream())
}

/// Convert all Parquet files in a directory to Vortex format.
///
/// This function reads Parquet files from `{input_path}/parquet/` and writes
/// Vortex files to `{input_path}/vortex-file-compressed/` (for Default compaction)
/// or `{input_path}/vortex-compact/` (for Compact compaction).
///
/// The conversion is idempotent - existing Vortex files will not be regenerated.
pub async fn convert_parquet_to_vortex(
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
                        let array_stream = parquet_to_vortex(parquet_file_path)?;
                        let mut f = OpenOptions::new()
                            .write(true)
                            .truncate(true)
                            .create(true)
                            .open(&vtx_file)
                            .await?;

                        let write_options = compaction.apply_options(SESSION.write_options());

                        write_options.write(&mut f, array_stream).await?;

                        anyhow::Ok(())
                    })
                    .await
                    .expect("Failed to write Vortex file")
                }
                .in_current_span(),
            )
        })
        .buffer_unordered(16)
        .try_collect::<Vec<_>>()
        .await?;
    Ok(())
}
