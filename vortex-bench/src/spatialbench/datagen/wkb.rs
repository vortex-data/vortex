// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! SpatialBench WKB base-table generation via the `spatialbench` crates (a tpchgen-rs fork).
//! Geometry is emitted as WKB; the native-Point encodings derive from these files in
//! [`super::native`].

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
// spatialbench emits arrow-56 batches, so they must be written with its matching arrow-56
// parquet crate, not the workspace's arrow-58 one. The parquet file itself is version-neutral.
use spatialbench::generators::TripGenerator;
use spatialbench_arrow::RecordBatchIterator;
use spatialbench_arrow::TripArrow;
use spatialbench_parquet::arrow::AsyncArrowWriter;
use spatialbench_parquet::basic::Compression;
use spatialbench_parquet::file::properties::WriterProperties;
use tokio::fs::File as TokioFile;
use tracing::info;

use super::table::TABLES;
use super::table::Table;
use crate::Format;
use crate::utils::file::idempotent_async;

/// Batch size matching the TPC-H generator's streaming batches.
const BATCH_SIZE: usize = 8192 * 64;

/// Batch iterator for one partition of `table`, from the arrow-56 `spatialbench` crates.
fn iterator(
    table: Table,
    scale_factor: f64,
    part: i32,
    part_count: i32,
) -> Box<dyn RecordBatchIterator> {
    match table {
        Table::Trip => Box::new(
            TripArrow::new(TripGenerator::new(scale_factor, part, part_count))
                .with_batch_size(BATCH_SIZE),
        ),
    }
}

/// Generate the SpatialBench base tables as parquet under `{output_dir}/parquet/`.
pub async fn generate_tables(scale_factor: &str, output_dir: PathBuf) -> Result<()> {
    let scale_factor = scale_factor.parse::<f64>()?;
    let parquet_dir = output_dir.join(Format::Parquet.name());
    fs::create_dir_all(&parquet_dir)?;

    // One part per unit of scale factor keeps each file near the ~350MB the trip generator
    // produces at SF1.
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let num_parts = (scale_factor.ceil() as usize).max(1);
    let part_count = i32::try_from(num_parts)?;

    for &table in TABLES {
        for part_idx in 0..num_parts {
            let output_file = parquet_dir.join(format!("{}_{part_idx}.parquet", table.name()));
            let part = i32::try_from(part_idx + 1)?;

            idempotent_async(output_file.to_string_lossy().as_ref(), |path| async move {
                info!(
                    scale_factor,
                    part,
                    part_count,
                    table = table.name(),
                    "Generating SpatialBench table"
                );

                let iter = iterator(table, scale_factor, part, part_count);
                let schema = Arc::clone(iter.schema());

                let file = TokioFile::create(&path).await?;
                let props = WriterProperties::builder()
                    .set_compression(Compression::SNAPPY)
                    .build();
                let mut writer = AsyncArrowWriter::try_new(file, schema, Some(props))?;
                for batch in iter {
                    writer.write(&batch).await?;
                }
                writer.close().await?;

                Ok::<(), anyhow::Error>(())
            })
            .await?;
        }
    }

    Ok(())
}
