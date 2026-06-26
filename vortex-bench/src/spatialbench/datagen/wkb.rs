// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! SpatialBench WKB base-table generation via the `spatialbench` crates (a tpchgen-rs fork).
//! Geometry is emitted as WKB, which DuckDB reads directly as `GEOMETRY` via `ST_GeomFromWKB`.

use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
// spatialbench emits arrow-56 batches, so they must be written with its matching arrow-56
// parquet crate, not the workspace's arrow-58 one. The parquet file itself is version-neutral.
use spatialbench::generators::BuildingGenerator;
use spatialbench::generators::CustomerGenerator;
use spatialbench::generators::TripGenerator;
use spatialbench_arrow::BuildingArrow;
use spatialbench_arrow::CustomerArrow;
use spatialbench_arrow::RecordBatchIterator;
use spatialbench_arrow::TripArrow;
use spatialbench_parquet::arrow::AsyncArrowWriter;
use spatialbench_parquet::basic::Compression;
use spatialbench_parquet::file::properties::WriterProperties;
use spatialbench_parquet::format::KeyValue;
use tokio::fs::File as TokioFile;
use tokio::process::Command;
use tracing::info;
use tracing::warn;

use super::table::Table;
use crate::Format;
use crate::utils::file::idempotent_async;

/// Batch size matching the TPC-H generator's streaming batches.
const BATCH_SIZE: usize = 8192 * 64;

impl Table {
    /// Batch iterator for one partition of this table, from the arrow-56 `spatialbench` crates. Only
    /// called for generated tables (see [`Table::is_generated`]).
    fn batch_iterator(
        self,
        scale_factor: f64,
        part: i32,
        part_count: i32,
    ) -> Box<dyn RecordBatchIterator> {
        match self {
            Table::Trip => Box::new(
                TripArrow::new(TripGenerator::new(scale_factor, part, part_count))
                    .with_batch_size(BATCH_SIZE),
            ),
            Table::Building => Box::new(
                BuildingArrow::new(BuildingGenerator::new(scale_factor, part, part_count))
                    .with_batch_size(BATCH_SIZE),
            ),
            Table::Customer => Box::new(
                CustomerArrow::new(CustomerGenerator::new(scale_factor, part, part_count))
                    .with_batch_size(BATCH_SIZE),
            ),
            Table::Zone => unreachable!("zone is sourced externally, not generated in-process"),
        }
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

    for table in Table::ALL.into_iter().filter(|table| table.is_generated()) {
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

                let iter = table.batch_iterator(scale_factor, part, part_count);
                let schema = Arc::clone(iter.schema());

                let file = TokioFile::create(&path).await?;
                let props = WriterProperties::builder()
                    .set_compression(Compression::SNAPPY)
                    .build();
                let mut writer = AsyncArrowWriter::try_new(file, schema, Some(props))?;
                for batch in iter {
                    writer.write(&batch).await?;
                }
                // Tag geometry columns with GeoParquet `geo` metadata so DuckDB's `read_parquet`
                // surfaces them as `GEOMETRY` directly.
                if let Some(geo) = geo_parquet_metadata(table) {
                    writer.append_key_value_metadata(KeyValue::new("geo".to_string(), Some(geo)));
                }
                writer.close().await?;

                Ok::<(), anyhow::Error>(())
            })
            .await?;
        }
    }

    // `zone` isn't generated in-process (`Table::is_generated` is false); it comes from the upstream
    // CLI. Best-effort: a missing/failed CLI shouldn't block the zone-free queries, so warn and
    // carry on.
    if let Err(e) = generate_zone(scale_factor, &parquet_dir).await {
        warn!(
            error = %e,
            "zone table not generated — SpatialBench queries Q2/Q4/Q6/Q10/Q11 need it. Install the \
             upstream generator (`cargo install --path <sedona-spatialbench>/spatialbench-cli`) or \
             set SPATIALBENCH_CLI to its binary, then re-run."
        );
    }

    Ok(())
}

/// Generate the externally-sourced `zone` table by shelling out to the upstream `spatialbench-cli`.
async fn generate_zone(scale_factor: f64, parquet_dir: &Path) -> Result<()> {
    if parquet_dir.join("zone_0.parquet").exists() {
        return Ok(());
    }
    let cli = std::env::var("SPATIALBENCH_CLI").unwrap_or_else(|_| "spatialbench-cli".to_string());

    // Generate into a scratch dir so the CLI's `zone.parquet` name can't collide with the base
    // tables, then move the produced parts into place as `zone_{part}.parquet`.
    // Start from an empty scratch dir (clear any leftover from an interrupted run).
    let scratch = parquet_dir.join(".zone-scratch");
    fs::remove_dir_all(&scratch).ok();
    fs::create_dir_all(&scratch)?;

    info!(
        scale_factor,
        cli, "Generating SpatialBench zone table via spatialbench-cli"
    );
    let status = Command::new(&cli)
        .arg("-s")
        .arg(scale_factor.to_string())
        .args(["-T", "zone", "-f", "parquet", "-o"])
        .arg(&scratch)
        .status()
        .await
        .with_context(|| format!("failed to spawn `{cli}` (is it installed / on PATH?)"))?;
    anyhow::ensure!(
        status.success(),
        "`{cli}` exited with {status} while generating zone"
    );

    // The CLI writes `zone.parquet` (single part) or `zone/zone.N.parquet`.
    let mut produced: Vec<PathBuf> = glob::glob(&scratch.join("**/*.parquet").to_string_lossy())?
        .collect::<std::result::Result<_, _>>()?;
    produced.sort();
    anyhow::ensure!(
        !produced.is_empty(),
        "`{cli}` produced no zone parquet under {}",
        scratch.display()
    );
    for (part_idx, src) in produced.iter().enumerate() {
        fs::rename(src, parquet_dir.join(format!("zone_{part_idx}.parquet")))?;
    }
    fs::remove_dir_all(&scratch).ok();
    Ok(())
}

/// GeoParquet metadata for WKB geometry columns, or `None` when it has none.
pub(crate) fn geo_parquet_metadata(table: Table) -> Option<String> {
    let geometry_columns = table.geometry_columns();
    let primary = geometry_columns.first()?;
    let columns: serde_json::Map<String, serde_json::Value> = geometry_columns
        .iter()
        .map(|&column| {
            (
                column.to_string(),
                serde_json::json!({ "encoding": "WKB", "geometry_types": [] }),
            )
        })
        .collect();
    Some(
        serde_json::json!({
            "version": "1.0.0",
            "primary_column": primary,
            "columns": columns,
        })
        .to_string(),
    )
}
