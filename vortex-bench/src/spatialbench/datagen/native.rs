// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native-geometry preparation for `points=native`: decode each table's WKB geometry columns to
//! native GeoArrow types in Arrow land (`geoarrow_cast`, so Vortex never decodes WKB), then write
//! them as a native Vortex file and a GeoParquet file. The decode is a one-time data-prep cost.

use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use arrow_array::RecordBatch;
use arrow_schema::DataType;
use arrow_schema::Schema;
use futures::TryStreamExt;
use geoarrow::array::GenericWkbArray;
use geoarrow::array::GeoArrowArray;
use geoarrow::array::WkbViewArray;
use geoarrow::datatypes::CoordType;
use geoarrow::datatypes::Crs;
use geoarrow::datatypes::Dimension;
use geoarrow::datatypes::GeoArrowType;
use geoarrow::datatypes::GeometryType;
use geoarrow::datatypes::Metadata;
use geoarrow::datatypes::PointType;
use geoarrow::datatypes::PolygonType;
use geoarrow::datatypes::WkbType;
use geoarrow_cast::cast::cast;
use parquet::arrow::AsyncArrowWriter;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::arrow::ProjectionMask;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tokio::fs::File as TokioFile;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrow::ArrowSessionExt;
use vortex::file::WriteOptionsSessionExt;

use super::table::GeometryKind;
use super::table::Table;
use crate::SESSION;
use crate::utils::file::idempotent_async;

/// EPSG:4326, the CRS the benchmark data and queries assume.
fn epsg_4326() -> Arc<Metadata> {
    Arc::new(Metadata::new(
        Crs::from_unknown_crs_type("EPSG:4326".to_string()),
        None,
    ))
}

/// Write `{native_dir}/{table}_0.vortex` with native geometry columns from the WKB parquet. Idempotent.
pub async fn write_native_vortex(
    table: Table,
    parquet_dir: &Path,
    native_dir: &Path,
) -> anyhow::Result<PathBuf> {
    idempotent_async(
        native_dir.join(format!("{}_0.vortex", table.name())),
        |path| async move {
            let chunks = map_source_batches(parquet_dir, table, |b| native_chunk(b, table)).await?;

            let dtype = chunks[0].dtype().clone();
            let chunked = ChunkedArray::try_new(chunks, dtype)?.into_array();
            let mut file = TokioFile::create(&path).await?;
            SESSION
                .write_options()
                .write(&mut file, chunked.to_array_stream())
                .await?;
            tracing::info!(path = %path.display(), table = table.name(), "wrote native geometry table");
            Ok(())
        },
    )
    .await
}

/// Write `{out_dir}/{table}_0.parquet` with native GeoArrow geometry columns (separated XY,
/// `geoarrow.*` field metadata). Idempotent.
pub async fn write_native_parquet(
    table: Table,
    parquet_dir: &Path,
    out_dir: &Path,
) -> anyhow::Result<PathBuf> {
    idempotent_async(
        out_dir.join(format!("{}_0.parquet", table.name())),
        |path| async move {
            let batches =
                map_source_batches(parquet_dir, table, |b| native_record_batch(b, table)).await?;

            let schema = batches.first().context("no batches to write")?.schema();
            let props = WriterProperties::builder()
                .set_compression(Compression::SNAPPY)
                .build();
            let mut writer =
                AsyncArrowWriter::try_new(TokioFile::create(&path).await?, schema, Some(props))?;
            for batch in &batches {
                writer.write(batch).await?;
            }
            writer.close().await?;
            tracing::info!(path = %path.display(), table = table.name(), "wrote native geometry parquet table");
            Ok(())
        },
    )
    .await
}

/// Apply `f` to every batch read from `table`'s base WKB parquet parts, projected to its columns.
async fn map_source_batches<T>(
    parquet_dir: &Path,
    table: Table,
    mut f: impl FnMut(RecordBatch) -> anyhow::Result<T>,
) -> anyhow::Result<Vec<T>> {
    let pattern = parquet_dir.join(format!("{}_*.parquet", table.name()));
    let mut files: Vec<PathBuf> =
        glob::glob(&pattern.to_string_lossy())?.collect::<Result<_, _>>()?;
    files.sort();
    anyhow::ensure!(!files.is_empty(), "no parquet matching {pattern:?}");

    let mut out = Vec::new();
    for file in files {
        let builder = ParquetRecordBatchStreamBuilder::new(TokioFile::open(&file).await?).await?;
        let mask =
            ProjectionMask::columns(builder.parquet_schema(), table.columns().iter().copied());
        let mut stream = builder.with_projection(mask).build()?;
        while let Some(batch) = stream.try_next().await? {
            out.push(f(batch)?);
        }
    }
    Ok(out)
}

/// Convert each of `table`'s WKB geometry columns to its native-lane representation, swapping the
/// column in so the field carries the matching `geoarrow.*` extension metadata.
fn native_record_batch(batch: RecordBatch, table: Table) -> anyhow::Result<RecordBatch> {
    let schema = batch.schema();
    let mut fields = schema.fields().to_vec();
    let mut columns = batch.columns().to_vec();

    for geom in table.geometry_columns() {
        let idx = schema.index_of(geom.name)?;
        let column = batch.column(idx).as_ref();
        let wkb_type = WkbType::new(epsg_4326());

        // Wrap the source WKB bytes in the matching GeoArrow array. SpatialBench's own tables emit
        // `Binary`; the externally-sourced `zone` parquet uses `BinaryView`.
        let wkb: Box<dyn GeoArrowArray> = match column.data_type() {
            DataType::Binary => Box::new(GenericWkbArray::<i32>::try_from((column, wkb_type))?),
            DataType::LargeBinary => {
                Box::new(GenericWkbArray::<i64>::try_from((column, wkb_type))?)
            }
            DataType::BinaryView => Box::new(WkbViewArray::try_from((column, wkb_type))?),
            other => anyhow::bail!("{}: unsupported WKB column type {other}", geom.name),
        };

        // Produce the native-lane array for this column's geometry kind.
        let native: Arc<dyn GeoArrowArray> = match geom.kind {
            // Homogeneous columns decode to a native, separated-XY GeoArrow type.
            GeometryKind::Point => cast(
                wkb.as_ref(),
                &GeoArrowType::Point(
                    PointType::new(Dimension::XY, epsg_4326())
                        .with_coord_type(CoordType::Separated),
                ),
            )?,
            GeometryKind::Polygon => cast(
                wkb.as_ref(),
                &GeoArrowType::Polygon(
                    PolygonType::new(Dimension::XY, epsg_4326())
                        .with_coord_type(CoordType::Separated),
                ),
            )?,
            // A mixed Polygon/MultiPolygon column (Overture zones) has no native Vortex type, so it
            // stays WKB — but re-encoded little-endian: Overture ships big-endian and DuckDB's direct
            // GEOMETRY ingestion only accepts little-endian. Round-tripping through a mixed geometry
            // array re-serializes it as little-endian WKB.
            GeometryKind::Wkb => {
                let mixed = cast(
                    wkb.as_ref(),
                    &GeoArrowType::Geometry(GeometryType::new(epsg_4326())),
                )?;
                cast(
                    mixed.as_ref(),
                    &GeoArrowType::Wkb(WkbType::new(epsg_4326())),
                )?
            }
        };

        columns[idx] = native.to_array_ref();
        fields[idx] = Arc::new(native.data_type().to_field(geom.name, false));
    }

    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?)
}

/// Convert a WKB batch to a Vortex struct chunk with `table`'s geometry columns as native types.
fn native_chunk(batch: RecordBatch, table: Table) -> anyhow::Result<ArrayRef> {
    let native_batch = native_record_batch(batch, table)?;
    let native_schema = native_batch.schema();
    SESSION
        .arrow()
        .from_arrow_record_batch(native_batch, &native_schema)
        .context("importing native batch")
}
