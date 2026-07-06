// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native-geometry prep for `points=native`: decode a table's WKB geometry to native
//! `vortex.geo.{point,polygon,multipolygon}` via `geoarrow_cast` (so Vortex never decodes WKB), then
//! write a Vortex file. A one-time cost; queries then see DuckDB `GEOMETRY` directly.

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
use geoarrow::datatypes::Metadata;
use geoarrow::datatypes::MultiPolygonType;
use geoarrow::datatypes::PointType;
use geoarrow::datatypes::PolygonType;
use geoarrow::datatypes::WkbType;
use geoarrow_cast::cast::cast;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
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

fn geo_metadata() -> Arc<Metadata> {
    Arc::new(Metadata::new(Crs::default(), None))
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

/// Apply `f` to every batch of `table`'s base WKB parquet parts. All columns are kept; only the
/// geometry columns are rewritten to native types.
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
        let mut stream = builder.build()?;
        while let Some(batch) = stream.try_next().await? {
            out.push(f(batch)?);
        }
    }
    Ok(out)
}

/// Rewrite each of `table`'s WKB geometry columns to its native-lane type, tagging the field with
/// the matching `geoarrow.*` extension.
fn native_record_batch(batch: RecordBatch, table: Table) -> anyhow::Result<RecordBatch> {
    let schema = batch.schema();
    let mut fields = schema.fields().to_vec();
    let mut columns = batch.columns().to_vec();

    for geom in table.geometry_columns() {
        let idx = schema.index_of(geom.name)?;
        let column = batch.column(idx).as_ref();
        let wkb_type = WkbType::new(geo_metadata());

        // Wrap the source WKB. SpatialBench tables emit `Binary`; the external `zone` parquet uses
        // `BinaryView`.
        let wkb: Box<dyn GeoArrowArray> = match column.data_type() {
            DataType::Binary => Box::new(GenericWkbArray::<i32>::try_from((column, wkb_type))?),
            DataType::LargeBinary => {
                Box::new(GenericWkbArray::<i64>::try_from((column, wkb_type))?)
            }
            DataType::BinaryView => Box::new(WkbViewArray::try_from((column, wkb_type))?),
            other => anyhow::bail!("{}: unsupported WKB column type {other}", geom.name),
        };

        // Decode to a native, separated-XY GeoArrow type. The columnar round-trip also normalizes
        // WKB endianness (Overture ships big-endian; native types carry none).
        let native: Arc<dyn GeoArrowArray> = match geom.kind {
            GeometryKind::Point => cast(
                wkb.as_ref(),
                &GeoArrowType::Point(
                    PointType::new(Dimension::XY, geo_metadata())
                        .with_coord_type(CoordType::Separated),
                ),
            )?,
            GeometryKind::Polygon => cast(
                wkb.as_ref(),
                &GeoArrowType::Polygon(
                    PolygonType::new(Dimension::XY, geo_metadata())
                        .with_coord_type(CoordType::Separated),
                ),
            )?,
            // Polygon promotes to a one-element multipolygon, so this also covers the mixed
            // `Polygon`/`MultiPolygon` zone boundaries.
            GeometryKind::MultiPolygon => cast(
                wkb.as_ref(),
                &GeoArrowType::MultiPolygon(
                    MultiPolygonType::new(Dimension::XY, geo_metadata())
                        .with_coord_type(CoordType::Separated),
                ),
            )?,
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
