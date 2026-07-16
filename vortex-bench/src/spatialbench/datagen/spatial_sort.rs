// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Spatial clustering of the source parquet, in place, so every downstream lane (parquet,
//! vortex-WKB, `vortex-geo-native`) reads the same layout.
//!
//! The geometry zone-map prune skips a chunk only when its bounding box is disjoint from the query
//! region. In generation order every chunk's box spans the whole map, so nothing prunes; ordering
//! rows on a Z-order (Morton) curve of each geometry's bounding-box center gives each chunk a
//! compact box instead. The center works uniformly for every geometry type.
//!
//! Every table with a geometry column is sorted by its first one, each part independently — global
//! order is unnecessary for per-chunk pruning. A parquet marker makes it idempotent; derived vortex
//! files from pre-sort parquet are deleted so the existence-keyed conversions regenerate.

use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use arrow_array::Array;
use arrow_array::ArrayRef;
use arrow_array::RecordBatch;
use arrow_array::UInt64Array;
use arrow_schema::DataType;
use arrow_select::concat::concat_batches;
use arrow_select::take::take;
use futures::TryStreamExt;
use geo::BoundingRect;
use geo::Geometry;
use geo_traits::to_geo::ToGeoGeometry;
use geoarrow::array::GenericWkbArray;
use geoarrow::array::GeoArrowArrayAccessor;
use geoarrow::array::WkbViewArray;
use geoarrow::datatypes::Crs;
use geoarrow::datatypes::Metadata;
use geoarrow::datatypes::WkbType;
use parquet::arrow::AsyncArrowWriter;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::basic::Compression;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use tokio::fs::File as TokioFile;
use tracing::info;

use super::table::Table;
use super::wkb::geo_parquet_metadata;

/// Parquet metadata marker: this file is already spatially sorted (makes the step idempotent).
const SORTED_KEY: &str = "vortex_spatial_sorted";

fn geo_metadata() -> Arc<Metadata> {
    Arc::new(Metadata::new(Crs::default(), None))
}

/// Spatially sort every table with a geometry column, in place.
pub async fn spatially_sort_tables(
    parquet_dir: &Path,
    derived_dirs: &[PathBuf],
) -> anyhow::Result<()> {
    for table in Table::ALL.into_iter().filter(|table| table.is_generated()) {
        sort_table_parquet(table, parquet_dir, derived_dirs).await?;
    }
    Ok(())
}

/// Sort every unsorted `{table}_*.parquet` part by its first geometry column's bounding-box center.
async fn sort_table_parquet(
    table: Table,
    parquet_dir: &Path,
    derived_dirs: &[PathBuf],
) -> anyhow::Result<()> {
    let Some(geom) = table.geometry_columns().first() else {
        return Ok(());
    };

    let pattern = parquet_dir.join(format!("{}_*.parquet", table.name()));
    let mut files: Vec<PathBuf> =
        glob::glob(&pattern.to_string_lossy())?.collect::<Result<_, _>>()?;
    files.sort();
    let mut pending = Vec::new();
    for file in files {
        if !is_sorted(&file).await? {
            pending.push(file);
        }
    }
    if pending.is_empty() {
        return Ok(());
    }

    // Delete before rewriting: an interrupted run leaves unsorted parts unmarked and re-deletes.
    for dir in derived_dirs {
        let stale = dir.join(format!("{}_*.vortex", table.name()));
        for file in glob::glob(&stale.to_string_lossy())?.flatten() {
            tokio::fs::remove_file(&file).await?;
            info!(path = %file.display(), "removed stale derived file from pre-sort parquet");
        }
    }

    for file in pending {
        sort_part(&file, table, geom.name).await?;
    }
    Ok(())
}

/// Whether `path` already carries the [`SORTED_KEY`] marker.
async fn is_sorted(path: &Path) -> anyhow::Result<bool> {
    let builder = ParquetRecordBatchStreamBuilder::new(TokioFile::open(path).await?).await?;
    Ok(builder
        .metadata()
        .file_metadata()
        .key_value_metadata()
        .is_some_and(|kvs| kvs.iter().any(|kv| kv.key == SORTED_KEY)))
}

/// Rewrite one parquet part with its rows in Z-order of `geom_col`.
async fn sort_part(path: &Path, table: Table, geom_col: &str) -> anyhow::Result<()> {
    let builder = ParquetRecordBatchStreamBuilder::new(TokioFile::open(path).await?).await?;
    let schema = Arc::clone(builder.schema());
    let mut reader = builder.build()?;
    let mut batches = Vec::new();
    while let Some(batch) = reader.try_next().await? {
        batches.push(batch);
    }
    if batches.iter().all(|batch| batch.num_rows() == 0) {
        return Ok(());
    }
    let batch = concat_batches(&schema, &batches)?;
    drop(batches);

    let geom_idx = schema.index_of(geom_col)?;
    let (xs, ys) = wkb_centers(batch.column(geom_idx).as_ref())
        .with_context(|| format!("decoding geometry column {geom_col} of {}", path.display()))?;
    let indices = morton_sort_indices(&xs, &ys);

    let columns: Vec<ArrayRef> = batch
        .columns()
        .iter()
        .map(|column| take(column.as_ref(), &indices, None))
        .collect::<Result<_, _>>()?;
    let sorted = RecordBatch::try_new(Arc::clone(&schema), columns)?;
    let num_rows = sorted.num_rows();

    let tmp_path = path.with_extension("parquet.sorttmp");
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer =
        AsyncArrowWriter::try_new(TokioFile::create(&tmp_path).await?, schema, Some(props))?;
    writer.write(&sorted).await?;
    // A fresh write drops metadata: re-tag geo so DuckDB reads `GEOMETRY`, and add the sorted marker.
    if let Some(geo) = geo_parquet_metadata(table) {
        writer.append_key_value_metadata(KeyValue::new("geo".to_string(), Some(geo)));
    }
    writer.append_key_value_metadata(KeyValue::new(
        SORTED_KEY.to_string(),
        Some(format!("morton:{geom_col}")),
    ));
    writer.close().await?;
    tokio::fs::rename(&tmp_path, path).await?;

    info!(
        path = %path.display(),
        rows = num_rows,
        column = geom_col,
        "spatially sorted parquet (morton z-order)"
    );
    Ok(())
}

/// The bounding-box center `(x, y)` of every geometry in a WKB column, whatever its geometry type.
fn wkb_centers(column: &dyn Array) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    let wkb_type = WkbType::new(geo_metadata());
    // Expanded per concrete WKB array type.
    macro_rules! centers {
        ($array:expr) => {{
            let mut xy = Vec::new();
            for item in $array.iter() {
                xy.push(match item {
                    Some(geometry) => bbox_center(&geometry?.to_geometry()),
                    None => (f64::NAN, f64::NAN),
                });
            }
            Ok(xy.into_iter().unzip())
        }};
    }
    match column.data_type() {
        DataType::Binary => centers!(GenericWkbArray::<i32>::try_from((column, wkb_type))?),
        DataType::LargeBinary => centers!(GenericWkbArray::<i64>::try_from((column, wkb_type))?),
        DataType::BinaryView => centers!(WkbViewArray::try_from((column, wkb_type))?),
        other => anyhow::bail!("unsupported WKB column type {other}"),
    }
}

/// The center of a geometry's bounding box, or `NaN` for an empty geometry.
fn bbox_center(geometry: &Geometry<f64>) -> (f64, f64) {
    geometry
        .bounding_rect()
        .map(|r| ((r.min().x + r.max().x) / 2.0, (r.min().y + r.max().y) / 2.0))
        .unwrap_or((f64::NAN, f64::NAN))
}

/// Row indices ordering `(x, y)` along a 32-bit-per-axis Z-order curve, with each axis
/// quantized over the data's own extent.
fn morton_sort_indices(xs: &[f64], ys: &[f64]) -> UInt64Array {
    let (mut xmin, mut xmax) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut ymin, mut ymax) = (f64::INFINITY, f64::NEG_INFINITY);
    for (&x, &y) in xs.iter().zip(ys) {
        if !x.is_nan() && !y.is_nan() {
            xmin = xmin.min(x);
            xmax = xmax.max(x);
            ymin = ymin.min(y);
            ymax = ymax.max(y);
        }
    }

    let keys: Vec<u64> = xs
        .iter()
        .zip(ys)
        .map(|(&x, &y)| {
            if x.is_nan() || y.is_nan() {
                0
            } else {
                interleave(quantize(x, xmin, xmax)) | (interleave(quantize(y, ymin, ymax)) << 1)
            }
        })
        .collect();

    let mut order: Vec<usize> = (0..xs.len()).collect();
    order.sort_unstable_by_key(|&i| keys[i]);
    UInt64Array::from_iter_values(order.into_iter().map(|i| i as u64))
}

/// Map `value` within `[min, max]` to the full `u32` range.
// `t` is clamped to [0, 1], so the product never exceeds `u32::MAX`.
#[expect(clippy::cast_possible_truncation)]
fn quantize(value: f64, min: f64, max: f64) -> u32 {
    let range = max - min;
    if range <= 0.0 {
        return 0;
    }
    let t = ((value - min) / range).clamp(0.0, 1.0);
    (t * f64::from(u32::MAX)) as u32
}

/// Spread a 32-bit value's bits into the even positions of a `u64`.
fn interleave(value: u32) -> u64 {
    let mut n = u64::from(value);
    n = (n | (n << 16)) & 0x0000_FFFF_0000_FFFF;
    n = (n | (n << 8)) & 0x00FF_00FF_00FF_00FF;
    n = (n | (n << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
    n = (n | (n << 2)) & 0x3333_3333_3333_3333;
    n = (n | (n << 1)) & 0x5555_5555_5555_5555;
    n
}
