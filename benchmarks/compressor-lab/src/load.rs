// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Loading a chunk's rows from its source.

use std::fs::File;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::bail;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::RecordBatch;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;

use crate::fetch::RawData;
use crate::source::ChunkInput;

/// Where a chunk's raw data is found.
#[derive(Debug, Clone)]
pub struct DataLocation {
    /// Where `parquet` sources' relative paths are resolved from.
    pub data_root: PathBuf,
    /// Where `remote` sources' files are cached or kept.
    pub raw: RawData,
}

/// The chunk's rows as an Arrow array.
pub fn arrow(input: &ChunkInput, data: &DataLocation) -> anyhow::Result<ArrowArrayRef> {
    match input {
        ChunkInput::Parquet {
            file,
            file_fingerprint,
            column,
            row_start,
            row_end,
            ..
        } => read_parquet(
            &data.data_root.join(file),
            file_fingerprint,
            column,
            *row_start,
            *row_end,
        ),
        ChunkInput::Remote {
            url,
            storage,
            file_fingerprint,
            column,
            row_start,
            row_end,
            ..
        } => {
            let path = data.raw.locate(url, file_fingerprint, *storage)?;
            read_parquet(&path, file_fingerprint, column, *row_start, *row_end)
        }
        ChunkInput::Tpch {
            table,
            scale_factor,
            column,
            row_start,
            row_end,
            ..
        } => {
            let batches = crate::tpch::table(table, *scale_factor)?;
            let first = batches.first().context("empty TPC-H table")?;
            let index = first.schema().index_of(column)?;
            let all = concat_column(&batches, index)?;
            Ok(all.slice(
                usize::try_from(*row_start)?,
                usize::try_from(row_end - row_start)?,
            ))
        }
        ChunkInput::Synthetic {
            generator,
            ptype,
            params,
            seed,
            rows,
            ..
        } => crate::synthetic::generate(generator, *ptype, params, *seed, *rows),
    }
}

fn read_parquet(
    path: &Path,
    file_fingerprint: &str,
    column: &str,
    row_start: u64,
    row_end: u64,
) -> anyhow::Result<ArrowArrayRef> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(
        File::open(path).with_context(|| format!("opening {}", path.display()))?,
    )?;
    let index = builder
        .schema()
        .index_of(column)
        .with_context(|| format!("{} has no column `{column}`", path.display()))?;
    let mask = ProjectionMask::roots(builder.parquet_schema(), [index]);
    let rows = usize::try_from(row_end - row_start)?;
    let reader = builder
        .with_projection(mask)
        .with_offset(usize::try_from(row_start)?)
        .with_limit(rows)
        .with_batch_size(rows.max(1))
        .build()?;
    let batches = reader.collect::<Result<Vec<RecordBatch>, _>>()?;
    let array = concat_column(&batches, 0)?;
    if array.len() != rows {
        bail!(
            "{} returned {} rows for {column} [{row_start}, {row_end}); expected {rows} \
             (fingerprint {file_fingerprint})",
            path.display(),
            array.len()
        );
    }
    Ok(array)
}

fn concat_column(batches: &[RecordBatch], index: usize) -> anyhow::Result<ArrowArrayRef> {
    let columns: Vec<&dyn arrow_array::Array> =
        batches.iter().map(|b| b.column(index).as_ref()).collect();
    if columns.len() == 1 {
        return Ok(std::sync::Arc::clone(batches[0].column(index)));
    }
    Ok(arrow_select::concat::concat(&columns)?)
}

/// Converts an Arrow array to its canonical Vortex form, for any dtype class.
pub fn to_vortex(array: &ArrowArrayRef, ctx: &mut ExecutionCtx) -> anyhow::Result<ArrayRef> {
    let nullable = array.null_count() > 0;
    let vortex =
        vortex_arrow::from_arrow_columnar(ArrowArrayRef::clone(array), array.len(), nullable, ctx)?;
    Ok(vortex.execute::<Canonical>(ctx)?.into_array())
}
