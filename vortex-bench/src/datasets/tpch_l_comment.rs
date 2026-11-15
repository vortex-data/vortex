// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Result;
use arrow_array::RecordBatchReader;
use async_trait::async_trait;
use futures::TryStreamExt;
use parquet::arrow::ArrowWriter;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::StructArray;
use vortex::dtype::Nullability::NonNullable;
use vortex::expr::col;
use vortex::expr::pack;
use vortex::file::OpenOptionsSessionExt;

use crate::Format;
use crate::SESSION;
use crate::datasets::Dataset;
use crate::datasets::tpch_sf1::TpchSf1Table;
use crate::idempotent;
use crate::utils::parquet::parquet_writer_properties;

pub struct TPCHLCommentChunked;

/// Path of TPC-H SF=1 `lineitem` in `format`, generating the TPC-H tables if it is missing.
async fn ensure_lineitem(format: Format) -> Result<PathBuf> {
    TpchSf1Table::LineItem.path(format).await
}

/// Rows per row group in the `l_comment` Parquet file, the same as the TPC-H generator writes.
const L_COMMENT_ROW_GROUP_ROWS: usize = 1024 * 1024;

/// Path of a Parquet file that holds only the `l_comment` column of TPC-H SF=1 `lineitem`.
///
/// The compression benchmark loads every format's input from [`Dataset::to_parquet_path`], so
/// this file, not the whole `lineitem` table, is what the `l_comment` datasets measure.
async fn l_comment_parquet() -> Result<PathBuf> {
    let lineitem = ensure_lineitem(Format::Parquet).await?;
    tokio::task::spawn_blocking(move || {
        idempotent(&"tpch-l_comment/1.0/l_comment.parquet", |output| {
            write_l_comment(&lineitem, output)
        })
    })
    .await?
}

/// Copy the `l_comment` column of the `lineitem` Parquet file to `output`.
fn write_l_comment(lineitem: &Path, output: &Path) -> Result<()> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(lineitem)?)?;
    let column = builder.schema().index_of("l_comment")?;
    let mask = ProjectionMask::roots(builder.parquet_schema(), [column]);
    let reader = builder.with_projection(mask).build()?;
    let props = parquet_writer_properties()
        .set_max_row_group_row_count(Some(L_COMMENT_ROW_GROUP_ROWS))
        .build();
    let mut writer = ArrowWriter::try_new(File::create(output)?, reader.schema(), Some(props))?;
    for batch in reader {
        writer.write(&batch?)?;
    }
    writer.close()?;
    Ok(())
}

#[async_trait]
impl Dataset for TPCHLCommentChunked {
    fn name(&self) -> &str {
        "TPC-H l_comment chunked"
    }

    async fn to_vortex_array(&self, ctx: &mut ExecutionCtx) -> Result<ArrayRef> {
        let path = ensure_lineitem(Format::OnDiskVortex).await?;
        let file = SESSION.open_options().open_path(path).await?;
        let projection = pack(vec![("l_comment", col("l_comment"))], NonNullable)
            .bind(file.dtype())?
            .optimize_recursive()?;
        let chunks: Vec<_> = file
            .scan()?
            .with_projection(projection)
            .map({
                let ctx = ctx.clone();
                move |a| {
                    let mut ctx = ctx.clone();
                    let canonical = a.execute::<Canonical>(&mut ctx)?;
                    Ok(canonical.into_array())
                }
            })
            .into_array_stream()?
            .try_collect()
            .await?;

        Ok(ChunkedArray::from_iter(chunks).into_array())
    }

    async fn to_parquet_path(&self) -> Result<PathBuf> {
        l_comment_parquet().await
    }
}

pub struct TPCHLCommentCanonical;

#[async_trait]
impl Dataset for TPCHLCommentCanonical {
    fn name(&self) -> &str {
        "TPC-H l_comment canonical"
    }

    async fn to_vortex_array(&self, ctx: &mut ExecutionCtx) -> Result<ArrayRef> {
        let comments_chunked = TPCHLCommentChunked.to_vortex_array(ctx).await?;
        let comments_canonical = comments_chunked.execute::<StructArray>(ctx)?.into_array();
        Ok(ChunkedArray::from_iter([comments_canonical]).into_array())
    }

    async fn to_parquet_path(&self) -> Result<PathBuf> {
        l_comment_parquet().await
    }
}
