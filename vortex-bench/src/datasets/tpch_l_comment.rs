// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;

use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;
use futures::TryStreamExt;
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
use crate::IdempotentPath;
use crate::SESSION;
use crate::datasets::Dataset;
use crate::tpch::tpchgen::TpchGenOptions;
use crate::tpch::tpchgen::generate_tpch_tables;

pub struct TPCHLCommentChunked;

async fn ensure_tpch_parquet() -> Result<PathBuf> {
    let base_path = "tpch".to_data_path();
    let scale_factor_dir = base_path.join("1.0");
    let data_dir = scale_factor_dir.join(Format::Parquet.name());

    // Generate TPC-H parquet data if it doesn't exist
    if !data_dir.exists() {
        let options =
            TpchGenOptions::new("1.0".to_string(), scale_factor_dir).with_format(Format::Parquet);
        generate_tpch_tables(options).await?;
    }

    let path = data_dir.join("lineitem.parquet");
    path.exists()
        .then_some(path)
        .ok_or_else(|| anyhow::anyhow!("No lineitem parquet file found"))
}

#[async_trait]
impl Dataset for TPCHLCommentChunked {
    fn name(&self) -> &str {
        "TPC-H l_comment chunked"
    }

    async fn to_vortex_array(&self, ctx: &mut ExecutionCtx) -> Result<ArrayRef> {
        let base_path = "tpch".to_data_path();
        let scale_factor_dir = base_path.join("1.0");
        let data_dir = scale_factor_dir.join(Format::OnDiskVortex.name());

        // Generate TPC-H Vortex data if it doesn't exist
        if !data_dir.exists() {
            // Use blocking call like TPC-H benchmark does
            let options = TpchGenOptions::new("1.0".to_string(), scale_factor_dir)
                .with_format(Format::OnDiskVortex);

            futures::executor::block_on(generate_tpch_tables(options))?;
        }

        let path = data_dir.join("lineitem.vortex");
        let file = SESSION.open_options().open_path(path).await?;
        let projection = pack(vec![("l_comment", col("l_comment"))], NonNullable)
            .optimize_recursive(file.dtype())?
            .bind(file.dtype())?;
        let chunks: Vec<_> = file
            .scan()?
            .with_projection(projection)
            .into_array_stream()?
            .map({
                let ctx = ctx.clone();
                move |a| {
                    let mut ctx = ctx.clone();
                    a.and_then(|a| {
                        let canonical = a.execute::<Canonical>(&mut ctx)?;
                        Ok(canonical.into_array())
                    })
                }
            })
            .try_collect()
            .await?;

        Ok(ChunkedArray::from_iter(chunks).into_array())
    }

    async fn to_parquet_path(&self) -> Result<PathBuf> {
        ensure_tpch_parquet().await
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
        ensure_tpch_parquet().await
    }
}
