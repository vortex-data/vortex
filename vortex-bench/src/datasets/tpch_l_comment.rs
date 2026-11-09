// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;

use anyhow::Result;
use async_trait::async_trait;
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
use crate::utils::split;

pub struct TPCHLCommentChunked;

/// Path of TPC-H SF=1 `lineitem` in `format`, generating the TPC-H tables if it is missing.
///
/// At SF=1 `lineitem` fits in a single file (see [`split::TARGET_FILE_COMPRESSED_BYTES`]), so this
/// reads the TPC-H benchmark's own data.
async fn ensure_lineitem(format: Format) -> Result<PathBuf> {
    let scale_factor_dir = "tpch".to_data_path().join("1.0");
    let format_dir = scale_factor_dir.join(format.name());
    let path = format_dir.join(split::part_file_name("lineitem", 1, format.ext()));
    if !path.exists() {
        let options = TpchGenOptions::new("1.0".to_string(), scale_factor_dir).with_format(format);
        generate_tpch_tables(options).await?;
    }
    anyhow::ensure!(path.exists(), "No lineitem {format} file found");
    anyhow::ensure!(
        !format_dir
            .join(split::part_file_name("lineitem", 2, format.ext()))
            .exists(),
        "TPC-H SF=1 lineitem is split into several {format} files"
    );
    Ok(path)
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
        ensure_lineitem(Format::Parquet).await
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
        ensure_lineitem(Format::Parquet).await
    }
}
