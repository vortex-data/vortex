// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::datasets::Dataset;
use crate::tpch::tpchgen::{generate_tpch_tables, TpchGenOptions};
use crate::{Format, IdempotentPath, SESSION};
use anyhow::Result;
use async_trait::async_trait;
use futures::TryStreamExt;
use glob::glob;
use vortex::arrays::ChunkedArray;
use vortex::dtype::Nullability::NonNullable;
use vortex::expr::{col, pack};
use vortex::file::OpenOptionsSessionExt;
use vortex::{Array, ArrayRef, IntoArray, ToCanonical};

pub struct TPCHLCommentChunked;

#[async_trait]
impl Dataset for TPCHLCommentChunked {
    fn name(&self) -> &str {
        "TPC-H l_comment chunked"
    }

    async fn to_vortex_array(&self) -> Result<ArrayRef> {
        let base_path = "tpch".to_data_path();
        let scale_factor_dir = base_path.join("1.0");
        let data_dir = scale_factor_dir.join(Format::OnDiskVortex.name());

        // Generate TPC-H CSV data if it doesn't exist
        if !data_dir.exists() {
            // Use blocking call like TPC-H benchmark does
            let options = TpchGenOptions::new("1.0".to_string(), scale_factor_dir)
                .with_format(Format::OnDiskVortex);

            futures::executor::block_on(generate_tpch_tables(options))?;
        }

        let mut chunks: Vec<ArrayRef> = vec![];
        for path in glob(
            data_dir
                .join("lineitem_*.vortex")
                .to_string_lossy()
                .as_ref(),
        )? {
            let file = SESSION.open_options().open(path?).await?;
            let file_chunks: Vec<_> = file
                .scan()?
                .with_projection(pack(vec![("l_comment", col("l_comment"))], NonNullable))
                .map(|a| Ok(a.to_canonical().into_array()))
                .into_array_stream()?
                .try_collect()
                .await?;
            chunks.extend(file_chunks);
        }

        Ok(ChunkedArray::from_iter(chunks).into_array())
    }
}

pub struct TPCHLCommentCanonical;

#[async_trait]
impl Dataset for TPCHLCommentCanonical {
    fn name(&self) -> &str {
        "TPC-H l_comment canonical"
    }

    async fn to_vortex_array(&self) -> Result<ArrayRef> {
        let comments_canonical = TPCHLCommentChunked
            .to_vortex_array()
            .await?
            .to_struct()
            .into_array();
        Ok(ChunkedArray::from_iter([comments_canonical]).into_array())
    }
}
