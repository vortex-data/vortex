// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The first file of the partitioned ClickBench `hits` table as a compression dataset.

use std::path::PathBuf;

use anyhow::Result;
use async_trait::async_trait;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;

use crate::Format;
use crate::IdempotentPath;
use crate::clickbench::CLICKBENCH_PARTITIONED_NAME;
use crate::clickbench::Flavor;
use crate::conversions::parquet_to_vortex_chunks;
use crate::datasets::Dataset;

/// `hits_0.parquet` from the partitioned ClickBench dataset: about 1M rows and 105 columns.
///
/// The full dataset has about 100M rows, which is too large to hold in memory for every format
/// at once. Each partitioned file is a contiguous slice of `hits`, not a random sample.
pub struct ClickBenchHits0;

#[async_trait]
impl Dataset for ClickBenchHits0 {
    fn name(&self) -> &str {
        // Kept apart from `clickbench`, which names the full dataset on the website.
        "clickbench hits_0"
    }

    async fn to_vortex_array(&self, _ctx: &mut ExecutionCtx) -> Result<ArrayRef> {
        let parquet = self.to_parquet_path().await?;
        Ok(parquet_to_vortex_chunks(parquet).await?.into_array())
    }

    async fn to_parquet_path(&self) -> Result<PathBuf> {
        let base = CLICKBENCH_PARTITIONED_NAME.to_data_path();
        Flavor::Partitioned.download(&base, 1).await?;
        let path = base.join(Format::Parquet.name()).join("hits_0.parquet");
        anyhow::ensure!(path.exists(), "No ClickBench hits_0 Parquet file found");
        Ok(path)
    }
}
