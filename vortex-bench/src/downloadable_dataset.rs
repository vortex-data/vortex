// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use tokio::fs::File;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::stream::ArrayStreamExt;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;

use crate::IdempotentPath;
use crate::SESSION;
use crate::conversions::parquet_to_vortex_chunks;
use crate::datasets::Dataset;
use crate::datasets::data_downloads::download_data;
use crate::idempotent_async;

/// Datasets which can be downloaded over HTTP in Parquet format.
///
/// # The Pcodec datasets
///
/// Twitter, CMS, and CalHousing are all TSVs, some compressed, so need some pre-processing
///
/// - Taxi. Already in datasets/taxi_data.rs.
/// - California Housing <https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html>.
/// - CMS payments <https://openpaymentsdata.cms.gov/dataset/fb3a65aa-c901-4a38-a813-b04b00dfa2a9>.
/// - Twitter <https://snap.stanford.edu/data/ego-Twitter.html>.
/// - r/place data <https://pcodec-public.s3.amazonaws.com/reddit_2022_place_numerical.parquet> (<https://github.com/pcodec/pcodec/blob/main/docs/benchmark_results.md>).
/// - AirQuality <https://pcodec-public.s3.amazonaws.com/devinrsmith-air-quality.20220714.zstd.parquet>.
pub enum DownloadableDataset {
    RPlace,
    AirQuality,
}

impl DownloadableDataset {
    fn parquet_url(&self) -> &str {
        match self {
            DownloadableDataset::RPlace => {
                "https://pcodec-public.s3.amazonaws.com/reddit_2022_place_numerical.parquet"
            }
            DownloadableDataset::AirQuality => {
                "https://pcodec-public.s3.amazonaws.com/devinrsmith-air-quality.20220714.zstd.parquet"
            }
        }
    }
}

use std::path::PathBuf;

#[async_trait]
impl Dataset for DownloadableDataset {
    fn name(&self) -> &str {
        match self {
            DownloadableDataset::AirQuality => "airquality",
            DownloadableDataset::RPlace => "rplace",
        }
    }

    async fn to_vortex_array(&self) -> anyhow::Result<ArrayRef> {
        let parquet = self.to_parquet_path().await?;
        let dir = format!("{}/", self.name()).to_data_path();
        let vortex = dir.join(format!("{}.vortex", self.name()));

        let data = parquet_to_vortex_chunks(parquet).await?;
        idempotent_async(&vortex, async |path| -> anyhow::Result<()> {
            SESSION
                .write_options()
                .write(
                    &mut File::create(path)
                        .await
                        .map_err(|e| anyhow::anyhow!("Failed to create file: {}", e))?,
                    data.into_array().to_array_stream(),
                )
                .await
                .map_err(|e| anyhow::anyhow!("Failed to write vortex file: {}", e))?;
            Ok(())
        })
        .await?;

        Ok(SESSION
            .open_options()
            .open_path(vortex.as_path())
            .await?
            .scan()?
            .into_array_stream()?
            .read_all()
            .await?)
    }

    async fn to_parquet_path(&self) -> anyhow::Result<PathBuf> {
        let dir = format!("{}/", self.name()).to_data_path();
        let parquet = dir.join(format!("{}.parquet", self.name()));
        download_data(parquet.clone(), self.parquet_url()).await?;
        Ok(parquet)
    }
}
