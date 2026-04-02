// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;

use anyhow::Result;
use async_trait::async_trait;
use tokio::fs::File as TokioFile;
use tokio::io::AsyncWriteExt;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::stream::ArrayStreamExt;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;

use crate::CompactionStrategy;
use crate::Format;
use crate::IdempotentPath;
use crate::SESSION;
use crate::conversions::parquet_to_vortex_chunks;
use crate::datasets::Dataset;
use crate::datasets::data_downloads::download_data;
use crate::idempotent_async;
use crate::random_access::BenchDataset;

/// Dataset identifier used for data path generation.
pub const DATASET: &str = "taxi";

/// Total number of rows in the taxi dataset.
pub const ROW_COUNT: u64 = 3_339_715;

pub struct TaxiData;

#[async_trait]
impl Dataset for TaxiData {
    fn name(&self) -> &str {
        "taxi"
    }

    async fn to_vortex_array(&self) -> Result<ArrayRef> {
        fetch_taxi_data().await
    }

    async fn to_parquet_path(&self) -> Result<PathBuf> {
        taxi_data_parquet().await
    }
}

#[async_trait]
impl BenchDataset for TaxiData {
    fn name(&self) -> &str {
        "taxi"
    }

    fn row_count(&self) -> u64 {
        ROW_COUNT
    }

    async fn path(&self, format: Format) -> Result<PathBuf> {
        match format {
            Format::OnDiskVortex => taxi_data_vortex().await,
            Format::VortexCompact => taxi_data_vortex_compact().await,
            Format::Parquet => taxi_data_parquet().await,
            other => unimplemented!("Random access bench not implemented for {other}"),
        }
    }
}

pub async fn taxi_data_parquet() -> Result<PathBuf> {
    let taxi_parquet_fpath = "taxi/taxi.parquet".to_data_path();
    let taxi_data_url =
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-11.parquet";
    download_data(taxi_parquet_fpath, taxi_data_url).await
}

pub async fn fetch_taxi_data() -> Result<ArrayRef> {
    let vortex_data = taxi_data_vortex().await?;
    Ok(SESSION
        .open_options()
        .open_path(vortex_data)
        .await?
        .scan()?
        .into_array_stream()?
        .read_all()
        .await?)
}

pub async fn taxi_data_vortex() -> Result<PathBuf> {
    idempotent_async("taxi/taxi.vortex", |output_fname| async move {
        let buf = output_fname.to_path_buf();
        let mut output_file = TokioFile::create(output_fname).await?;

        let data = parquet_to_vortex_chunks(taxi_data_parquet().await?).await?;

        SESSION
            .write_options()
            .write(&mut output_file, data.into_array().to_array_stream())
            .await?;
        output_file.flush().await?;
        Ok(buf)
    })
    .await
}

pub async fn taxi_data_vortex_compact() -> Result<PathBuf> {
    idempotent_async("taxi/taxi-compact.vortex", |output_fname| async move {
        let buf = output_fname.to_path_buf();
        let mut output_file = TokioFile::create(output_fname).await?;

        // This is the only difference to `taxi_data_vortex`.
        let write_options = CompactionStrategy::Compact.apply_options(SESSION.write_options());

        let data = parquet_to_vortex_chunks(taxi_data_parquet().await?).await?;

        write_options
            .write(&mut output_file, data.into_array().to_array_stream())
            .await?;

        output_file.flush().await?;
        Ok(buf)
    })
    .await
}
