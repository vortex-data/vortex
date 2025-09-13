// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;

use anyhow::Result;
use async_trait::async_trait;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use vortex::ArrayRef;
use vortex::file::{VortexOpenOptions, VortexWriteOptions};
use vortex::iter::ArrayIteratorExt;

use crate::conversions::parquet_to_vortex;
use crate::datasets::Dataset;
use crate::datasets::data_downloads::download_data;
use crate::{IdempotentPath, idempotent_async};

pub struct TaxiData;

#[async_trait]
impl Dataset for TaxiData {
    fn name(&self) -> &str {
        "taxi"
    }

    async fn to_vortex_array(&self) -> Result<ArrayRef> {
        fetch_taxi_data().await
    }
}

pub async fn taxi_data_parquet() -> Result<PathBuf> {
    let taxi_parquet_fpath = "yellow-tripdata-2023-11.parquet".to_data_path();
    let taxi_data_url =
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-11.parquet";
    download_data(taxi_parquet_fpath, taxi_data_url).await
}

pub async fn fetch_taxi_data() -> Result<ArrayRef> {
    let vortex_data = taxi_data_vortex().await?;
    Ok(VortexOpenOptions::new()
        .open(vortex_data)
        .await?
        .scan()?
        .into_array_iter_multithread()?
        .read_all()?)
}

pub async fn taxi_data_vortex() -> Result<PathBuf> {
    idempotent_async("taxi.vortex", |output_fname| async move {
        let buf = output_fname.to_path_buf();
        let mut output_file = File::create(output_fname).await?;
        VortexWriteOptions::default()
            .write(
                &mut output_file,
                parquet_to_vortex(taxi_data_parquet().await?)?,
            )
            .await?;
        output_file.flush().await?;
        Ok(buf)
    })
    .await
}
