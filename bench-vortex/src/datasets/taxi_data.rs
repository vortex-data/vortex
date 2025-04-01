use std::path::PathBuf;

use async_trait::async_trait;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tokio::runtime::Handle;
use vortex::ArrayRef;
use vortex::error::VortexError;
use vortex::file::{VortexOpenOptions, VortexWriteOptions};
use vortex::io::TokioFile;
use vortex::stream::ArrayStreamExt;

use crate::conversions::parquet_to_vortex;
use crate::datasets::BenchmarkDataset;
use crate::datasets::data_downloads::download_data;
use crate::{IdempotentPath, idempotent_async};

pub struct TaxiData;

#[async_trait]
impl BenchmarkDataset for TaxiData {
    fn name(&self) -> &str {
        "taxi"
    }

    async fn to_vortex_array(&self) -> ArrayRef {
        fetch_taxi_data().await
    }
}

pub async fn taxi_data_parquet() -> PathBuf {
    let taxi_parquet_fpath = "yellow-tripdata-2023-11.parquet".to_data_path();
    let taxi_data_url =
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-11.parquet";
    download_data(taxi_parquet_fpath, taxi_data_url).await
}

pub async fn fetch_taxi_data() -> ArrayRef {
    let vortex_data = taxi_data_vortex().await;
    VortexOpenOptions::file()
        .open(TokioFile::open(vortex_data).unwrap())
        .await
        .unwrap()
        .scan()
        .unwrap()
        .spawn_tokio(Handle::current())
        .unwrap()
        .read_all()
        .await
        .unwrap()
}

pub async fn taxi_data_vortex() -> PathBuf {
    idempotent_async("taxi.vortex", |output_fname| async move {
        let buf = output_fname.to_path_buf();
        let output_file = File::create(output_fname).await?;
        VortexWriteOptions::default()
            .write(
                output_file,
                parquet_to_vortex(taxi_data_parquet().await).unwrap(),
            )
            .await?
            .flush()
            .await?;
        Ok::<PathBuf, VortexError>(buf)
    })
    .await
    .unwrap()
}
