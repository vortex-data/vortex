// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;

use anyhow::Result;
use async_trait::async_trait;
use tokio::fs::File as TokioFile;
use tokio::io::AsyncWriteExt;
use vortex::array::ArrayRef;
use vortex::array::stream::ArrayStreamExt;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
#[cfg(feature = "lance")]
#[rustfmt::skip]
use {
    lance::dataset::Dataset as LanceDataset,
    lance::dataset::WriteParams,
    lance_encoding::version::LanceFileVersion,
    parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder,
    std::fs::File,
};

use crate::CompactionStrategy;
use crate::IdempotentPath;
use crate::SESSION;
use crate::conversions::parquet_to_vortex;
use crate::datasets::Dataset;
use crate::datasets::data_downloads::download_data;
use crate::idempotent_async;

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
    let taxi_parquet_fpath = "taxi/taxi.parquet".to_data_path();
    let taxi_data_url =
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-11.parquet";
    download_data(taxi_parquet_fpath, taxi_data_url).await
}

pub async fn fetch_taxi_data() -> Result<ArrayRef> {
    let vortex_data = taxi_data_vortex().await?;
    Ok(SESSION
        .open_options()
        .open(vortex_data)
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
        SESSION
            .write_options()
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

pub async fn taxi_data_vortex_compact() -> Result<PathBuf> {
    idempotent_async("taxi/taxi-compact.vortex", |output_fname| async move {
        let buf = output_fname.to_path_buf();
        let mut output_file = TokioFile::create(output_fname).await?;

        // This is the only difference to `taxi_data_vortex`.
        let write_options = CompactionStrategy::Compact.apply_options(SESSION.write_options());

        write_options
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

#[cfg(feature = "lance")]
pub async fn taxi_data_lance() -> Result<PathBuf> {
    idempotent_async("taxi/taxi.lance", |output_fname| async move {
        let parquet_path = taxi_data_parquet().await?;

        let file = File::open(&parquet_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let reader = builder.build()?;

        let write_params = WriteParams::with_storage_version(LanceFileVersion::V2_1);
        LanceDataset::write(reader, output_fname.to_str().unwrap(), Some(write_params)).await?;

        Ok(output_fname.to_path_buf())
    })
    .await
}
