// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::path::PathBuf;

use anyhow::anyhow;
use async_trait::async_trait;
use lance::Dataset;
use lance::dataset::ProjectionRequest;
use lance::dataset::WriteParams;
use lance_encoding::version::LanceFileVersion;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex_bench::Format;
use vortex_bench::datasets::feature_vectors;
use vortex_bench::datasets::nested_lists;
use vortex_bench::datasets::nested_structs;
use vortex_bench::datasets::taxi_data;
use vortex_bench::idempotent_async;
use vortex_bench::random_access::RandomAccessor;
use vortex_bench::random_access::RandomAccessorRet;
use vortex_bench::random_access::data_path;

/// Convert a parquet file to lance format.
///
/// Uses `idempotent_async` to skip conversion if the output already exists.
async fn parquet_to_lance_file(parquet_path: PathBuf, lance_path: &str) -> anyhow::Result<PathBuf> {
    idempotent_async(lance_path, |output_fname| async move {
        let file = File::open(&parquet_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let reader = builder.build()?;

        let write_params = WriteParams::with_storage_version(LanceFileVersion::V2_1);
        Dataset::write(
            reader,
            output_fname
                .to_str()
                .ok_or_else(|| anyhow!("Invalid output file path"))?,
            Some(write_params),
        )
        .await?;

        Ok(output_fname.to_path_buf())
    })
    .await
}

pub async fn taxi_data_lance() -> anyhow::Result<PathBuf> {
    let parquet_path = taxi_data::taxi_data_parquet().await?;
    parquet_to_lance_file(parquet_path, &data_path(taxi_data::DATASET, Format::Lance)).await
}

pub async fn feature_vectors_lance() -> anyhow::Result<PathBuf> {
    let parquet_path = feature_vectors::feature_vectors_parquet().await?;
    parquet_to_lance_file(
        parquet_path,
        &data_path(feature_vectors::DATASET, Format::Lance),
    )
    .await
}

pub async fn nested_lists_lance() -> anyhow::Result<PathBuf> {
    let parquet_path = nested_lists::nested_lists_parquet().await?;
    parquet_to_lance_file(
        parquet_path,
        &data_path(nested_lists::DATASET, Format::Lance),
    )
    .await
}

pub async fn nested_structs_lance() -> anyhow::Result<PathBuf> {
    let parquet_path = nested_structs::nested_structs_parquet().await?;
    parquet_to_lance_file(
        parquet_path,
        &data_path(nested_structs::DATASET, Format::Lance),
    )
    .await
}

/// Random accessor for Lance format files.
///
/// The dataset handle is opened at construction time and reused across `take()` calls.
pub struct LanceRandomAccessor {
    name: String,
    dataset: Dataset,
    projection: ProjectionRequest,
}

impl LanceRandomAccessor {
    /// Open a Lance dataset and return a ready-to-use accessor.
    pub async fn open(path: PathBuf, name: impl Into<String>) -> anyhow::Result<Self> {
        let dataset = Dataset::open(
            path.to_str()
                .ok_or_else(|| anyhow!("Invalid dataset path"))?,
        )
        .await?;
        let projection = ProjectionRequest::from_schema(dataset.schema().clone());
        Ok(Self {
            name: name.into(),
            dataset,
            projection,
        })
    }
}

#[async_trait]
impl RandomAccessor for LanceRandomAccessor {
    fn format(&self) -> Format {
        Format::Lance
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn take(&self, indices: &[u64]) -> anyhow::Result<RandomAccessorRet> {
        Ok(RandomAccessorRet::RecordBatch(
            self.dataset.take(indices, self.projection.clone()).await?,
        ))
    }
}
