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
use vortex_bench::datasets::feature_vectors::feature_vectors_parquet;
use vortex_bench::datasets::nested_lists::nested_lists_parquet;
use vortex_bench::datasets::nested_structs::nested_structs_parquet;
use vortex_bench::datasets::taxi_data::taxi_data_parquet;
use vortex_bench::idempotent_async;
use vortex_bench::random_access::RandomAccessor;

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
    let parquet_path = taxi_data_parquet().await?;
    parquet_to_lance_file(parquet_path, "taxi/taxi.lance").await
}

pub async fn feature_vectors_lance() -> anyhow::Result<PathBuf> {
    let parquet_path = feature_vectors_parquet().await?;
    parquet_to_lance_file(parquet_path, "feature_vectors/feature_vectors.lance").await
}

pub async fn nested_lists_lance() -> anyhow::Result<PathBuf> {
    let parquet_path = nested_lists_parquet().await?;
    parquet_to_lance_file(parquet_path, "nested_lists/nested_lists.lance").await
}

pub async fn nested_structs_lance() -> anyhow::Result<PathBuf> {
    let parquet_path = nested_structs_parquet().await?;
    parquet_to_lance_file(parquet_path, "nested_structs/nested_structs.lance").await
}

pub struct LanceRandomAccessor {
    path: PathBuf,
    name: String,
}

impl LanceRandomAccessor {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            name: "random-access/lance-tokio-local-disk".to_string(),
        }
    }

    /// Create a new Lance random accessor with a custom name.
    pub fn with_name(path: PathBuf, name: impl Into<String>) -> Self {
        Self {
            path,
            name: name.into(),
        }
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

    fn path(&self) -> &PathBuf {
        &self.path
    }

    async fn take(&self, indices: Vec<u64>) -> anyhow::Result<usize> {
        let dataset = Dataset::open(
            self.path
                .to_str()
                .ok_or_else(|| anyhow!("Invalid dataset path"))?,
        )
        .await?;
        let projection = ProjectionRequest::from_schema(dataset.schema().clone()); // All columns.
        let result = dataset.take(indices.as_slice(), projection).await?;
        Ok(result.num_rows())
    }
}
