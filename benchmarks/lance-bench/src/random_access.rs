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
use vortex_bench::datasets::taxi_data::taxi_data_parquet;
use vortex_bench::idempotent_async;
use vortex_bench::random_access::RandomAccessor;

pub async fn taxi_data_lance() -> anyhow::Result<PathBuf> {
    idempotent_async("taxi/taxi.lance", |output_fname| async move {
        let parquet_path = taxi_data_parquet().await?;

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

pub struct LanceRandomAccessor {
    path: PathBuf,
}

impl LanceRandomAccessor {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

#[async_trait]
impl RandomAccessor for LanceRandomAccessor {
    fn format(&self) -> Format {
        Format::Lance
    }

    fn name(&self) -> &str {
        "random-access/lance-tokio-local-disk"
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
