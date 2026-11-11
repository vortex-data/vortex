// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::path::PathBuf;

use anyhow::anyhow;
use async_trait::async_trait;
use lance::Dataset;
use lance::dataset::WriteParams;
use lance::dataset::fragment::FileFragment;
use lance::datatypes::Schema;
use lance_file::version::LanceFileVersion;
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

        // Write Lance's current stable format into a single fragment, so a take reads one file
        // like the Vortex and Parquet accessors do instead of opening a reader per fragment.
        let write_params = WriteParams {
            max_rows_per_file: usize::MAX,
            ..WriteParams::with_storage_version(LanceFileVersion::Stable)
        };
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
/// Like the Vortex and Parquet accessors, this reads the dataset's single file: rows are taken
/// straight from its one fragment, opened once at construction and reused across `take()` calls.
/// Going through `Dataset::take` would instead rebuild a projection plan and map dataset offsets
/// to fragments on every call.
pub struct LanceRandomAccessor {
    name: String,
    fragment: FileFragment,
    schema: Schema,
}

impl LanceRandomAccessor {
    /// Open a Lance dataset and return a ready-to-use accessor.
    pub async fn open(path: PathBuf, name: impl Into<String>) -> anyhow::Result<Self> {
        Self::open_uri(
            path.to_str()
                .ok_or_else(|| anyhow!("Invalid dataset path"))?,
            name,
        )
        .await
    }

    /// Open a Lance dataset from any URI (local path, `s3://...`, ...).
    pub async fn open_uri(uri: &str, name: impl Into<String>) -> anyhow::Result<Self> {
        let dataset = Dataset::open(uri).await?;
        let schema = dataset.schema().clone();
        let fragment = match <[_; 1]>::try_from(dataset.get_fragments()) {
            Ok([fragment]) => fragment,
            Err(fragments) => anyhow::bail!(
                "Lance random access expects one fragment, found {}",
                fragments.len()
            ),
        };
        Ok(Self {
            name: name.into(),
            fragment,
            schema,
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
        let indices = indices
            .iter()
            .map(|&idx| u32::try_from(idx))
            .collect::<Result<Vec<_>, _>>()?;
        let batch = self.fragment.take(&indices, &self.schema).await?;
        Ok(RandomAccessorRet::Native(Box::new(batch)))
    }
}
