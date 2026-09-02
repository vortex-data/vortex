// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::path::PathBuf;

use anyhow::Result;
use anyhow::bail;
use async_trait::async_trait;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;

use crate::Format;
use crate::conversions::parquet_to_vortex_chunks;
use crate::datasets::Dataset;
use crate::random_access::BenchDataset;

/// A local Parquet file for compression and random-access benchmarks.
pub struct LocalParquetData {
    name: String,
    path: PathBuf,
    row_count: u64,
}

impl LocalParquetData {
    /// Read the dataset name and row count from a local Parquet file.
    pub fn try_new(path: PathBuf) -> Result<Self> {
        let name = path
            .file_stem()
            .and_then(|value| value.to_str())
            .ok_or_else(|| anyhow::anyhow!("local Parquet path has no valid file stem"))?
            .to_string();
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&path)?)?;
        let row_count = u64::try_from(reader.metadata().file_metadata().num_rows())?;
        Ok(Self {
            name,
            path,
            row_count,
        })
    }
}

#[async_trait]
impl Dataset for LocalParquetData {
    fn name(&self) -> &str {
        &self.name
    }

    async fn to_vortex_array(&self, _ctx: &mut ExecutionCtx) -> Result<ArrayRef> {
        Ok(parquet_to_vortex_chunks(self.path.clone()).await?.into())
    }

    async fn to_parquet_path(&self) -> Result<PathBuf> {
        Ok(self.path.clone())
    }
}

#[async_trait]
impl BenchDataset for LocalParquetData {
    fn name(&self) -> &str {
        &self.name
    }

    fn row_count(&self) -> u64 {
        self.row_count
    }

    async fn path(&self, format: Format) -> Result<PathBuf> {
        if format == Format::Parquet {
            return Ok(self.path.clone());
        }
        bail!("local Parquet dataset does not provide {format}")
    }
}
