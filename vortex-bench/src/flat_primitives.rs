// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::sync::Arc;

use anyhow::Result;
use arrow_array::RecordBatch;
use arrow_array::UInt64Array;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use parquet::arrow::ArrowWriter;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use url::Url;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::IdempotentPath;
use crate::TableSpec;
use crate::utils::file::idempotent_async;

const DEFAULT_FILES: usize = 10;
const DEFAULT_ROWS_PER_FILE: usize = 5_000_000;
const DEFAULT_COLUMNS: usize = 1;
const BATCH_ROWS: usize = 1_000_000;
const FILES_KEY: &str = "files";
const ROWS_PER_FILE_KEY: &str = "rows-per-file";
const COLUMNS_KEY: &str = "columns";

pub struct FlatPrimitivesBenchmark {
    files: usize,
    rows_per_file: usize,
    columns: usize,
    data_url: Url,
}

impl FlatPrimitivesBenchmark {
    pub fn new(files: usize, rows_per_file: usize, columns: usize) -> Result<Self> {
        let files = files.max(1);
        let rows_per_file = rows_per_file.max(1);
        let columns = columns.max(1);
        let dirname = format!(
            "flat_primitives/c{}_f{}_rpf{}",
            columns, files, rows_per_file
        );
        let path = dirname.to_data_path();
        let data_url = Url::from_directory_path(&path)
            .map_err(|_| anyhow::anyhow!("Failed to create URL from directory path: {path:?}"))?;
        Ok(Self {
            files,
            rows_per_file,
            columns,
            data_url,
        })
    }

    pub fn from_opts(opts: &crate::Opts) -> Result<Self> {
        let files = opts.get_as::<usize>(FILES_KEY).unwrap_or(DEFAULT_FILES);
        let rows_per_file = opts
            .get_as::<usize>(ROWS_PER_FILE_KEY)
            .unwrap_or(DEFAULT_ROWS_PER_FILE);
        let columns = opts.get_as::<usize>(COLUMNS_KEY).unwrap_or(DEFAULT_COLUMNS);
        Self::new(files, rows_per_file, columns)
    }

    fn schema(&self) -> Schema {
        Schema::new(
            (0..self.columns)
                .map(|idx| Field::new(format!("value_{idx}"), DataType::UInt64, false))
                .collect::<Vec<_>>(),
        )
    }
}

#[async_trait::async_trait]
impl Benchmark for FlatPrimitivesBenchmark {
    fn queries(&self) -> Result<Vec<(usize, String)>> {
        Ok(Vec::new())
    }

    async fn generate_base_data(&self) -> Result<()> {
        let base_data_dir = self
            .data_url
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Invalid file URL: {}", self.data_url))?;
        let parquet_dir = base_data_dir.join(crate::Format::Parquet.name());
        tokio::fs::create_dir_all(&parquet_dir).await?;

        let schema = Arc::new(self.schema());

        for file_idx in 0..self.files {
            let file_path = parquet_dir.join(format!("flat_{file_idx}.parquet"));
            let schema = schema.clone();
            let rows_per_file = self.rows_per_file;
            let columns = self.columns;
            idempotent_async(file_path, move |output_path| async move {
                let file = File::create(&output_path)?;
                let mut writer = ArrowWriter::try_new(file, schema.clone(), None)?;
                let mut rng = StdRng::seed_from_u64(42 + u64::try_from(file_idx).unwrap_or(0));

                for batch_start in (0..rows_per_file).step_by(BATCH_ROWS) {
                    let batch_len = BATCH_ROWS.min(rows_per_file - batch_start);
                    let arrays = (0..columns)
                        .map(|_| {
                            Arc::new(UInt64Array::from_iter_values(
                                (0..batch_len).map(|_| rng.random::<u64>()),
                            )) as Arc<dyn arrow_array::Array>
                        })
                        .collect::<Vec<_>>();
                    let batch = RecordBatch::try_new(schema.clone(), arrays)?;
                    writer.write(&batch)?;
                }

                writer.close()?;
                Ok(())
            })
            .await?;
        }

        Ok(())
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::FlatPrimitives {
            files: self.files,
            rows_per_file: self.rows_per_file,
            columns: self.columns,
        }
    }

    fn dataset_name(&self) -> &str {
        "flat-primitives"
    }

    fn dataset_display(&self) -> String {
        format!(
            "flat-primitives(c={},files={},rows-per-file={})",
            self.columns, self.files, self.rows_per_file
        )
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }

    fn table_specs(&self) -> Vec<TableSpec> {
        vec![TableSpec::new("flat", Some(self.schema()))]
    }
}
