// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! ALP compression benchmark implementation.

use std::fs;

use anyhow::Result;
use arrow_schema::Schema;
use url::Url;

use super::data::{alp_floats_schema, generate_alp_floats_parquet};
use crate::benchmark::TableSpec;
use crate::{Benchmark, BenchmarkDataset, Format, IdempotentPath, idempotent, workspace_root};

const TABLE_NAME: &str = "alp_floats";

/// Default number of rows (2 million).
const DEFAULT_N_ROWS: usize = 2_000_000;

/// SQL benchmark exercising ALP-RD compression on synthetic f64 columns.
pub struct AlpCompressBenchmark {
    data_url: Url,
    n_rows: usize,
    schema: Schema,
}

impl AlpCompressBenchmark {
    /// Create a new benchmark with the given scale factor (1 = 2M rows).
    pub fn new(scale_factor: usize) -> Result<Self> {
        let n_rows = scale_factor * DEFAULT_N_ROWS;
        let data_path = "alp_compress".to_data_path().join(format!("{n_rows}/"));
        let data_url =
            Url::from_directory_path(data_path).map_err(|_| anyhow::anyhow!("bad data path"))?;
        let schema = (*alp_floats_schema()).clone();

        Ok(Self {
            data_url,
            n_rows,
            schema,
        })
    }

    fn parquet_path(&self) -> Result<std::path::PathBuf> {
        self.data_url
            .join("parquet/")?
            .join(&format!("{TABLE_NAME}.parquet"))?
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("failed to convert data URL to filesystem path"))
    }
}

#[async_trait::async_trait]
impl Benchmark for AlpCompressBenchmark {
    fn queries(&self) -> Result<Vec<(usize, String)>> {
        let queries_file = workspace_root()
            .join("vortex-bench")
            .join("alp_compress")
            .with_extension("sql");
        let contents = fs::read_to_string(queries_file)?;
        Ok(contents
            .trim()
            .split_terminator(';')
            .map(str::to_string)
            .enumerate()
            .collect())
    }

    async fn generate_base_data(&self) -> Result<()> {
        if self.data_url.scheme() != "file" {
            anyhow::bail!(
                "unsupported URL scheme '{}' - only 'file://' URLs are supported",
                self.data_url.scheme()
            );
        }

        let n_rows = self.n_rows;
        let parquet_path = self.parquet_path()?;
        idempotent(&parquet_path, |tmp_path| {
            tracing::info!(
                n_rows,
                path = %parquet_path.display(),
                "generating ALP compress benchmark parquet data"
            );
            generate_alp_floats_parquet(n_rows, tmp_path)
        })?;
        Ok(())
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::AlpCompress {
            n_rows: self.n_rows,
        }
    }

    fn dataset_name(&self) -> &str {
        "alp-compress"
    }

    fn dataset_display(&self) -> String {
        format!("alp-compress(n_rows={})", self.n_rows)
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }

    fn table_specs(&self) -> Vec<TableSpec> {
        vec![TableSpec::new(TABLE_NAME, Some(self.schema.clone()))]
    }

    #[expect(clippy::expect_used, clippy::unwrap_in_result)]
    fn pattern(&self, _table_name: &str, format: Format) -> Option<glob::Pattern> {
        Some(
            format!("{TABLE_NAME}.{}", format.ext())
                .parse()
                .expect("valid glob pattern"),
        )
    }
}
