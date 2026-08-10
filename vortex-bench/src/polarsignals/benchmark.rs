// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! PolarSignals benchmark implementation.

use std::fs;

use anyhow::Result;
use url::Url;

use super::data::generate_polarsignals_parquet;
use super::schema::STACKTRACES_SCHEMA;
use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::Format;
use crate::IdempotentPath;
use crate::SetupCtx;
use crate::TableSpec;
use crate::idempotent;
use crate::workspace_root;

const FILE_NAME: &str = "stacktraces";

/// Benchmark modeled on PolarSignals profiling data to exercise scan-layer
/// performance (projection + filter pushdown) on deeply nested schemas.
pub struct PolarSignalsBenchmark {
    /// Base URL for the dataset location.
    pub data_url: Url,
    /// Scale factor (1 = 1M rows).
    pub scale_factor: usize,
    /// Total number of rows in the dataset.
    pub n_rows: usize,
}

impl PolarSignalsBenchmark {
    /// Creates a new PolarSignalsBenchmark.
    ///
    /// `scale_factor` of 1 produces 1M rows.
    pub fn new(scale_factor: usize) -> Result<Self> {
        let n_rows = scale_factor * 1_000_000;
        let data_path = "polarsignals".to_data_path().join(format!("{n_rows}/"));
        let data_url =
            Url::from_directory_path(data_path).map_err(|_| anyhow::anyhow!("bad data path"))?;

        Ok(Self {
            data_url,
            scale_factor,
            n_rows,
        })
    }
}

#[async_trait::async_trait]
impl Benchmark for PolarSignalsBenchmark {
    fn doc_path(&self) -> &'static str {
        "vortex-bench/sql/polarsignals.md"
    }

    fn queries(&self) -> Result<Vec<(usize, String)>> {
        let queries_file = workspace_root()
            .join("vortex-bench")
            .join("sql")
            .join("polarsignals")
            .with_extension("sql");
        let contents = fs::read_to_string(queries_file)?;
        Ok(contents
            .trim()
            .split_terminator(";")
            .map(str::to_string)
            .enumerate()
            .collect())
    }

    async fn setup(&self, ctx: &SetupCtx, _format: Format) -> Result<()> {
        let n_rows = self.n_rows;
        let parquet_path = ctx.staging().join(format!("{FILE_NAME}.parquet"));
        idempotent(&parquet_path, |tmp_path| {
            tracing::info!(
                n_rows,
                path = %parquet_path.display(),
                "generating PolarSignals parquet data"
            );
            generate_polarsignals_parquet(n_rows, tmp_path)
        })?;
        ctx.emit(FILE_NAME, parquet_path);
        Ok(())
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::PolarSignals {
            n_rows: self.n_rows,
        }
    }

    fn dataset_name(&self) -> &str {
        "polarsignals"
    }

    fn dataset_display(&self) -> String {
        format!("polarsignals(sf={})", self.scale_factor)
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }

    fn table_specs(&self) -> Vec<TableSpec> {
        vec![TableSpec::new(
            "stacktraces",
            Some(STACKTRACES_SCHEMA.clone()),
        )]
    }

    #[expect(clippy::expect_used)]
    fn pattern(&self, _table_name: &str, format: Format) -> Option<glob::Pattern> {
        Some(
            format!("{FILE_NAME}.{}", format.ext())
                .parse()
                .expect("valid glob pattern"),
        )
    }
}
