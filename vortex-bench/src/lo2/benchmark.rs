// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! LO2 benchmark implementation.

use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use url::Url;

use super::data::ARCHIVE_URL;
use super::data::convert_to_parquet;
use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::Format;
use crate::IdempotentPath;
use crate::TableSpec;
use crate::datasets::data_downloads::download_data;
use crate::idempotent;
use crate::workspace_root;

const DATASET_NAME: &str = "lo2";
const TABLE_NAME: &str = "samples";

/// Benchmark over real node_exporter and cAdvisor metrics in Prometheus layout, running real
/// recording rules, alerts, dashboard panels and prombench queries translated to SQL.
pub struct Lo2Benchmark {
    data_url: Url,
}

impl Lo2Benchmark {
    pub fn new() -> Result<Self> {
        let data_path = DATASET_NAME.to_data_path();
        let data_url =
            Url::from_directory_path(data_path).map_err(|_| anyhow::anyhow!("bad data path"))?;
        Ok(Self { data_url })
    }

    fn file_path(&self, relative: &str) -> Result<PathBuf> {
        self.data_url
            .join(relative)?
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("failed to convert data URL to filesystem path"))
    }
}

#[async_trait::async_trait]
impl Benchmark for Lo2Benchmark {
    fn doc_path(&self) -> &'static str {
        "vortex-bench/sql/lo2.md"
    }

    /// Queries numbered from Q0 in `sql/lo2.sql` file order.
    fn queries(&self) -> Result<Vec<(usize, String)>> {
        // `;`-separated, so a `;` must not appear in a comment.
        let queries_file = workspace_root()
            .join("vortex-bench")
            .join("sql")
            .join(DATASET_NAME)
            .with_extension("sql");
        let contents = fs::read_to_string(queries_file)?;
        Ok(contents
            .split_terminator(';')
            .map(str::trim)
            .filter(|stmt| !stmt.is_empty())
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

        let archive_path =
            download_data(self.file_path("raw/light-oauth2-metrics.zip")?, ARCHIVE_URL).await?;

        let parquet_path = self.file_path(&format!("parquet/{TABLE_NAME}.parquet"))?;
        idempotent(&parquet_path, |tmp_path| {
            tracing::info!(
                path = %parquet_path.display(),
                "converting LO2 metrics to Prometheus layout"
            );
            convert_to_parquet(&archive_path, tmp_path)
        })?;
        Ok(())
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::Lo2
    }

    fn dataset_name(&self) -> &str {
        DATASET_NAME
    }

    fn dataset_display(&self) -> String {
        DATASET_NAME.to_string()
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }

    /// The label struct holds every label key found in the data, so the schema is read from
    /// the generated files rather than declared here.
    fn table_specs(&self) -> Vec<TableSpec> {
        vec![TableSpec::new(TABLE_NAME, None)]
    }

    #[expect(clippy::expect_used)]
    fn pattern(&self, _table_name: &str, format: Format) -> Option<glob::Pattern> {
        Some(
            format!("{TABLE_NAME}.{}", format.ext())
                .parse()
                .expect("valid glob pattern"),
        )
    }
}
