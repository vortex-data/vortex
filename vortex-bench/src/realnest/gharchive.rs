// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Data from the GitHub Archive.
//!
//! This dataset applies a bunch of events this way

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use tracing::info;
use url::Url;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::Format;
use crate::SetupCtx;
use crate::TableSpec;
use crate::idempotent;
use crate::utils::file::resolve_data_url;
use crate::workspace_root;

/// Template URL for raw JSON dataset
fn raw_json_url(hour: usize) -> String {
    assert!(hour <= 23);
    format!("https://data.gharchive.org/2024-10-01-{hour}.json.gz")
}

pub struct GithubArchiveBenchmark {
    data_url: Url,
}

impl GithubArchiveBenchmark {
    pub fn new(data_url: Url) -> Self {
        Self { data_url }
    }

    pub fn with_remote_data_dir(use_remote_data_dir: Option<String>) -> anyhow::Result<Self> {
        let data_url = Self::create_data_url(use_remote_data_dir.as_deref())?;
        Ok(Self { data_url })
    }

    fn create_data_url(remote_data_dir: Option<&str>) -> anyhow::Result<Url> {
        resolve_data_url(remote_data_dir, "gharchive")
    }
}

impl GithubArchiveBenchmark {
    fn json_dir(&self) -> anyhow::Result<PathBuf> {
        self.data_url
            .join("json/")?
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Failed to convert data URL to filesystem path - ensure data_url uses 'file://' scheme"))
    }
}

#[async_trait::async_trait]
impl Benchmark for GithubArchiveBenchmark {
    fn doc_path(&self) -> &'static str {
        "vortex-bench/sql/gharchive.md"
    }

    /// GitHub Archive queries, numbered from Q0 in `sql/gharchive.sql` file order.
    fn queries(&self) -> anyhow::Result<Vec<(usize, String)>> {
        // `;`-separated; a `;` must not appear in a comment, or it would split a statement in two.
        let queries_file = workspace_root()
            .join("vortex-bench")
            .join("sql")
            .join("gharchive")
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

    async fn setup(&self, ctx: &SetupCtx, _format: Format) -> anyhow::Result<()> {
        // One file per hour, fetched in parallel through the shared pool. Each lands
        // separately so a failure re-fetches only the missing hours; DuckDB then reads the
        // whole set through a glob.
        let json_dir = self.json_dir()?;
        fs::create_dir_all(&json_dir)?;
        let downloads: Vec<(String, PathBuf)> = (0..=23)
            .map(|hour| {
                (
                    raw_json_url(hour),
                    json_dir.join(format!("2024-10-01-{hour}.json.gz")),
                )
            })
            .collect();
        info!(
            "Downloading {} GithubArchive JSON source files",
            downloads.len()
        );
        ctx.download(downloads).await?;

        let json_glob = json_dir.join("*.json.gz").display().to_string();

        let output_path = ctx.staging().join("events.parquet");
        let parquet = idempotent(&output_path, move |parquet_path| {
            let parquet = parquet_path.display().to_string();
            info!(
                "Converting GithubArchive JSON to Parquet with DuckDB @ {}",
                parquet_path.display()
            );
            let result = Command::new("duckdb")
                .arg("-c")
                .arg(format!(
                    "
                    CREATE TABLE events AS select * from read_ndjson_auto('{json_glob}', ignore_errors = true);
                    COPY events TO '{parquet}' (FORMAT parquet);
                    "
                ))
                .spawn()?
                .wait()?;

            if !result.success() {
                anyhow::bail!("DuckDB subprocess failed converting JSON to Parquet");
            }

            Ok(())
        })?;

        info!("gharchive base data generated in {}", parquet.display());
        ctx.emit("events", parquet);

        Ok(())
    }

    fn expected_row_counts(&self) -> Option<Vec<usize>> {
        Some(vec![1, 2, 100, 10, 82468])
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::GhArchive
    }

    fn dataset_name(&self) -> &str {
        "gharchive"
    }

    fn dataset_display(&self) -> String {
        "gharchive".to_owned()
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }

    fn table_specs(&self) -> Vec<TableSpec> {
        vec![TableSpec::new("events", None)]
    }
}
