// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Data from the GitHub Archive.
//!
//! This dataset applies a bunch of events this way

use std::path::PathBuf;
use std::process::Command;

use tokio::io::AsyncWriteExt;
use tracing::info;
use url::Url;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::TableSpec;
use crate::idempotent;
use crate::idempotent_async;
use crate::utils::file::resolve_data_url;

/// Template URL for raw JSON dataset
fn raw_json_url(hour: usize) -> String {
    assert!(hour <= 23);
    format!("https://data.gharchive.org/2024-10-01-{hour}.json.gz")
}

const QUERIES: &[&str] = &[
    "select count(*) from events where payload.ref = 'refs/heads/main'",
    "select distinct repo.name from events where repo.name like 'spiraldb/%'",
    "select distinct org.id as org_id from events order by org_id limit 100",
    "select actor.login, count() as freq from events group by actor.login order by freq desc limit 10",
    "select actor.avatar_url from events where actor.login = 'renovate[bot]'",
];

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
    fn json_path(&self) -> anyhow::Result<PathBuf> {
        self.data_url
            .join("json/")?
            .join("events.json.gz")?
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Failed to convert data URL to filesystem path - ensure data_url uses 'file://' scheme"))
    }

    fn parquet_path(&self) -> anyhow::Result<PathBuf> {
        self.data_url
            .join("parquet/")?
            .join("events.parquet")?
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Failed to convert data URL to filesystem path - ensure data_url uses 'file://' scheme"))
    }
}

#[async_trait::async_trait]
impl Benchmark for GithubArchiveBenchmark {
    fn queries(&self) -> anyhow::Result<Vec<(usize, String)>> {
        Ok(QUERIES.iter().map(|s| s.to_string()).enumerate().collect())
    }

    async fn generate_base_data(&self) -> anyhow::Result<()> {
        if self.data_url.scheme() != "file" {
            return Ok(());
        }

        let json = idempotent_async(&self.json_path()?, |json_path| async move {
            info!("Downloading GithubArchive JSON source files");
            let mut w = tokio::fs::File::create(json_path).await?;
            let client = reqwest::Client::new();
            for hour in 0..=23 {
                let url = raw_json_url(hour);
                info!("Downloading archive {url}");
                let response = client
                    .get(url)
                    .send()
                    .await?
                    .error_for_status()
                    .map_err(|err| anyhow::anyhow!("error fetching gharchive data: {err}"))?;

                let body = response.bytes().await?;

                w.write_all(&body).await?;
                w.flush().await?;
            }

            Ok(())
        })
        .await?;

        let json_path = json.display().to_string();

        let parquet = idempotent(&self.parquet_path()?, move |parquet_path| {
            let parquet = parquet_path.display().to_string();
            info!(
                "Converting GithubArchive JSON to Parquet with DuckDB @ {}",
                parquet_path.display()
            );
            let result = Command::new("duckdb")
                .arg("-c")
                .arg(format!(
                    "
                    CREATE TABLE events AS select * from read_ndjson_auto('{json_path}', ignore_errors = true);
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
