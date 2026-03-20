// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Data from the GitHub Archive.
//!
//! This dataset applies a bunch of events this way

use std::path::PathBuf;
use std::process::Command;

use tokio::io::AsyncWriteExt;
use tracing::info;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::BenchmarkDescriptor;
use crate::QuerySource;
use crate::idempotent;
use crate::idempotent_async;
use crate::resolve_data_url;

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
    descriptor: BenchmarkDescriptor,
}

impl GithubArchiveBenchmark {
    pub fn with_remote_data_dir(use_remote_data_dir: Option<String>) -> anyhow::Result<Self> {
        let data_url = resolve_data_url("gharchive", use_remote_data_dir.as_deref())?;

        Ok(Self {
            descriptor: BenchmarkDescriptor::new(
                "gharchive",
                data_url,
                BenchmarkDataset::GhArchive,
            )
            .with_queries(QuerySource::Inline(QUERIES.to_vec()))
            .with_table("events", None)
            .with_expected_row_counts(vec![1, 2, 100, 10, 82468]),
        })
    }

    fn json_path(&self) -> anyhow::Result<PathBuf> {
        self.descriptor
            .data_url
            .join("json/")?
            .join("events.json.gz")?
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Failed to convert data URL to filesystem path - ensure data_url uses 'file://' scheme"))
    }

    fn parquet_path(&self) -> anyhow::Result<PathBuf> {
        self.descriptor
            .data_url
            .join("parquet/")?
            .join("events.parquet")?
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Failed to convert data URL to filesystem path - ensure data_url uses 'file://' scheme"))
    }
}

#[async_trait::async_trait]
impl Benchmark for GithubArchiveBenchmark {
    fn descriptor(&self) -> &BenchmarkDescriptor {
        &self.descriptor
    }

    async fn generate_base_data(&self) -> anyhow::Result<()> {
        if self.descriptor.data_url.scheme() != "file" {
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
}
