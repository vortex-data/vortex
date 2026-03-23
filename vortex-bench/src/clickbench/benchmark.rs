// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use reqwest::Client;
use url::Url;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::IdempotentPath;
use crate::TableSpec;
use crate::clickbench::*;

/// ClickBench benchmark implementation
pub struct ClickBenchBenchmark {
    pub flavor: Flavor,
    pub queries_file: Option<String>,
    pub data_url: Url,
}

impl ClickBenchBenchmark {
    pub fn new(
        flavor: Flavor,
        queries_file: Option<String>,
        use_remote_data_dir: Option<String>,
    ) -> Result<Self> {
        let url = Self::create_data_url(&use_remote_data_dir, flavor)?;
        Ok(Self {
            flavor,
            queries_file,
            data_url: url,
        })
    }

    /// Returns the path to the queries file.
    fn queries_file_path(&self) -> PathBuf {
        if let Some(file) = &self.queries_file {
            return file.into();
        }
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        manifest_dir.join("clickbench_queries.sql")
    }

    fn create_data_url(remote_data_dir: &Option<String>, flavor: Flavor) -> Result<Url> {
        match remote_data_dir {
            None => {
                let basepath = format!("clickbench_{flavor}").to_data_path();
                Url::from_directory_path(basepath)
                    .map_err(|_| anyhow::anyhow!("Failed to convert ClickBench data path to URL"))
            }
            Some(remote_data_dir) => {
                if !remote_data_dir.ends_with("/") {
                    tracing::warn!(
                        "Supply a --use-remote-data-dir argument which ends in a slash e.g. s3://vortex-bench-dev-eu/parquet/"
                    );
                }
                tracing::info!(
                    concat!(
                        "Assuming data already exists at this remote (e.g. S3, GCS) URL: {}.\\n",
                        "If it does not, you should kill this command, locally generate the files (by running without\\n",
                        "--use-remote-data-dir) and upload data/clickbench/ to some remote location.",
                    ),
                    remote_data_dir,
                );
                Ok(Url::parse(remote_data_dir)?)
            }
        }
    }
}

#[async_trait::async_trait]
impl Benchmark for ClickBenchBenchmark {
    fn queries(&self) -> Result<Vec<(usize, String)>> {
        let queries_filepath = self.queries_file_path();

        Ok(fs::read_to_string(queries_filepath)?
            .split(';')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .enumerate()
            .collect())
    }

    async fn generate_base_data(&self) -> Result<()> {
        if self.data_url.scheme() != "file" {
            return Ok(());
        }

        let basepath = clickbench_flavor(self.flavor).to_data_path();
        self.flavor.download(Client::default(), basepath).await?;

        Ok(())
    }

    fn expected_row_counts(&self) -> Option<Vec<usize>> {
        Some(vec![
            1, 1, 1, 1, 1, 1, 1, 18, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 4, 1, 10, 10, 10,
            10, 10, 10, 25, 25, 1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
        ])
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::ClickBench {
            flavor: self.flavor,
        }
    }

    fn dataset_name(&self) -> &str {
        "clickbench"
    }

    fn dataset_display(&self) -> String {
        format!("clickbench_{}", self.flavor)
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }

    fn table_specs(&self) -> Vec<TableSpec> {
        vec![TableSpec::new("hits", Some(HITS_SCHEMA.clone()))]
    }
}

fn clickbench_flavor(flavor: Flavor) -> String {
    format!("clickbench_{flavor}")
}
