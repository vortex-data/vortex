// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::env;
use std::fs;
use std::path::Path;

use anyhow::Result;
use url::Url;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::IdempotentPath;
use crate::TableSpec;
use crate::clickbench::*;
use crate::utils::file::resolve_data_url;

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
        let url = Self::create_data_url(use_remote_data_dir.as_deref(), flavor)?;
        Ok(Self {
            flavor,
            queries_file,
            data_url: url,
        })
    }

    fn create_data_url(remote_data_dir: Option<&str>, flavor: Flavor) -> Result<Url> {
        resolve_data_url(remote_data_dir, &format!("clickbench_{flavor}"))
    }
}

/// ClickBench sorted by event date and event time.
pub struct ClickBenchSortedBenchmark {
    pub queries_file: Option<String>,
    pub data_url: Url,
}

impl ClickBenchSortedBenchmark {
    /// Create the sorted ClickBench benchmark, optionally using a remote data directory.
    pub fn new(use_remote_data_dir: Option<String>) -> Result<Self> {
        Ok(Self {
            queries_file: None,
            data_url: resolve_data_url(use_remote_data_dir.as_deref(), CLICKBENCH_SORTED_NAME)?,
        })
    }
}

fn read_clickbench_queries(queries_file: Option<&str>) -> Result<Vec<(usize, String)>> {
    let queries_filepath = match queries_file {
        Some(file) => file.into(),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("clickbench_queries.sql"),
    };

    Ok(fs::read_to_string(queries_filepath)?
        .split(';')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .enumerate()
        .collect())
}

#[async_trait::async_trait]
impl Benchmark for ClickBenchBenchmark {
    fn queries(&self) -> Result<Vec<(usize, String)>> {
        read_clickbench_queries(self.queries_file.as_deref())
    }

    async fn generate_base_data(&self) -> Result<()> {
        if self.data_url.scheme() != "file" {
            return Ok(());
        }

        let basepath = clickbench_flavor(self.flavor).to_data_path();
        self.flavor.download(basepath).await?;

        Ok(())
    }

    fn expected_row_counts(&self) -> Option<Vec<usize>> {
        Some(clickbench_expected_row_counts())
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

#[async_trait::async_trait]
impl Benchmark for ClickBenchSortedBenchmark {
    fn queries(&self) -> Result<Vec<(usize, String)>> {
        Ok(read_clickbench_queries(self.queries_file.as_deref())?
            .into_iter()
            .filter(|(idx, _)| CLICKBENCH_SORTED_QUERY_IDS.contains(idx))
            .collect())
    }

    async fn generate_base_data(&self) -> Result<()> {
        if self.data_url.scheme() != "file" {
            return Ok(());
        }

        generate_sorted_clickbench(CLICKBENCH_SORTED_NAME.to_data_path()).await
    }

    fn expected_row_counts(&self) -> Option<Vec<usize>> {
        Some(clickbench_expected_row_counts())
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::ClickBenchSorted
    }

    fn dataset_name(&self) -> &str {
        CLICKBENCH_SORTED_NAME
    }

    fn dataset_display(&self) -> String {
        CLICKBENCH_SORTED_NAME.to_string()
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
