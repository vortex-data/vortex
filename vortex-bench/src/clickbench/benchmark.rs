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
    /// How many of the partitioned dataset's files to use, from 1 to [`CLICKBENCH_FILES`].
    pub files: u32,
    pub queries_file: Option<String>,
    pub data_url: Url,
}

impl ClickBenchBenchmark {
    pub fn new(
        flavor: Flavor,
        files: u32,
        queries_file: Option<String>,
        use_remote_data_dir: Option<String>,
    ) -> Result<Self> {
        anyhow::ensure!(
            (1..=CLICKBENCH_FILES).contains(&files),
            "ClickBench files must be between 1 and {CLICKBENCH_FILES}, got {files}"
        );
        anyhow::ensure!(
            flavor == Flavor::Partitioned || files == CLICKBENCH_FILES,
            "ClickBench files applies only to the partitioned flavor"
        );
        let url = resolve_data_url(
            use_remote_data_dir.as_deref(),
            &clickbench_data_dir(flavor, files),
        )?;
        Ok(Self {
            flavor,
            files,
            queries_file,
            data_url: url,
        })
    }
}

/// ClickBench sorted by event time, with shard filenames shuffled to exercise sort pushdown.
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
        None => Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("sql")
            .join("clickbench_queries.sql"),
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
    fn doc_path(&self) -> &'static str {
        "vortex-bench/sql/clickbench.md"
    }

    fn queries(&self) -> Result<Vec<(usize, String)>> {
        read_clickbench_queries(self.queries_file.as_deref())
    }

    async fn generate_base_data(&self) -> Result<()> {
        if self.data_url.scheme() != "file" {
            return Ok(());
        }

        let basepath = clickbench_data_dir(self.flavor, self.files).to_data_path();
        self.flavor.download(basepath, self.files).await?;

        Ok(())
    }

    fn expected_row_counts(&self) -> Option<Vec<usize>> {
        // The known row counts hold only for the complete dataset.
        (self.files == CLICKBENCH_FILES).then(clickbench_expected_row_counts)
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::ClickBench {
            flavor: self.flavor,
            files: (self.files != CLICKBENCH_FILES).then_some(self.files),
        }
    }

    fn dataset_name(&self) -> &str {
        "clickbench"
    }

    fn dataset_display(&self) -> String {
        clickbench_data_dir(self.flavor, self.files)
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
    fn doc_path(&self) -> &'static str {
        "vortex-bench/sql/clickbench.md#sorted-variant"
    }

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

fn clickbench_data_dir(flavor: Flavor, files: u32) -> String {
    if files == CLICKBENCH_FILES {
        format!("clickbench_{flavor}")
    } else {
        format!("clickbench_{flavor}_{files}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subsets_skip_the_full_dataset_row_counts() -> Result<()> {
        let subset = ClickBenchBenchmark::new(Flavor::Partitioned, 10, None, None)?;
        assert!(subset.expected_row_counts().is_none());
        assert!(
            subset
                .data_url
                .path()
                .ends_with("/clickbench_partitioned_10/")
        );

        let full = ClickBenchBenchmark::new(Flavor::Partitioned, CLICKBENCH_FILES, None, None)?;
        assert!(full.expected_row_counts().is_some());
        assert!(full.data_url.path().ends_with("/clickbench_partitioned/"));

        assert!(ClickBenchBenchmark::new(Flavor::Single, 10, None, None).is_err());
        Ok(())
    }
}
