// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;

use tracing::info;
use url::Url;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::TableSpec;
use crate::datasets::data_downloads::download_data;
use crate::utils::file::resolve_data_url;

/// URL to the sample file
const SAMPLE_URL: &str = "https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/v1.4.0/sample/10BT/001_00000.parquet";

/// Some basic string-focused queries.
const QUERIES: &[&str] = &[
    // simple summary
    "SELECT count(DISTINCT dump) FROM fineweb",
    // selective string equality filter
    "SELECT * FROM fineweb WHERE dump = 'CC-MAIN-2016-30'",
    // LIKE with prefix filter
    "SELECT * FROM fineweb WHERE date LIKE '2020-10-%'",
    // LIKE with simple containment filter
    "SELECT * FROM fineweb WHERE url LIKE '%google%' AND text LIKE '%Google%'",
    // LIKE with larger containment filter
    "SELECT * FROM fineweb WHERE url LIKE '%.google.%' OR text LIKE '% Google %'",
    "SELECT * FROM fineweb WHERE text LIKE '% vortex %'",
    // More LIKE filters
    "SELECT * FROM fineweb WHERE url LIKE '%espn%' AND language = 'en' AND language_score > 0.92",
    "SELECT * FROM fineweb WHERE url LIKE '%espn%' OR url LIKE '%www.espn.go.com%' OR url LIKE '%espn.go.com%'",
    // no results, stats cannot prune but tokenized bloom filters could
    "SELECT * FROM fineweb WHERE file_path LIKE '%/CC-MAIN-2014-%'",
];

/// A benchmark using the HuggingFace FineWeb dataset.
///
/// This is a very string-heavy dataset, and exercises dictionary and FSST encoding heavily.
///
/// The queries for this benchmark are hand-crafted to showcase just how many of these we have here.
pub struct FinewebBenchmark {
    data_url: Url,
}

impl FinewebBenchmark {
    pub fn new(data_url: Url) -> Self {
        Self { data_url }
    }

    pub fn with_remote_data_dir(use_remote_data_dir: Option<String>) -> anyhow::Result<Self> {
        let data_url = Self::create_data_url(use_remote_data_dir.as_deref())?;
        Ok(Self { data_url })
    }

    fn create_data_url(remote_data_dir: Option<&str>) -> anyhow::Result<Url> {
        resolve_data_url(remote_data_dir, "fineweb")
    }
}

impl FinewebBenchmark {
    fn parquet_path(&self) -> anyhow::Result<PathBuf> {
        self.data_url
            .join("parquet/")?
            .join("sample.parquet")?
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Failed to convert data URL to filesystem path - ensure data_url uses 'file://' scheme"))
    }
}

#[async_trait::async_trait]
impl Benchmark for FinewebBenchmark {
    fn queries(&self) -> anyhow::Result<Vec<(usize, String)>> {
        Ok(QUERIES.iter().map(|s| s.to_string()).enumerate().collect())
    }

    async fn generate_base_data(&self) -> anyhow::Result<()> {
        if self.data_url.scheme() != "file" {
            return Ok(());
        }

        let parquet = download_data(self.parquet_path()?, SAMPLE_URL).await?;
        info!("fineweb base data generated in {}", parquet.display());

        Ok(())
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::Fineweb
    }

    fn dataset_name(&self) -> &str {
        "fineweb"
    }

    fn dataset_display(&self) -> String {
        "fineweb".to_owned()
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }

    fn table_specs(&self) -> Vec<TableSpec> {
        vec![TableSpec::new("fineweb", None)]
    }
}
