// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;

use tokio::io::AsyncWriteExt;
use tracing::info;
use url::Url;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::IdempotentPath;
use crate::TableSpec;
use crate::idempotent_async;

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
        let data_url = Self::create_data_url(&use_remote_data_dir)?;
        Ok(Self { data_url })
    }

    fn create_data_url(remote_data_dir: &Option<String>) -> anyhow::Result<Url> {
        match remote_data_dir {
            None => {
                let data_dir = "fineweb".to_data_path();
                Url::from_directory_path(&data_dir).map_err(|_| {
                    anyhow::anyhow!("Failed to create URL from directory path: {:?}", &data_dir)
                })
            }
            Some(remote_data_dir) => {
                if !remote_data_dir.ends_with("/") {
                    tracing::warn!(
                        "Supply a --use-remote-data-dir argument which ends in a slash e.g. s3://vortex-bench-dev-eu/develop/12345/fineweb/"
                    );
                }
                tracing::info!(
                    concat!(
                        "Assuming data already exists at this remote (e.g. S3, GCS) URL: {}.\n",
                        "If it does not, you should kill this command, locally generate the files (by running without\n",
                        "--use-remote-data-dir) and upload data/fineweb/ to some remote location.",
                    ),
                    remote_data_dir,
                );
                Ok(Url::parse(remote_data_dir)?)
            }
        }
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

        let parquet = idempotent_async(&self.parquet_path()?, |parquet_path| async move {
            info!("Downloading FineWeb Parquet source from HuggingFace");

            let response = reqwest::get(SAMPLE_URL)
                .await?
                .error_for_status()
                .map_err(|err| {
                    anyhow::anyhow!("error fetching fineweb sample from HuggingFace: {err}")
                })?;

            let bytes = response.bytes().await?;
            let mut w = tokio::fs::File::create(parquet_path).await?;

            w.write_all(&bytes).await?;

            w.flush().await?;

            Ok(())
        })
        .await?;

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
