// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;

use tokio::io::AsyncWriteExt;
use tracing::info;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::BenchmarkDescriptor;
use crate::QuerySource;
use crate::idempotent_async;
use crate::resolve_data_url;

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
    descriptor: BenchmarkDescriptor,
}

impl FinewebBenchmark {
    pub fn with_remote_data_dir(use_remote_data_dir: Option<String>) -> anyhow::Result<Self> {
        let data_url = resolve_data_url("fineweb", use_remote_data_dir.as_deref())?;

        Ok(Self {
            descriptor: BenchmarkDescriptor::new("fineweb", data_url, BenchmarkDataset::Fineweb)
                .with_queries(QuerySource::Inline(QUERIES.to_vec()))
                .with_table("fineweb", None),
        })
    }

    fn parquet_path(&self) -> anyhow::Result<PathBuf> {
        self.descriptor
            .data_url
            .join("parquet/")?
            .join("sample.parquet")?
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Failed to convert data URL to filesystem path - ensure data_url uses 'file://' scheme"))
    }
}

#[async_trait::async_trait]
impl Benchmark for FinewebBenchmark {
    fn descriptor(&self) -> &BenchmarkDescriptor {
        &self.descriptor
    }

    async fn generate_base_data(&self) -> anyhow::Result<()> {
        if self.descriptor.data_url.scheme() != "file" {
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
}
