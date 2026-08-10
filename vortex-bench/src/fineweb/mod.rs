// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;

use url::Url;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::Format;
use crate::SetupCtx;
use crate::TableSpec;
use crate::utils::file::resolve_data_url;
use crate::workspace_root;

/// URL to the sample file
const SAMPLE_URL: &str = "https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/v1.4.0/sample/10BT/001_00000.parquet";

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

impl FinewebBenchmark {}

#[async_trait::async_trait]
impl Benchmark for FinewebBenchmark {
    fn doc_path(&self) -> &'static str {
        "vortex-bench/sql/fineweb.md"
    }

    /// Some basic string-focused queries, numbered from Q0 in `sql/fineweb.sql` file order.
    fn queries(&self) -> anyhow::Result<Vec<(usize, String)>> {
        // `;`-separated; a `;` must not appear in a comment, or it would split a statement in two.
        let queries_file = workspace_root()
            .join("vortex-bench")
            .join("sql")
            .join("fineweb")
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
        let path = ctx.staging().join("sample.parquet");
        ctx.download([(SAMPLE_URL, path.clone())]).await?;
        ctx.emit("fineweb", path);
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
