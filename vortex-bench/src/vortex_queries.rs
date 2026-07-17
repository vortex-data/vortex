// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmark that runs queries which aren't part of any existing benchmark
//! suite but which performance we want to track.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use glob::Pattern;
use tracing::debug;
use url::Url;
use vortex::error::VortexExpect;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::Format;
use crate::TableSpec;
use crate::bench_dir;
use crate::resolve_data_url;

// Path to script that creates Parquet data
const INIT_SQL: &str = "init.sql";

pub struct VortexBenchmark {
    data_url: Url,
    queries_dir: PathBuf,
    query: Option<PathBuf>,
}

impl VortexBenchmark {
    pub fn new() -> Result<Self> {
        let queries_dir = bench_dir().join("sql").join("vortex");
        let data_url = resolve_data_url(None, "vortex")?;
        Ok(Self {
            data_url,
            queries_dir,
            query: None,
        })
    }

    // Same as new(), but run only for "query" SQL file
    pub fn with_query(mut self, query: &str) -> Result<Self> {
        let as_path = PathBuf::from(query);
        let path = if as_path.is_file() {
            as_path
        } else if query.ends_with(".sql") {
            self.queries_dir.join(query)
        } else {
            self.queries_dir.join(format!("{query}.sql"))
        };
        if !path.is_file() {
            bail!("{query} file not found in {}", self.queries_dir.display());
        }
        self.query = Some(path);
        Ok(self)
    }

    fn query_files(&self) -> Result<Vec<PathBuf>> {
        if let Some(query) = &self.query {
            return Ok(vec![query.clone()]);
        }

        let entries = fs::read_dir(&self.queries_dir)
            .with_context(|| format!("cannot list queries in {}", self.queries_dir.display()))?
            .collect::<std::io::Result<Vec<_>>>()?;

        let mut files: Vec<PathBuf> = entries
            .into_iter()
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension().is_some_and(|ext| ext == "sql")
                    && path.file_name().is_some_and(|name| name != INIT_SQL)
            })
            .collect();
        files.sort();

        if files.is_empty() {
            bail!("no query files found in {}", self.queries_dir.display());
        }
        Ok(files)
    }
}

#[async_trait::async_trait]
impl Benchmark for VortexBenchmark {
    fn doc_path(&self) -> Option<&'static str> {
        Some("vortex-bench/sql/vortex/README.md")
    }

    fn queries(&self) -> Result<Vec<(usize, String)>> {
        self.query_files()?
            .iter()
            .map(|path| {
                let idx = path
                    .file_name()
                    .vortex_expect("no file name")
                    .to_str()
                    .vortex_expect("not utf-8")
                    .split_once("_")
                    .vortex_expect("query without a number")
                    .0
                    .parse::<usize>()?;
                let query = fs::read_to_string(path)
                    .with_context(|| format!("cannot read query {}", path.display()))?;
                debug!(idx, file = %path.display(), "Loaded vortex query");
                Ok((idx, query))
            })
            .collect()
    }

    async fn generate_base_data(&self) -> Result<()> {
        let data_dir = self
            .data_url
            .to_file_path()
            .map_err(|_| anyhow!("Invalid file URL: {}", self.data_url.as_str()))?;
        let parquet_dir = data_dir.join(Format::Parquet.name());
        fs::create_dir_all(&parquet_dir)?;
        let parquet_file = parquet_dir.join("test.parquet");

        if parquet_file.exists() {
            debug!("Parquet data present in {}", parquet_dir.display());
            return Ok(());
        }

        let init_path = self.queries_dir.join(INIT_SQL);
        let script = fs::read_to_string(&init_path)
            .with_context(|| format!("cannot read {}", init_path.display()))?;

        let output = Command::new("duckdb")
            .current_dir(&parquet_dir)
            .arg("-c")
            .arg(&script)
            .output()
            .context("cannot run duckdb")?;
        if !output.status.success() {
            bail!(
                "duckdb {INIT_SQL} failed: stdout={:?} stderr={:?}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
        }

        if !parquet_file.exists() {
            bail!("{INIT_SQL} did not create Parquet files");
        }

        debug!("Parquet data generated in {}", parquet_dir.display());
        Ok(())
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::VortexQueries
    }

    fn dataset_name(&self) -> &str {
        "vortex"
    }

    fn dataset_display(&self) -> String {
        "vortex".to_owned()
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }

    fn table_specs(&self) -> Vec<TableSpec> {
        vec![TableSpec::new("test", None)]
    }

    fn pattern(&self, table_name: &str, format: Format) -> Option<Pattern> {
        Some(
            Pattern::new(&format!("{table_name}.{}", format.ext()))
                .expect("table name is a valid identifier"),
        )
    }
}
