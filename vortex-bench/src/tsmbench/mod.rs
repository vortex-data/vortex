// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [TSM-Bench](https://github.com/eXascaleInfolab/TSM-Bench), a time-series database benchmark
//! from VLDB 2023, run over its `d1` dataset.
//!
//! `d1` holds hydrometric sensor readings augmented from a real seed dataset: 10 stations with
//! 100 `DOUBLE` sensor columns each, sampled every 10 seconds, ordered by station then time.

use std::fs;
use std::fs::File;
use std::io;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::process::Stdio;

use anyhow::Context;
use itertools::Itertools;
use tracing::info;
use url::Url;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::Format;
use crate::TableSpec;
use crate::datasets::data_downloads::download_many;
use crate::utils::file::resolve_data_url;
use crate::utils::file::temp_download_filepath;
use crate::workspace_root;

/// Benchmark and local data directory name.
pub const TSM_BENCH_NAME: &str = "tsm-bench";

/// TSM-Bench commit the dataset is downloaded from, pinned so the data cannot drift.
const TSM_BENCH_COMMIT: &str = "58f7f6a5319aa746bf4d45586d22d19156e9f00e";

/// The `d1` archive is split into parts with suffixes `aa` through `bk`.
const D1_SPLIT_COUNT: u8 = 37;

const N_SENSORS: usize = 100;

/// TSM-Bench over the `d1` dataset.
pub struct TsmBenchBenchmark {
    data_url: Url,
}

impl TsmBenchBenchmark {
    /// Create the benchmark, optionally using a remote data directory.
    pub fn new(use_remote_data_dir: Option<String>) -> anyhow::Result<Self> {
        Ok(Self {
            data_url: resolve_data_url(use_remote_data_dir.as_deref(), TSM_BENCH_NAME)?,
        })
    }
}

#[async_trait::async_trait]
impl Benchmark for TsmBenchBenchmark {
    fn doc_path(&self) -> &'static str {
        "vortex-bench/sql/tsm-bench.md"
    }

    fn queries(&self) -> anyhow::Result<Vec<(usize, String)>> {
        let queries_file = workspace_root()
            .join("vortex-bench")
            .join("sql")
            .join("tsm-bench.sql");
        Ok(split_queries(&fs::read_to_string(queries_file)?))
    }

    async fn generate_base_data(&self) -> anyhow::Result<()> {
        if self.data_url.scheme() != "file" {
            return Ok(());
        }

        let base_path = self
            .data_url
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Invalid file URL: {}", self.data_url))?;
        let parquet_dir = base_path.join(Format::Parquet.name());
        if parquet_dir.exists() {
            info!(
                "TSM-Bench parquet already exists at {}",
                parquet_dir.display()
            );
            return Ok(());
        }

        let temp_root = temp_download_filepath();
        let result = generate_d1_parquet(&base_path, &temp_root, &parquet_dir).await;
        drop(fs::remove_dir_all(&temp_root));
        result
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::TsmBench
    }

    fn dataset_name(&self) -> &str {
        TSM_BENCH_NAME
    }

    fn dataset_display(&self) -> String {
        TSM_BENCH_NAME.to_string()
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }

    fn table_specs(&self) -> Vec<TableSpec> {
        vec![TableSpec::new("d1", None)]
    }
}

/// Split `;`-separated SQL into numbered queries. A `;` must not appear in a comment.
fn split_queries(sql: &str) -> Vec<(usize, String)> {
    sql.split_terminator(';')
        .map(str::trim)
        .filter(|stmt| !stmt.is_empty())
        .map(str::to_string)
        .enumerate()
        .collect()
}

async fn generate_d1_parquet(
    base_path: &Path,
    temp_root: &Path,
    parquet_dir: &Path,
) -> anyhow::Result<()> {
    let splits_dir = base_path.join("d1_splits");
    let split_paths = download_many((0..D1_SPLIT_COUNT).map(|idx| {
        let name = split_file_name(idx);
        let url = format!(
            "https://raw.githubusercontent.com/eXascaleInfolab/TSM-Bench/{TSM_BENCH_COMMIT}/datasets/d1_splits/{name}"
        );
        (splits_dir.join(name), url)
    }))
    .await?;
    let split_paths = split_paths.into_iter().sorted().collect::<Vec<_>>();

    fs::create_dir_all(temp_root)
        .with_context(|| format!("Failed to create temp dir {}", temp_root.display()))?;
    let csv_path = temp_root.join("d1.csv");
    extract_d1_csv(&split_paths, &csv_path)?;

    let temp_parquet_dir = temp_root.join(Format::Parquet.name());
    fs::create_dir_all(&temp_parquet_dir)?;
    convert_csv_to_parquet(&csv_path, &temp_parquet_dir.join("d1.parquet"))?;
    fs::remove_file(&csv_path)?;

    if let Some(parent) = parquet_dir.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::rename(&temp_parquet_dir, parquet_dir).with_context(|| {
        format!(
            "Failed to move TSM-Bench parquet from {} to {}",
            temp_parquet_dir.display(),
            parquet_dir.display()
        )
    })?;
    fs::remove_dir_all(&splits_dir)?;
    Ok(())
}

fn split_file_name(idx: u8) -> String {
    let first = char::from(b'a' + idx / 26);
    let second = char::from(b'a' + idx % 26);
    format!("datasets_splits.{first}{second}")
}

/// Stream the concatenated archive parts through `tar` to write the single `d1.csv` member.
fn extract_d1_csv(split_paths: &[PathBuf], csv_path: &Path) -> anyhow::Result<()> {
    info!("Extracting TSM-Bench d1.csv to {}", csv_path.display());
    let mut child = Command::new("tar")
        .arg("-xzO")
        .stdin(Stdio::piped())
        .stdout(File::create(csv_path)?)
        .stderr(Stdio::piped())
        .spawn()
        .context("Failed to run tar while extracting TSM-Bench data")?;

    let mut stdin = child
        .stdin
        .take()
        .context("Failed to open tar stdin while extracting TSM-Bench data")?;
    for path in split_paths {
        io::copy(&mut File::open(path)?, &mut stdin)
            .with_context(|| format!("Failed to stream {} into tar", path.display()))?;
    }
    drop(stdin);

    let output = child.wait_with_output()?;
    anyhow::ensure!(
        output.status.success(),
        "tar failed extracting TSM-Bench data: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

fn convert_csv_to_parquet(csv_path: &Path, parquet_path: &Path) -> anyhow::Result<()> {
    info!("Converting TSM-Bench d1.csv to {}", parquet_path.display());
    let output = Command::new("duckdb")
        .arg("-c")
        .arg(csv_to_parquet_sql(csv_path, parquet_path))
        .output()
        .context("Failed to run DuckDB CLI while converting TSM-Bench data")?;
    anyhow::ensure!(
        output.status.success(),
        "DuckDB failed converting TSM-Bench data: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

fn csv_to_parquet_sql(csv_path: &Path, parquet_path: &Path) -> String {
    let columns = [
        "'time': 'TIMESTAMP'".to_string(),
        "'id_station': 'VARCHAR'".to_string(),
    ]
    .into_iter()
    .chain((0..N_SENSORS).map(|idx| format!("'s{idx}': 'DOUBLE'")))
    .join(", ");
    format!(
        "COPY (SELECT * FROM read_csv({csv}, header = true, columns = {{{columns}}})) \
         TO {parquet} (FORMAT parquet, COMPRESSION zstd, COMPRESSION_LEVEL 3);",
        csv = sql_string_literal(&csv_path.display().to_string()),
        parquet = sql_string_literal(&parquet_path.display().to_string()),
    )
}

fn sql_string_literal(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_file_names_cover_aa_through_bk() {
        assert_eq!(split_file_name(0), "datasets_splits.aa");
        assert_eq!(split_file_name(25), "datasets_splits.az");
        assert_eq!(split_file_name(26), "datasets_splits.ba");
        assert_eq!(split_file_name(D1_SPLIT_COUNT - 1), "datasets_splits.bk");
    }

    #[test]
    fn conversion_sql_types_every_column_and_uses_zstd_level_3() {
        let sql = csv_to_parquet_sql(Path::new("/tmp/d1.csv"), Path::new("/tmp/d1.parquet"));

        assert!(sql.contains("'time': 'TIMESTAMP'"));
        assert!(sql.contains("'s0': 'DOUBLE'"));
        assert!(sql.contains("'s99': 'DOUBLE'"));
        assert!(sql.contains("COMPRESSION zstd, COMPRESSION_LEVEL 3"));
    }

    #[test]
    fn queries_file_holds_the_seven_tsm_bench_queries() -> anyhow::Result<()> {
        let sql = fs::read_to_string(
            workspace_root()
                .join("vortex-bench")
                .join("sql")
                .join("tsm-bench.sql"),
        )?;
        let queries = split_queries(&sql);

        assert_eq!(queries.len(), 7);
        assert!(queries.iter().all(|(_, query)| query.contains("FROM d1")));
        Ok(())
    }
}
