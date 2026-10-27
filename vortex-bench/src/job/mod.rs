// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Join Order Benchmark (JOB): the 113 queries from Leis et al. (VLDB 2015) over IMDB.
//!
//! Mirrors DuckDB's `benchmark/imdb` suite. Queries live in `vortex-bench/sql/job/` and are
//! numbered `1..=113` in file-name order (`01a` = 1, ..., `33c` = 113). The 21 tables are
//! downloaded from DuckDB's published Parquet files and re-encoded with zstd level 3.

use std::fs;
use std::fs::File;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use glob::Pattern;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::basic::ZstdLevel;
use parquet::file::properties::WriterProperties;
use tracing::info;
use url::Url;
use vortex::error::VortexExpect;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::Format;
use crate::TableSpec;
use crate::datasets::data_downloads::download_many;
use crate::utils::file::idempotent;
use crate::utils::file::resolve_data_url;

/// Release that hosts DuckDB's JOB Parquet files, as referenced by DuckDB's
/// `benchmark/imdb/init/load.sql`. Each table lives at `{UPSTREAM_BASE_URL}/job_{table}.parquet`.
const UPSTREAM_BASE_URL: &str = "https://github.com/duckdb/duckdb-data/releases/download/v1.0";

/// The 21 IMDB tables referenced by the JOB queries.
pub(crate) const TABLES: &[&str] = &[
    "aka_name",
    "aka_title",
    "cast_info",
    "char_name",
    "comp_cast_type",
    "company_name",
    "company_type",
    "complete_cast",
    "info_type",
    "keyword",
    "kind_type",
    "link_type",
    "movie_companies",
    "movie_info",
    "movie_info_idx",
    "movie_keyword",
    "movie_link",
    "name",
    "person_info",
    "role_type",
    "title",
];

/// All JOB queries as `(index, sql)` pairs, numbered in file-name order.
pub fn job_queries() -> impl Iterator<Item = (usize, String)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("sql")
        .join("job");
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .vortex_expect("cannot list JOB queries")
        .map(|e| e.vortex_expect("cannot read JOB query dir entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "sql"))
        .collect();
    paths.sort();
    paths.into_iter().enumerate().map(|(i, p)| {
        let sql = fs::read_to_string(p).vortex_expect("cannot load JOB query from file");
        (i + 1, sql)
    })
}
/// Benchmark over the [Join Order Benchmark][paper] on the IMDB dataset.
///
/// [paper]: https://www.vldb.org/pvldb/vol9/p204-leis.pdf
pub struct JobBenchmark {
    data_url: Url,
}

impl JobBenchmark {
    pub fn with_remote_data_dir(use_remote_data_dir: Option<String>) -> anyhow::Result<Self> {
        let data_url = resolve_data_url(use_remote_data_dir.as_deref(), "job")?;
        Ok(Self { data_url })
    }

    fn base_dir(&self) -> anyhow::Result<PathBuf> {
        self.data_url
            .to_file_path()
            .map_err(|_| anyhow::anyhow!(
                "Failed to convert data URL to filesystem path - ensure data_url uses 'file://' scheme"
            ))
    }
}

#[async_trait::async_trait]
impl Benchmark for JobBenchmark {
    fn doc_path(&self) -> &'static str {
        "vortex-bench/sql/job/README.md"
    }

    fn queries(&self) -> anyhow::Result<Vec<(usize, String)>> {
        Ok(job_queries().collect())
    }

    async fn generate_base_data(&self) -> anyhow::Result<()> {
        if self.data_url.scheme() != "file" {
            return Ok(());
        }

        let base_dir = self.base_dir()?;
        let parquet_dir = base_dir.join(Format::Parquet.name());
        let missing: Vec<&str> = TABLES
            .iter()
            .copied()
            .filter(|t| !parquet_dir.join(format!("{t}.parquet")).exists())
            .collect();
        if missing.is_empty() {
            info!(
                "job: {} Parquet tables already present in {}",
                TABLES.len(),
                parquet_dir.display(),
            );
            return Ok(());
        }

        let upstream_dir = base_dir.join("upstream");
        fs::create_dir_all(&upstream_dir)?;
        fs::create_dir_all(&parquet_dir)?;

        download_many(missing.iter().map(|t| {
            (
                upstream_dir.join(format!("job_{t}.parquet")),
                format!("{UPSTREAM_BASE_URL}/job_{t}.parquet"),
            )
        }))
        .await?;

        for table in missing {
            let src = upstream_dir.join(format!("job_{table}.parquet"));
            let dst = parquet_dir.join(format!("{table}.parquet"));
            tokio::task::spawn_blocking(move || reencode_parquet(&src, &dst)).await??;
        }

        info!(
            "job base data generated in {} ({} Parquet tables)",
            parquet_dir.display(),
            TABLES.len(),
        );
        Ok(())
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::Job
    }

    fn dataset_name(&self) -> &str {
        "job"
    }

    fn dataset_display(&self) -> String {
        "job".to_owned()
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }

    fn table_specs(&self) -> Vec<TableSpec> {
        TABLES
            .iter()
            .map(|name| TableSpec::new(name, None))
            .collect()
    }

    #[expect(clippy::expect_used)]
    fn pattern(&self, table_name: &str, format: Format) -> Option<Pattern> {
        Some(
            format!("{}.{}", table_name, format.ext())
                .parse()
                .expect("valid glob pattern"),
        )
    }
}

/// Rewrite the upstream Parquet file at `src` to `dst` with zstd level 3 compression.
fn reencode_parquet(src: &Path, dst: &Path) -> anyhow::Result<PathBuf> {
    idempotent(dst, |tmp| {
        let builder = ParquetRecordBatchReaderBuilder::try_new(
            File::open(src).with_context(|| format!("opening {}", src.display()))?,
        )?;
        let schema = Arc::clone(builder.schema());
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
            .build();
        let mut writer = ArrowWriter::try_new(File::create(tmp)?, schema, Some(props))?;
        for batch in builder.build()? {
            writer.write(&batch?)?;
        }
        writer.close()?;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_query_file_loads() {
        let queries: Vec<_> = job_queries().collect();
        assert_eq!(queries.len(), 113);
        assert_eq!(queries.first().map(|(i, _)| *i), Some(1));
        assert_eq!(queries.last().map(|(i, _)| *i), Some(113));
        assert!(queries.iter().all(|(_, sql)| sql.contains("SELECT")));
    }
}
