// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [JSONBench](https://github.com/ClickHouse/JSONBench): ClickHouse's analytical benchmark over
//! semi-structured JSON, run against Bluesky social network events.
//!
//! Every event lives in a single `data` column of the `bluesky` table. The baseline Parquet format
//! stores it as a JSON string; `parquet-variant` and the Vortex formats store it as a shredded
//! Variant. The queries in `sql/jsonbench.sql` address JSON paths with placeholders that
//! [`JsonBenchBenchmark::query_for`] expands into each engine's idiom for each format.

pub mod data;

use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::LazyLock;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use anyhow::Context;
use regex::Captures;
use regex::Regex;
use tokio::io::AsyncWriteExt;
use tracing::info;
use url::Url;
use vortex::utils::parallelism::get_available_parallelism;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::CompactionStrategy;
use crate::Engine;
use crate::Format;
use crate::TableSpec;
use crate::conversions::convert_parquet_file_to_vortex;
use crate::idempotent;
use crate::idempotent_async;
use crate::jsonbench::data::ShreddingSchema;
use crate::jsonbench::data::output_stem;
use crate::jsonbench::data::raw_json_name;
use crate::jsonbench::data::raw_json_url;
use crate::jsonbench::data::write_json_parquet;
use crate::jsonbench::data::write_variant_parquet;
use crate::utils::file::data_dir;
use crate::workspace_root;

/// The single table every query reads.
const TABLE: &str = "bluesky";

/// Matches a JSON path placeholder such as `{str:commit.collection}` or `{i64:time_us}`.
static PLACEHOLDER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\{(str|i64):([A-Za-z0-9_$.]+)\}").expect("placeholder regex is valid")
});

pub struct JsonBenchBenchmark {
    /// Number of one-million-row raw files the dataset is built from.
    n_files: usize,
    data_url: Url,
    /// Directory holding the downloaded raw files, shared by every scale.
    raw_dir: PathBuf,
}

impl JsonBenchBenchmark {
    /// A JSONBench dataset of `scale_factor` million events.
    pub fn new(scale_factor: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(scale_factor > 0, "jsonbench scale factor must be positive");
        let root = data_dir().join("jsonbench");
        let data_dir = root.join(format!("{scale_factor}m"));
        let data_url = Url::from_directory_path(&data_dir)
            .map_err(|_| anyhow::anyhow!("invalid data directory {}", data_dir.display()))?;
        Ok(Self {
            n_files: scale_factor,
            data_url,
            raw_dir: root.join("json"),
        })
    }

    fn base_path(&self) -> anyhow::Result<PathBuf> {
        self.data_url
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("jsonbench data URL must be a file URL"))
    }

    fn raw_path(&self, file_idx: usize) -> PathBuf {
        self.raw_dir.join(raw_json_name(file_idx))
    }

    fn file_indices(&self) -> std::ops::RangeInclusive<usize> {
        1..=self.n_files
    }

    fn shredding_schema(&self) -> anyhow::Result<ShreddingSchema> {
        let path = self.base_path()?.join("shredding.json");
        idempotent(&path, |tmp| {
            let schema = ShreddingSchema::infer(&self.raw_path(1))?;
            info!(
                "inferred {} shredded JSONBench paths: {}",
                schema.paths.len(),
                schema
                    .paths
                    .iter()
                    .map(|p| p.path.join("."))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            schema.save(tmp)
        })?;
        ShreddingSchema::load(&path)
    }

    /// Run `convert(raw, output)` for every raw file whose `{stem}.{ext}` output is missing in
    /// `format_dir`, in parallel.
    fn convert_files(
        &self,
        format_dir: &Path,
        ext: &str,
        convert: impl Fn(&Path, &Path) -> anyhow::Result<()> + Sync,
    ) -> anyhow::Result<()> {
        fs::create_dir_all(format_dir)?;
        let indices: Vec<usize> = self.file_indices().collect();
        let next = AtomicUsize::new(0);
        let workers = get_available_parallelism()
            .unwrap_or(1)
            .min(indices.len())
            .max(1);
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..workers)
                .map(|_| {
                    scope.spawn(|| -> anyhow::Result<()> {
                        while let Some(&file_idx) =
                            indices.get(next.fetch_add(1, Ordering::Relaxed))
                        {
                            let output =
                                format_dir.join(format!("{}.{ext}", output_stem(file_idx)));
                            idempotent(&output, |tmp| convert(&self.raw_path(file_idx), tmp))?;
                        }
                        Ok(())
                    })
                })
                .collect();
            handles
                .into_iter()
                .try_for_each(|handle| handle.join().expect("conversion worker panicked"))
        })
    }

    fn prepare_variant_parquet(&self) -> anyhow::Result<PathBuf> {
        let shredding = self.shredding_schema()?;
        let dir = self.base_path()?.join(Format::ParquetVariant.name());
        self.convert_files(&dir, "parquet", |raw, output| {
            write_variant_parquet(raw, output, &shredding)
        })?;
        Ok(dir)
    }

    async fn prepare_vortex(&self, format: Format) -> anyhow::Result<()> {
        let compaction = match format {
            Format::VortexCompact => CompactionStrategy::Compact,
            _ => CompactionStrategy::Default,
        };
        let variant_dir = self.prepare_variant_parquet()?;
        let vortex_dir = self.base_path()?.join(format.name());
        fs::create_dir_all(&vortex_dir)?;
        for file_idx in self.file_indices() {
            let stem = output_stem(file_idx);
            let parquet = variant_dir.join(format!("{stem}.parquet"));
            idempotent_async(vortex_dir.join(format!("{stem}.vortex")), |tmp| async move {
                convert_parquet_file_to_vortex(&parquet, &tmp, compaction).await
            })
            .await?;
        }
        Ok(())
    }

    /// Expand `{str:path}` and `{i64:path}` placeholders into `engine`'s idiom for reading that
    /// JSON path from `format`.
    pub fn expand_placeholders(engine: Engine, format: Format, query: &str) -> String {
        PLACEHOLDER
            .replace_all(query, |caps: &Captures| {
                let ty = &caps[1];
                let path = &caps[2];
                expand_path(engine, format, ty, path)
            })
            .into_owned()
    }
}

/// The SQL expression reading JSON `path` as `ty` (`str` or `i64`) on `engine` from `format`.
fn expand_path(engine: Engine, format: Format, ty: &str, path: &str) -> String {
    let string_storage = matches!(format, Format::Parquet | Format::OnDiskDuckDB);
    match engine {
        Engine::DataFusion if string_storage => {
            let keys = path
                .split('.')
                .map(|key| format!("'{key}'"))
                .collect::<Vec<_>>()
                .join(", ");
            match ty {
                "str" => format!("json_get_str(data, {keys})"),
                _ => format!("json_get_int(data, {keys})"),
            }
        }
        Engine::DataFusion | Engine::Vortex => {
            let arrow_type = if ty == "str" { "Utf8" } else { "Int64" };
            format!("variant_get(data, '{path}', '{arrow_type}')")
        }
        Engine::DuckDB if string_storage => match ty {
            "str" => format!("json_extract_string(data, '$.{path}')"),
            _ => format!("CAST(json_extract_string(data, '$.{path}') AS BIGINT)"),
        },
        Engine::DuckDB => {
            let sql_type = if ty == "str" { "VARCHAR" } else { "BIGINT" };
            let quoted = path
                .split('.')
                .map(|key| format!("\"{key}\""))
                .collect::<Vec<_>>()
                .join(".");
            format!("CAST(data.{quoted} AS {sql_type})")
        }
    }
}

#[async_trait::async_trait]
impl Benchmark for JsonBenchBenchmark {
    fn queries(&self) -> anyhow::Result<Vec<(usize, String)>> {
        // `;`-separated; a `;` must not appear in a comment, or it would split a statement in two.
        let queries_file = workspace_root()
            .join("vortex-bench")
            .join("sql")
            .join("jsonbench.sql");
        let contents = fs::read_to_string(&queries_file)
            .with_context(|| format!("reading {}", queries_file.display()))?;
        Ok(contents
            .split_terminator(';')
            .map(str::trim)
            .filter(|stmt| !stmt.is_empty())
            .map(str::to_string)
            .enumerate()
            .collect())
    }

    fn query_for(&self, engine: Engine, format: Format, query: &str) -> String {
        Self::expand_placeholders(engine, format, query)
    }

    async fn generate_base_data(&self) -> anyhow::Result<()> {
        fs::create_dir_all(&self.raw_dir)?;
        let client = reqwest::Client::new();
        for file_idx in self.file_indices() {
            let client = &client;
            idempotent_async(self.raw_path(file_idx), |tmp| async move {
                let url = raw_json_url(file_idx);
                info!("downloading {url}");
                let body = client
                    .get(&url)
                    .send()
                    .await?
                    .error_for_status()
                    .with_context(|| format!("fetching {url}"))?
                    .bytes()
                    .await?;
                let mut file = tokio::fs::File::create(&tmp).await?;
                file.write_all(&body).await?;
                file.flush().await?;
                anyhow::Ok(())
            })
            .await?;
        }

        let parquet_dir = self.base_path()?.join(Format::Parquet.name());
        tokio::task::block_in_place(|| {
            self.shredding_schema()?;
            self.convert_files(&parquet_dir, "parquet", write_json_parquet)
        })
    }

    async fn prepare_format(&self, format: Format, _base_path: &Path) -> anyhow::Result<()> {
        match format {
            Format::ParquetVariant => {
                tokio::task::block_in_place(|| self.prepare_variant_parquet())?;
            }
            Format::OnDiskVortex | Format::VortexCompact => {
                tokio::task::block_in_place(|| self.prepare_variant_parquet())?;
                self.prepare_vortex(format).await?;
            }
            _ => {}
        }
        Ok(())
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::JsonBench {
            n_rows: self.n_files * data::ROWS_PER_FILE,
        }
    }

    fn doc_path(&self) -> &'static str {
        "vortex-bench/sql/jsonbench.md"
    }

    fn dataset_name(&self) -> &str {
        "jsonbench"
    }

    fn dataset_display(&self) -> String {
        format!("jsonbench({}m)", self.n_files)
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }

    fn table_specs(&self) -> Vec<TableSpec> {
        vec![TableSpec::new(TABLE, None)]
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const QUERY: &str = "SELECT {str:commit.collection}, {i64:time_us} FROM bluesky";

    #[rstest]
    #[case(
        Engine::DataFusion,
        Format::Parquet,
        "SELECT json_get_str(data, 'commit', 'collection'), json_get_int(data, 'time_us') FROM bluesky"
    )]
    #[case(
        Engine::DataFusion,
        Format::OnDiskVortex,
        "SELECT variant_get(data, 'commit.collection', 'Utf8'), variant_get(data, 'time_us', 'Int64') FROM bluesky"
    )]
    #[case(
        Engine::DuckDB,
        Format::Parquet,
        "SELECT json_extract_string(data, '$.commit.collection'), CAST(json_extract_string(data, '$.time_us') AS BIGINT) FROM bluesky"
    )]
    #[case(
        Engine::DuckDB,
        Format::ParquetVariant,
        "SELECT CAST(data.\"commit\".\"collection\" AS VARCHAR), CAST(data.\"time_us\" AS BIGINT) FROM bluesky"
    )]
    fn expands_placeholders(
        #[case] engine: Engine,
        #[case] format: Format,
        #[case] expected: &str,
    ) {
        assert_eq!(
            JsonBenchBenchmark::expand_placeholders(engine, format, QUERY),
            expected
        );
    }
}
