// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [JSONBench]: analytics over a dataset of Bluesky events stored as JSON documents.
//!
//! The raw NDJSON files are parsed into a single Variant column `data`, shredding the paths that
//! a sample of the documents shows to be common and consistently typed (see [`shredding`]). The
//! shredded Variant column is written to Parquet, which is the base for the other formats.
//!
//! Queries extract fields with the `variant_get(data, path, type)` function.
//!
//! [JSONBench]: https://github.com/ClickHouse/JSONBench

pub mod shredding;

use std::fs;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use arrow_array::ArrayRef;
use arrow_array::RecordBatch;
use arrow_array::StringArray;
use arrow_schema::DataType;
use arrow_schema::Schema;
use futures::StreamExt;
use futures::TryStreamExt;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::basic::ZstdLevel;
use parquet::file::properties::WriterProperties;
use parquet::variant::VariantArray;
use parquet::variant::json_to_variant;
use parquet::variant::shred_variant;
use serde::de::IgnoredAny;
use tokio::io::AsyncWriteExt;
use tracing::info;
use tracing::warn;
use url::Url;
use vortex::utils::parallelism::get_available_parallelism;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::TableSpec;
use crate::idempotent;
use crate::idempotent_async;
use crate::utils::file::resolve_data_url;
use crate::workspace_root;

/// Each source file holds one million documents.
const ROWS_PER_FILE: usize = 1_000_000;
/// Number of documents sampled from the first file to infer the shredding schema.
const SHREDDING_SAMPLE_ROWS: usize = 100_000;
/// Number of documents converted to Variant per Arrow batch.
const BATCH_ROWS: usize = 16 * 1024;
/// Parquet row group size; matches DuckDB's default so a file splits across threads.
const ROW_GROUP_ROWS: usize = 122_880;

fn json_url(file: usize) -> String {
    format!("https://clickhouse-public-datasets.s3.amazonaws.com/bluesky/file_{file:04}.json.gz")
}

/// JSONBench over the first `files` million-document files of the Bluesky dataset.
pub struct JsonBenchBenchmark {
    data_url: Url,
    files: usize,
}

impl JsonBenchBenchmark {
    /// Creates the benchmark over `files` source files, i.e. `files` million documents.
    pub fn new(files: usize, remote_data_dir: Option<String>) -> anyhow::Result<Self> {
        anyhow::ensure!(files > 0, "jsonbench needs at least one source file");
        let data_url = resolve_data_url(remote_data_dir.as_deref(), &Self::subdir(files))?;
        Ok(Self { data_url, files })
    }

    fn subdir(files: usize) -> String {
        format!("jsonbench/{}m", files * ROWS_PER_FILE / 1_000_000)
    }

    fn base_path(&self) -> anyhow::Result<PathBuf> {
        self.data_url
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("jsonbench data URL must be a file:// URL"))
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
        let contents = fs::read_to_string(queries_file)?;
        Ok(contents
            .split_terminator(';')
            .map(str::trim)
            .filter(|stmt| !stmt.is_empty())
            .map(str::to_string)
            .enumerate()
            .collect())
    }

    async fn generate_base_data(&self) -> anyhow::Result<()> {
        if self.data_url.scheme() != "file" {
            return Ok(());
        }
        let base_path = self.base_path()?;
        let json_dir = base_path.join("json");
        let parquet_dir = base_path.join("parquet");

        let mut json_paths = Vec::with_capacity(self.files);
        let client = reqwest::Client::new();
        for file in 1..=self.files {
            let path = json_dir.join(format!("file_{file:04}.json.gz"));
            let path = idempotent_async(path, |tmp| {
                let client = client.clone();
                async move {
                    let url = json_url(file);
                    info!("Downloading {url}");
                    let mut response = client.get(&url).send().await?.error_for_status()?;
                    let mut out = tokio::fs::File::create(&tmp).await?;
                    while let Some(chunk) = response.chunk().await? {
                        out.write_all(&chunk).await?;
                    }
                    out.flush().await?;
                    anyhow::Ok(())
                }
            })
            .await?;
            json_paths.push(path);
        }

        let first = json_paths
            .first()
            .ok_or_else(|| anyhow::anyhow!("no jsonbench source files"))?;
        let shredding = infer_shredding(first)?;
        match &shredding {
            Some(dtype) => info!(
                "Shredding Variant documents as {}",
                shredding::duckdb_shredding_type(dtype)
            ),
            None => info!("No common paths found, writing unshredded Variant documents"),
        }

        futures::stream::iter(json_paths.into_iter().enumerate())
            .map(|(idx, json_path)| {
                let parquet_path = parquet_dir.join(format!("bluesky_{:04}.parquet", idx + 1));
                let shredding = shredding.clone();
                tokio::task::spawn_blocking(move || {
                    idempotent(&parquet_path, |tmp| {
                        info!(
                            "Converting {} to {}",
                            json_path.display(),
                            parquet_path.display()
                        );
                        json_to_parquet(&json_path, tmp, shredding.as_ref())
                    })
                })
            })
            .buffer_unordered(get_available_parallelism().unwrap_or(1))
            .map(|result| result?)
            .try_collect::<Vec<_>>()
            .await?;

        Ok(())
    }

    fn expected_row_counts(&self) -> Option<Vec<usize>> {
        None
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::JsonBench { files: self.files }
    }

    fn doc_path(&self) -> &'static str {
        "vortex-bench/sql/jsonbench.md"
    }

    fn dataset_name(&self) -> &str {
        "jsonbench"
    }

    fn dataset_display(&self) -> String {
        format!("jsonbench({}m)", self.files * ROWS_PER_FILE / 1_000_000)
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }

    fn table_specs(&self) -> Vec<TableSpec> {
        vec![TableSpec::new("bluesky", None)]
    }
}

fn read_json_lines(path: &Path) -> anyhow::Result<impl Iterator<Item = std::io::Result<String>>> {
    let file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    Ok(BufReader::with_capacity(1 << 20, flate2::read::GzDecoder::new(file)).lines())
}

fn infer_shredding(path: &Path) -> anyhow::Result<Option<DataType>> {
    let sample = read_json_lines(path)?
        .take(SHREDDING_SAMPLE_ROWS)
        .collect::<std::io::Result<Vec<_>>>()?;
    Ok(shredding::infer_shredding_schema(
        sample.iter().map(String::as_str),
    ))
}

/// Parses NDJSON documents into a (shredded) Variant column and writes them to Parquet.
fn json_to_parquet(
    json_path: &Path,
    parquet_path: &Path,
    shredding: Option<&DataType>,
) -> anyhow::Result<()> {
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
        .set_max_row_group_row_count(Some(ROW_GROUP_ROWS))
        .build();
    let mut writer: Option<ArrowWriter<File>> = None;

    let mut lines = read_json_lines(json_path)?.peekable();
    let mut batch = Vec::with_capacity(BATCH_ROWS);
    while lines.peek().is_some() {
        batch.clear();
        for line in lines.by_ref().take(BATCH_ROWS) {
            batch.push(line?);
        }

        let strings: ArrayRef = Arc::new(StringArray::from_iter_values(&batch));
        let variant = match json_to_variant(&strings) {
            Ok(variant) => variant,
            Err(_) => {
                // A few source documents contain raw control characters, which split them
                // across lines. Like the JSONBench loaders, skip lines that are not valid JSON.
                let before = batch.len();
                batch.retain(|line| serde_json::from_str::<IgnoredAny>(line).is_ok());
                warn!(
                    "Skipping {} invalid JSON lines in {}",
                    before - batch.len(),
                    json_path.display()
                );
                let strings: ArrayRef = Arc::new(StringArray::from_iter_values(&batch));
                json_to_variant(&strings)?
            }
        };
        let variant: VariantArray = match shredding {
            Some(dtype) => shred_variant(&variant, dtype)?,
            None => variant,
        };
        let schema = Arc::new(Schema::new(vec![variant.field("data")]));
        let record_batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![ArrayRef::from(variant)])?;

        let writer = match &mut writer {
            Some(writer) => writer,
            None => writer.insert(ArrowWriter::try_new(
                File::create(parquet_path)?,
                schema,
                Some(props.clone()),
            )?),
        };
        writer.write(&record_batch)?;
    }

    writer
        .ok_or_else(|| anyhow::anyhow!("{} has no documents", json_path.display()))?
        .close()?;
    Ok(())
}
