// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [JSONBench](https://github.com/ClickHouse/JSONBench): analytics over semi-structured Bluesky
//! events, stored in a single Variant column.

mod data;

use std::fmt::Display;
use std::fs;
use std::path::PathBuf;
use std::str::FromStr;

use arrow_schema::DataType;
use futures::StreamExt;
use futures::TryStreamExt;
use parquet_variant_compute::ShreddedSchemaBuilder;
use serde::Deserialize;
use serde::Serialize;
use tracing::info;
use url::Url;
use vortex::utils::parallelism::get_available_parallelism;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::Engine;
use crate::TableSpec;
use crate::data_dir;
use crate::datasets::data_downloads::download_many;
use crate::idempotent;
use crate::utils::file::resolve_data_url;
use crate::workspace_root;

/// Each source file holds this many Bluesky events.
const ROWS_PER_FILE: usize = 1_000_000;

/// JSONBench publishes 1,000 source files (one billion events).
const MAX_FILES: usize = 1000;

/// URL of the `number`th (1-based) gzipped NDJSON source file.
fn source_url(number: usize) -> String {
    format!("https://clickhouse-public-datasets.s3.amazonaws.com/bluesky/file_{number:04}.json.gz")
}

/// Which fields of the Variant column are shredded into typed columns.
///
/// Fields that are not shredded stay readable through the residual Variant value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Shredding {
    /// No shredding: every document is a single binary Variant value.
    #[default]
    None,
    /// The top-level `kind`, `did`, and `time_us` fields.
    Partial,
    /// Every field the queries read: `partial` plus `commit.collection` and `commit.operation`.
    QueryFields,
}

impl Shredding {
    pub fn name(&self) -> &'static str {
        match self {
            Shredding::None => "none",
            Shredding::Partial => "partial",
            Shredding::QueryFields => "query-fields",
        }
    }

    /// The Parquet Variant shredding schema, or `None` when nothing is shredded.
    fn schema(&self) -> anyhow::Result<Option<DataType>> {
        if *self == Shredding::None {
            return Ok(None);
        }
        let mut builder = ShreddedSchemaBuilder::new()
            .with_path("kind", &DataType::Utf8)?
            .with_path("did", &DataType::Utf8)?
            .with_path("time_us", &DataType::Int64)?;
        if *self == Shredding::QueryFields {
            builder = builder
                .with_path("commit.collection", &DataType::Utf8)?
                .with_path("commit.operation", &DataType::Utf8)?;
        }
        Ok(Some(builder.build()))
    }
}

impl FromStr for Shredding {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "none" => Ok(Shredding::None),
            "partial" => Ok(Shredding::Partial),
            "query-fields" => Ok(Shredding::QueryFields),
            _ => anyhow::bail!(
                "Failed to parse {s}, valid shredding values are `none`, `partial`, and \
                 `query-fields`"
            ),
        }
    }
}

impl Display for Shredding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// JSONBench over a single `bluesky` table whose only column, `data`, is a Variant.
pub struct JsonBenchBenchmark {
    data_url: Url,
    /// How many million-row source files the dataset holds.
    files: usize,
    shredding: Shredding,
}

impl JsonBenchBenchmark {
    /// Creates the benchmark over `files` million-row source files.
    pub fn new(
        files: usize,
        shredding: Shredding,
        remote_data_dir: Option<String>,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            (1..=MAX_FILES).contains(&files),
            "jsonbench scale factor must be between 1 and {MAX_FILES} (millions of rows), got \
             {files}"
        );
        let data_url = resolve_data_url(
            remote_data_dir.as_deref(),
            &format!("jsonbench/{files}m-{shredding}"),
        )?;
        Ok(Self {
            data_url,
            files,
            shredding,
        })
    }

    /// Where the `number`th source file is cached. All shredding layouts share the downloads.
    fn source_path(number: usize) -> PathBuf {
        data_dir()
            .join("jsonbench")
            .join("json")
            .join(format!("file_{number:04}.json.gz"))
    }

    fn parquet_path(&self, number: usize) -> anyhow::Result<PathBuf> {
        self.data_url
            .join("parquet/")?
            .join(&format!("bluesky_{number:04}.parquet"))?
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("failed to convert data URL to filesystem path"))
    }

    /// Loads `sql/jsonbench/<engine>.sql`, numbering its statements from Q0 in file order.
    fn load_queries(engine_name: &str) -> anyhow::Result<Vec<(usize, String)>> {
        // `;`-separated; a `;` must not appear in a comment, or it would split a statement in two.
        let queries_file = workspace_root()
            .join("vortex-bench")
            .join("sql")
            .join("jsonbench")
            .join(engine_name)
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
}

#[async_trait::async_trait]
impl Benchmark for JsonBenchBenchmark {
    fn doc_path(&self) -> &'static str {
        "vortex-bench/sql/jsonbench.md"
    }

    /// The five JSONBench queries in DataFusion's dialect.
    fn queries(&self) -> anyhow::Result<Vec<(usize, String)>> {
        self.queries_for_engine(Engine::DataFusion)
    }

    /// Variant access has no common SQL syntax, so each engine has its own query file.
    fn queries_for_engine(&self, engine: Engine) -> anyhow::Result<Vec<(usize, String)>> {
        match engine {
            Engine::DataFusion => Self::load_queries("datafusion"),
            Engine::DuckDB => Self::load_queries("duckdb"),
            Engine::Vortex => anyhow::bail!("jsonbench has no queries for the {engine} engine"),
        }
    }

    async fn generate_base_data(&self) -> anyhow::Result<()> {
        if self.data_url.scheme() != "file" {
            return Ok(());
        }

        let missing = (1..=self.files)
            .map(|number| Ok((number, self.parquet_path(number)?)))
            .filter(|entry| !matches!(entry, Ok((_, path)) if path.exists()))
            .collect::<anyhow::Result<Vec<_>>>()?;
        if missing.is_empty() {
            return Ok(());
        }

        info!("Downloading {} JSONBench source files", missing.len());
        download_many(
            missing
                .iter()
                .map(|(number, _)| (Self::source_path(*number), source_url(*number))),
        )
        .await?;

        let shredding = self.shredding.schema()?;
        // Parsing JSON is CPU bound, so convert one source file per core.
        let concurrency = get_available_parallelism().unwrap_or(1);
        futures::stream::iter(missing)
            .map(|(number, parquet_path)| {
                let shredding = shredding.clone();
                tokio::task::spawn_blocking(move || {
                    idempotent(&parquet_path, |tmp_path| {
                        info!("Converting JSONBench file {number} to Parquet Variant");
                        data::convert_json_to_parquet(
                            &Self::source_path(number),
                            tmp_path,
                            shredding.as_ref(),
                        )
                    })
                })
            })
            .buffer_unordered(concurrency)
            .map(|joined| joined?)
            .try_collect::<Vec<_>>()
            .await?;

        Ok(())
    }

    fn expected_row_counts(&self) -> Option<Vec<usize>> {
        // Only the first source file has known counts: more files add collections.
        (self.files == 1).then(|| vec![15, 13, 3, 3, 3])
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::JsonBench {
            n_rows: self.files * ROWS_PER_FILE,
            shredding: self.shredding,
        }
    }

    fn dataset_name(&self) -> &str {
        "jsonbench"
    }

    fn dataset_display(&self) -> String {
        format!("jsonbench(sf={}, shredding={})", self.files, self.shredding)
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }

    fn table_specs(&self) -> Vec<TableSpec> {
        vec![TableSpec::new("bluesky", None)]
    }
}
