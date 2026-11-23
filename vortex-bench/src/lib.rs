// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
#![expect(clippy::expect_used)]

use std::clone::Clone;
use std::fmt::Display;
use std::path::Path;
use std::process::Command;
use std::str::FromStr;
use std::sync::LazyLock;

use anyhow::bail;
use appian::AppianBenchmark;
use clap::ValueEnum;
use clickbench::CLICKBENCH_FILES;
use clickbench::ClickBenchBenchmark;
use clickbench::ClickBenchSortedBenchmark;
use clickbench::Flavor;
use fineweb::FinewebBenchmark;
use itertools::Itertools;
use job::JobBenchmark;
use polarsignals::PolarSignalsBenchmark;
use public_bi::PBIDataset;
use public_bi::PublicBiBenchmark;
use realnest::gharchive::GithubArchiveBenchmark;
use serde::Deserialize;
use serde::Serialize;
use statpopgen::StatPopGenBenchmark;
use tpcds::TpcDsBenchmark;
use tpch::benchmark::TpcHBenchmark;
use tpch_nested::Shape;
use tpch_nested::TpcHNestedBenchmark;
pub use utils::file::*;
pub use utils::logging::*;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::error::VortexExpect;
use vortex::error::vortex_err;
use vortex::file::VortexWriteOptions;
use vortex::file::WriteStrategyBuilder;
use vortex::utils::aliases::hash_map::HashMap;

use crate::spatialbench::SpatialBenchBenchmark;
use crate::vortex_queries::VortexBenchmark;

pub mod appian;
pub mod benchmark;
pub mod clickbench;
pub mod compress;
pub mod conversions;
pub mod datasets;
pub mod display;
pub mod downloadable_dataset;
pub mod fineweb;
pub mod job;
pub mod measurements;
pub mod memory;
pub mod output;
pub mod parquet_copies;
pub mod polarsignals;
pub mod public_bi;
pub mod random_access;
pub mod realnest;
pub mod runner;
pub mod spatialbench;
pub mod statpopgen;
pub mod tpcds;
pub mod tpch;
pub mod tpch_nested;
pub mod utils;
pub mod v3;
pub mod vector_dataset;
pub mod vortex_queries;

pub use benchmark::Benchmark;
pub use benchmark::TableSpec;
pub use datasets::BenchmarkDataset;
pub use output::BenchmarkOutput;
pub use output::create_output_writer;
use vortex::VortexSessionDefault;
use vortex::editions::DEFAULT_PREVIEW_EDITION;
use vortex::editions::Edition;
use vortex::editions::EditionDeclaration;
use vortex::editions::EditionFamily;
use vortex::editions::EditionId;
use vortex::editions::EditionMember;
use vortex::editions::EditionSessionExt;
use vortex::editions::UNSTABLE;
pub use vortex::error::vortex_panic;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;

/// Tunes mimalloc for benchmarking. Call this first thing in each benchmark binary's `main`.
///
/// mimalloc returns freed memory to the OS after `purge_delay` milliseconds. That purging runs
/// in the middle of measured queries, so it is disabled and freed memory stays with the process
/// for the whole run.
pub fn configure_allocator() {
    // `mi_option_purge_delay`, which `libmimalloc-sys` does not export as a constant.
    const MI_OPTION_PURGE_DELAY: libmimalloc_sys::mi_option_t = 15;
    // SAFETY: setting a mimalloc option has no memory-safety preconditions and is valid at any
    // time; `-1` is the documented value for "never purge".
    unsafe { libmimalloc_sys::mi_option_set(MI_OPTION_PURGE_DELAY, -1) };
}

const BENCH_EDITION_FAMILY: EditionFamily = EditionFamily {
    name: "bench",
    origin: "vortex-bench",
    doc: "Unstable components that benchmarks write, without the rest of the unstable edition.",
};

/// The list layout from the unstable edition, on its own.
const UNSTABLE_BENCH: EditionId = EditionId::new("bench", 2026, 10, 0);

static UNSTABLE_BENCH_DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: UNSTABLE_BENCH,
        immutable: false,
        min_library_version: None,
    },
    added: &[EditionMember::layout(&"vortex.list")],
};

/// Enables the editions holding encodings not yet in `core` on `session`: the newest `preview`
/// edition ([`DEFAULT_PREVIEW_EDITION`]) and [`UNSTABLE`]. Its compressor may then also choose
/// those encodings.
///
/// Every benchmark session calls this, so benchmarks measure Vortex files written with them.
pub fn enable_unstable_encodings(session: &VortexSession) {
    session
        .enable_edition(DEFAULT_PREVIEW_EDITION)
        .vortex_expect("default preview edition is registered");
    session
        .enable_edition(UNSTABLE)
        .vortex_expect("unstable edition is registered");
}

pub static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::default().with_tokio();
    vortex_spatial::initialize(&session);
    session
        .editions()
        .declare_family(&BENCH_EDITION_FAMILY)
        .vortex_expect("bench edition family is valid");
    session
        .register_edition(&UNSTABLE_BENCH_DECLARATION)
        .vortex_expect("bench edition declaration is valid");
    enable_unstable_encodings(&session);
    session
});

/// Enables [`UNSTABLE_BENCH`] on [`SESSION`], so it writes lists with the list layout.
///
/// The V2 scan cannot read the list layout yet, so only suites that read through V1 call this.
fn enable_unstable_bench_edition() {
    SESSION
        .enable_edition(UNSTABLE_BENCH)
        .vortex_expect("bench edition is registered");
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct Target {
    pub engine: Engine,
    pub format: Format,
}

impl FromStr for Target {
    type Err = anyhow::Error;

    fn from_str(target_string: &str) -> Result<Self, Self::Err> {
        let split = target_string.split(":").collect_vec();
        let [engine_str, format_str] = split.as_slice() else {
            vortex_panic!("invalid target string {}", target_string);
        };

        Ok(Self {
            engine: Engine::from_str(engine_str, true)
                .map_err(|e| {
                    vortex_err!(
                        "cannot convert str ({}) to an Engine oneof([{}]), got error {}",
                        *engine_str,
                        Engine::value_variants().iter().join(","),
                        e
                    )
                })
                .vortex_expect("operation should succeed in benchmark"),
            format: Format::from_str(format_str, true)
                .map_err(|e| {
                    vortex_err!(
                        "cannot convert str ({}) to a Format oneof([{}]), got error {}",
                        *format_str,
                        Format::value_variants().iter().join(","),
                        e
                    )
                })
                .vortex_expect("operation should succeed in benchmark"),
        })
    }
}

impl Target {
    pub fn new(engine: Engine, format: Format) -> Self {
        Self { engine, format }
    }
}

impl Display for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.engine, self.format)
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Format {
    #[clap(name = "arrow-ipc")]
    ArrowIpc,
    #[clap(name = "csv")]
    Csv,
    #[clap(name = "parquet")]
    Parquet,
    #[clap(name = "vortex")]
    #[serde(rename = "vortex")]
    OnDiskVortex,
    #[clap(name = "vortex-compact")]
    #[serde(rename = "vortex-compact")]
    VortexCompact,
    #[clap(name = "vortex-spatial-native")]
    #[serde(rename = "vortex-spatial-native")]
    VortexSpatialNative,
    #[clap(name = "duckdb")]
    #[serde(rename = "duckdb")]
    OnDiskDuckDB,
    #[clap(name = "lance")]
    #[serde(rename = "lance")]
    Lance,
    #[clap(name = "litdata")]
    #[serde(rename = "litdata")]
    LitData,
}

impl Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Allowed formats for benchmark CLI arguments.
pub const ALLOWED_FORMATS: &[Format] = &[
    Format::ArrowIpc,
    Format::Parquet,
    Format::OnDiskVortex,
    Format::Lance,
    Format::LitData,
];

impl Format {
    /// Clap value parser that only accepts formats supported by random-access benchmarks.
    pub fn parse_allowed(s: &str) -> Result<Format, String> {
        let format = Format::from_str(s, true)?;
        if ALLOWED_FORMATS.contains(&format) {
            Ok(format)
        } else {
            Err(format!(
                "invalid format '{}': allowed values are [{}]",
                s,
                ALLOWED_FORMATS.iter().map(|f| f.to_string()).join(", "),
            ))
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Format::ArrowIpc => "arrow-ipc",
            Format::Csv => "csv",
            Format::Parquet => "parquet",
            Format::OnDiskVortex => "vortex-file-compressed",
            Format::VortexCompact => "vortex-compact",
            Format::VortexSpatialNative => "vortex-spatial-native",
            Format::OnDiskDuckDB => "duckdb",
            Format::Lance => "lance",
            Format::LitData => "litdata",
        }
    }

    pub fn ext(&self) -> &'static str {
        match self {
            Format::ArrowIpc => "arrow",
            Format::Csv => "csv",
            Format::Parquet => "parquet",
            Format::OnDiskVortex => "vortex",
            Format::VortexCompact => "vortex",
            Format::VortexSpatialNative => "vortex",
            Format::OnDiskDuckDB => "duckdb",
            Format::Lance => "lance",
            Format::LitData => "litdata",
        }
    }
}

#[derive(ValueEnum, Clone, Copy, Debug, Hash, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Engine {
    #[default]
    Vortex,
    #[clap(name = "datafusion")]
    #[serde(rename = "datafusion")]
    DataFusion,
    #[clap(name = "duckdb")]
    #[serde(rename = "duckdb")]
    DuckDB,
}

impl Display for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Engine::DataFusion => write!(f, "datafusion"),
            Engine::DuckDB => write!(f, "duckdb"),
            Engine::Vortex => write!(f, "vortex"),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub enum CompactionStrategy {
    Compact,
    #[default]
    Default,
}

impl CompactionStrategy {
    pub fn apply_options(&self, options: VortexWriteOptions) -> VortexWriteOptions {
        match self {
            CompactionStrategy::Compact => options.with_strategy(
                WriteStrategyBuilder::from_session(&SESSION)
                    .with_btrblocks_builder(
                        BtrBlocksCompressorBuilder::from_session(&SESSION).with_compact(),
                    )
                    .build(),
            ),
            CompactionStrategy::Default => options,
        }
    }
}

/// Verify that local data has already been prepared for the requested benchmark formats.
///
/// Engine-specific benchmark binaries call this before running queries. Data generation itself
/// belongs to the `data-gen` binary.
pub fn require_prepared_data<B>(benchmark: &B, formats: &[Format]) -> anyhow::Result<()>
where
    B: Benchmark + ?Sized,
{
    if benchmark.data_url().scheme() != "file" {
        return Ok(());
    }

    let base_path = benchmark
        .data_url()
        .to_file_path()
        .map_err(|_| anyhow::anyhow!("Invalid file URL: {}", benchmark.data_url()))?;

    let mut missing = Vec::new();
    for format in formats.iter().copied().unique() {
        let required_path = match format {
            Format::Parquet => base_path.join(Format::Parquet.name()),
            Format::OnDiskVortex => base_path.join(Format::OnDiskVortex.name()),
            Format::VortexCompact => base_path.join(Format::VortexCompact.name()),
            Format::OnDiskDuckDB => base_path
                .join(Format::OnDiskDuckDB.name())
                .join("duckdb.db"),
            format => base_path.join(format.name()),
        };

        if !required_path.exists() {
            missing.push((format, required_path));
        }
    }

    if missing.is_empty() {
        return Ok(());
    }

    let missing_data = missing
        .iter()
        .map(|(format, path)| format!("{format} ({})", path.display()))
        .join(", ");
    let requested_formats = formats
        .iter()
        .copied()
        .unique()
        .map(|format| format!("\"{format}\""))
        .join(",");

    anyhow::bail!(
        "prepared data is missing for {}: {missing_data}. Generate it first with \
         `vx-bench prepare-data {} --formats-json '[{requested_formats}]'` or \
         `cargo run --bin data-gen -- {} --formats {}` using the same --opt values.",
        benchmark.dataset_display(),
        benchmark.dataset_name(),
        benchmark.dataset_name(),
        formats.iter().copied().unique().join(","),
    );
}

/// CLI argument for selecting which benchmark to run.
#[derive(clap::ValueEnum, Clone, Copy)]
pub enum BenchmarkArg {
    #[clap(name = "appian")]
    Appian,
    #[clap(name = "clickbench")]
    ClickBench,
    #[clap(name = "clickbench-sorted")]
    ClickBenchSorted,
    #[clap(name = "tpch")]
    TpcH,
    #[clap(name = "tpcds")]
    TpcDS,
    #[clap(name = "tpch-nested")]
    TpcHNested,
    #[clap(name = "statpopgen")]
    StatPopGen,
    #[clap(name = "fineweb")]
    Fineweb,
    #[clap(name = "gharchive")]
    GhArchive,
    #[clap(name = "job")]
    Job,
    #[clap(name = "polarsignals")]
    PolarSignals,
    #[clap(name = "public-bi")]
    PublicBi,
    #[clap(name = "spatialbench")]
    SpatialBench,
    #[clap(name = "vortex")]
    VortexQueries,
}

/// Default scale factor for TPC-related benchmarks
const DEFAULT_SCALE_FACTOR: &str = "1.0";

const SCALE_FACTOR_KEY: &str = "scale-factor";
const REMOTE_DATA_KEY: &str = "remote-data-dir";
/// Rows in each row group of the Parquet that TPC-H, TPC-DS, and nested TPC-H generate. The
/// default is [`LOCAL_ROW_GROUP_ROWS`]; the benchmarks read from S3 use [`S3_ROW_GROUP_ROWS`].
///
/// [`LOCAL_ROW_GROUP_ROWS`]: utils::parquet::LOCAL_ROW_GROUP_ROWS
/// [`S3_ROW_GROUP_ROWS`]: utils::parquet::S3_ROW_GROUP_ROWS
const ROW_GROUP_ROWS_KEY: &str = "row-group-rows";

/// The `row-group-rows` option, or [`utils::parquet::LOCAL_ROW_GROUP_ROWS`].
fn row_group_rows(opts: &Opts) -> usize {
    opts.get_as::<usize>(ROW_GROUP_ROWS_KEY)
        .unwrap_or(utils::parquet::LOCAL_ROW_GROUP_ROWS)
}

/// Factory function to create a benchmark instance from CLI arguments.
pub fn create_benchmark(b: BenchmarkArg, opts: &Opts) -> anyhow::Result<Box<dyn Benchmark>> {
    match b {
        BenchmarkArg::Appian => {
            let remote_data_dir = opts.get_as::<String>(REMOTE_DATA_KEY);
            let benchmark = AppianBenchmark::with_remote_data_dir(remote_data_dir)?;
            Ok(Box::new(benchmark) as _)
        }
        BenchmarkArg::ClickBench => {
            let flavor = opts.get_as::<Flavor>("flavor").unwrap_or_default();
            let files = opts.get_as::<u32>("files").unwrap_or(CLICKBENCH_FILES);
            let remote_data_dir = opts.get_as::<String>(REMOTE_DATA_KEY);
            let benchmark = ClickBenchBenchmark::new(flavor, files, None, remote_data_dir)?;
            Ok(Box::new(benchmark) as _)
        }
        BenchmarkArg::ClickBenchSorted => {
            let remote_data_dir = opts.get_as::<String>(REMOTE_DATA_KEY);
            let benchmark = ClickBenchSortedBenchmark::new(remote_data_dir)?;
            Ok(Box::new(benchmark) as _)
        }
        BenchmarkArg::TpcH => {
            let scale_factor = opts.get(SCALE_FACTOR_KEY).unwrap_or(DEFAULT_SCALE_FACTOR);
            let remote_data_dir = opts.get_as::<String>(REMOTE_DATA_KEY);
            let benchmark = TpcHBenchmark::new(scale_factor.to_string(), remote_data_dir)?
                .with_row_group_rows(row_group_rows(opts));
            Ok(Box::new(benchmark) as _)
        }
        BenchmarkArg::TpcDS => {
            let scale_factor = opts.get(SCALE_FACTOR_KEY).unwrap_or(DEFAULT_SCALE_FACTOR);
            let remote_data_dir = opts.get_as::<String>(REMOTE_DATA_KEY);
            let benchmark = TpcDsBenchmark::new(scale_factor.to_string(), remote_data_dir)?
                .with_row_group_rows(row_group_rows(opts));
            Ok(Box::new(benchmark) as _)
        }
        BenchmarkArg::TpcHNested => {
            enable_unstable_bench_edition();
            let scale_factor = opts.get(SCALE_FACTOR_KEY).unwrap_or(DEFAULT_SCALE_FACTOR);
            let shape = opts.get_as::<Shape>("shape").unwrap_or_default();
            let remote_data_dir = opts.get_as::<String>(REMOTE_DATA_KEY);
            let benchmark =
                TpcHNestedBenchmark::new(scale_factor.to_string(), shape, remote_data_dir)?
                    .with_row_group_rows(row_group_rows(opts));
            Ok(Box::new(benchmark) as _)
        }
        BenchmarkArg::StatPopGen => {
            enable_unstable_bench_edition();
            let scale_factor = opts.get_as::<u64>(SCALE_FACTOR_KEY).unwrap_or(1);
            let benchmark = StatPopGenBenchmark::new(scale_factor)?;
            Ok(Box::new(benchmark) as _)
        }
        BenchmarkArg::Fineweb => {
            let remote_data_dir = opts.get_as::<String>(REMOTE_DATA_KEY);
            let benchmark = FinewebBenchmark::with_remote_data_dir(remote_data_dir)?;
            Ok(Box::new(benchmark) as _)
        }
        BenchmarkArg::GhArchive => {
            let remote_data_dir = opts.get_as::<String>(REMOTE_DATA_KEY);
            let benchmark = GithubArchiveBenchmark::with_remote_data_dir(remote_data_dir)?;
            Ok(Box::new(benchmark) as _)
        }
        BenchmarkArg::Job => {
            let remote_data_dir = opts.get_as::<String>(REMOTE_DATA_KEY);
            let benchmark = JobBenchmark::with_remote_data_dir(remote_data_dir)?;
            Ok(Box::new(benchmark) as _)
        }
        BenchmarkArg::PolarSignals => {
            enable_unstable_bench_edition();
            let scale_factor = opts.get_as::<usize>(SCALE_FACTOR_KEY).unwrap_or(1);
            let benchmark = PolarSignalsBenchmark::new(scale_factor)?;
            Ok(Box::new(benchmark) as _)
        }
        BenchmarkArg::PublicBi => {
            let dataset = opts.get_as::<PBIDataset>("dataset").ok_or_else(|| {
                anyhow::anyhow!("public-bi benchmark requires --opt dataset=<name>")
            })?;
            let benchmark = PublicBiBenchmark::new(dataset)?;
            Ok(Box::new(benchmark) as _)
        }
        BenchmarkArg::SpatialBench => {
            let scale_factor = opts.get(SCALE_FACTOR_KEY).unwrap_or(DEFAULT_SCALE_FACTOR);
            let remote_data_dir = opts.get_as::<String>(REMOTE_DATA_KEY);
            let benchmark = SpatialBenchBenchmark::new(scale_factor.to_string(), remote_data_dir)?;
            Ok(Box::new(benchmark) as _)
        }
        BenchmarkArg::VortexQueries => {
            let mut benchmark = VortexBenchmark::new()?;
            if let Some(query) = opts.get("query") {
                benchmark = benchmark.with_query(query)?;
            }
            Ok(Box::new(benchmark) as _)
        }
    }
}

/// A single key-value option for benchmark configuration.
#[derive(Clone, Debug)]
pub struct Opt {
    key: String,
    value: String,
}

/// Collection of benchmark configuration options.
pub struct Opts {
    inner: HashMap<String, String>,
}

impl Opts {
    pub fn get(&self, key: &str) -> Option<&str> {
        self.inner.get(key).map(|s| s.as_str())
    }

    #[expect(clippy::panic)]
    pub fn get_as<T>(&self, key: &str) -> Option<T>
    where
        T: FromStr,
        <T as FromStr>::Err: std::fmt::Debug,
    {
        self.inner.get(key).map(|v| {
            v.parse().unwrap_or_else(|_| {
                panic!("opts value {key} was parsed into an inappropriate type")
            })
        })
    }
}

impl From<Vec<Opt>> for Opts {
    fn from(value: Vec<Opt>) -> Self {
        value.into_iter().collect()
    }
}

impl FromIterator<Opt> for Opts {
    fn from_iter<T: IntoIterator<Item = Opt>>(iter: T) -> Self {
        let inner = HashMap::from_iter(iter.into_iter().map(|o| (o.key, o.value)));
        Self { inner }
    }
}

impl Display for Opt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}={}", self.key, self.value)
    }
}

impl FromStr for Opt {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let split = s.split([' ', '=']).collect::<Vec<_>>();
        let [key, value] = split.as_slice() else {
            bail!("invalid option: {}", s);
        };

        let opt = Opt {
            key: key.to_string(),
            value: value.to_string(),
        };

        Ok(opt)
    }
}

/// Generate SQL commands to create DuckDB tables/views from data files.
///
/// # Arguments
/// * `benchmark` - The benchmark providing table specs and patterns
/// * `base_dir` - Base directory path (without trailing slash)
/// * `load_format` - The format to load from (determines file extension)
/// * `object_type` - Either "TABLE" or "VIEW"
pub fn generate_duckdb_registration_sql<B>(
    benchmark: &B,
    base_dir: &str,
    load_format: Format,
    object_type: &str,
) -> Vec<String>
where
    B: Benchmark + ?Sized,
{
    let extension = load_format.ext();
    let mut sql_statements = Vec::new();

    for table_spec in benchmark.table_specs() {
        let name = table_spec.name;
        let pattern = benchmark
            .pattern(name, load_format)
            .map(|p| p.to_string())
            .unwrap_or_else(|| format!("*.{}", extension));

        sql_statements.push(format!(
            "CREATE {object_type} IF NOT EXISTS {name} AS SELECT * FROM read_{extension}('{base_dir}/{pattern}');\n",
        ));
    }

    sql_statements
}

/// Generate a DuckDB database from Parquet files using the DuckDB CLI.
pub fn generate_duckdb(base_path: &Path, benchmark: &dyn Benchmark) -> anyhow::Result<()> {
    generate_duckdb_with(base_path, benchmark, |db_path, stmt| {
        let output = Command::new("duckdb")
            .arg(db_path)
            .arg("-c")
            .arg(stmt)
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("DuckDB CLI failed: {}", stderr);
        }

        Ok(())
    })
}

fn generate_duckdb_with(
    base_path: &Path,
    benchmark: &dyn Benchmark,
    mut run_stmt: impl FnMut(&Path, &str) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let duckdb_dir = base_path.join(Format::OnDiskDuckDB.name());
    std::fs::create_dir_all(&duckdb_dir)?;

    let db_path = duckdb_dir.join("duckdb.db");

    // Skip if database already exists
    if db_path.exists() {
        tracing::info!("DuckDB database already exists at {}", db_path.display());
        return Ok(());
    }

    let parquet_dir = base_path.join(Format::Parquet.name());
    let sql = generate_duckdb_registration_sql(
        benchmark,
        parquet_dir
            .to_str()
            .vortex_expect("value must be str displayable"),
        Format::Parquet,
        "TABLE",
    );
    if sql.is_empty() {
        return Ok(());
    }

    let staging_dir = tempfile::tempdir_in(&duckdb_dir)?;
    let staged_db_path = staging_dir.path().join("duckdb.db");

    for stmt in sql {
        run_stmt(&staged_db_path, &stmt)?;
    }

    std::fs::rename(&staged_db_path, &db_path)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::path::Path;

    use anyhow::Result;
    use anyhow::bail;
    use glob::Pattern;
    use url::Url;

    use crate::Benchmark;
    use crate::BenchmarkDataset;
    use crate::Format;
    use crate::TableSpec;
    use crate::generate_duckdb_with;

    struct TwoTableBenchmark {
        url: Url,
    }

    #[async_trait::async_trait]
    impl Benchmark for TwoTableBenchmark {
        fn queries(&self) -> Result<Vec<(usize, String)>> {
            Ok(Vec::new())
        }

        async fn generate_base_data(&self) -> Result<()> {
            Ok(())
        }

        fn dataset(&self) -> BenchmarkDataset {
            BenchmarkDataset::TpcH {
                scale_factor: "1".into(),
            }
        }

        fn doc_path(&self) -> &'static str {
            ""
        }

        fn dataset_name(&self) -> &str {
            "two_table"
        }

        fn dataset_display(&self) -> String {
            "two_table".into()
        }

        fn data_url(&self) -> &Url {
            &self.url
        }

        fn table_specs(&self) -> Vec<TableSpec> {
            vec![
                TableSpec::new("first_table", None),
                TableSpec::new("second_table", None),
            ]
        }

        fn pattern(&self, table_name: &str, format: Format) -> Option<Pattern> {
            Pattern::new(&format!("{table_name}.{}", format.ext())).ok()
        }
    }

    fn simulate_duckdb_import(parquet_dir: &Path, db_path: &Path, stmt: &str) -> Result<()> {
        let table = ["first_table", "second_table"]
            .into_iter()
            .find(|name| stmt.contains(name))
            .expect("statement names a known table");

        if !parquet_dir.join(format!("{table}.parquet")).exists() {
            bail!("missing input for {table}");
        }

        let mut file = OpenOptions::new().create(true).append(true).open(db_path)?;
        writeln!(file, "{table}")?;
        Ok(())
    }

    #[test]
    fn generate_duckdb_retries_after_partial_import_and_reuses_complete_db() -> Result<()> {
        let retry_root = tempfile::tempdir()?;
        let parquet_dir = retry_root.path().join(Format::Parquet.name());
        std::fs::create_dir_all(&parquet_dir)?;
        std::fs::write(parquet_dir.join("first_table.parquet"), b"p1")?;

        let bench = TwoTableBenchmark {
            url: Url::from_directory_path(retry_root.path())
                .map_err(|()| anyhow::anyhow!("invalid directory path"))?,
        };

        // First attempt fails on the missing second table and must not leave duckdb.db behind.
        let err = generate_duckdb_with(retry_root.path(), &bench, |db, stmt| {
            simulate_duckdb_import(&parquet_dir, db, stmt)
        })
        .unwrap_err();
        assert!(err.to_string().contains("second_table"));

        let duckdb_dir = retry_root.path().join(Format::OnDiskDuckDB.name());
        let db_path = duckdb_dir.join("duckdb.db");
        assert!(!db_path.exists());
        assert_eq!(std::fs::read_dir(&duckdb_dir)?.count(), 0);

        // Once the second input arrives, retrying at the same path publishes both tables.
        std::fs::write(parquet_dir.join("second_table.parquet"), b"p2")?;
        generate_duckdb_with(retry_root.path(), &bench, |db, stmt| {
            simulate_duckdb_import(&parquet_dir, db, stmt)
        })?;

        // Compare against a fresh generation with both inputs present from the start.
        let fresh_root = tempfile::tempdir()?;
        let fresh_parquet = fresh_root.path().join(Format::Parquet.name());
        std::fs::create_dir_all(&fresh_parquet)?;
        std::fs::write(fresh_parquet.join("first_table.parquet"), b"p1")?;
        std::fs::write(fresh_parquet.join("second_table.parquet"), b"p2")?;
        let fresh_bench = TwoTableBenchmark {
            url: Url::from_directory_path(fresh_root.path())
                .map_err(|()| anyhow::anyhow!("invalid directory path"))?,
        };
        generate_duckdb_with(fresh_root.path(), &fresh_bench, |db, stmt| {
            simulate_duckdb_import(&fresh_parquet, db, stmt)
        })?;

        let retry_contents = std::fs::read_to_string(&db_path)?;
        let fresh_contents = std::fs::read_to_string(
            fresh_root
                .path()
                .join(Format::OnDiskDuckDB.name())
                .join("duckdb.db"),
        )?;
        assert_eq!(retry_contents, "first_table\nsecond_table\n");
        assert_eq!(retry_contents, fresh_contents);

        // Re-running on an existing complete database reuses it without running any statements.
        let mut invoked = 0;
        generate_duckdb_with(retry_root.path(), &bench, |_db, _stmt| {
            invoked += 1;
            Ok(())
        })?;
        assert_eq!(invoked, 0);

        Ok(())
    }
}
