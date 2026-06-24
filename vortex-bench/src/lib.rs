// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
#![expect(clippy::expect_used)]

use std::clone::Clone;
use std::fmt::Display;
use std::str::FromStr;
use std::sync::LazyLock;

use anyhow::bail;
use appian::AppianBenchmark;
use clap::ValueEnum;
use clickbench::ClickBenchBenchmark;
use clickbench::ClickBenchSortedBenchmark;
use clickbench::Flavor;
use fineweb::FinewebBenchmark;
use itertools::Itertools;
use polarsignals::PolarSignalsBenchmark;
use public_bi::PBIDataset;
use public_bi::PublicBiBenchmark;
use realnest::gharchive::GithubArchiveBenchmark;
use serde::Deserialize;
use serde::Serialize;
use statpopgen::StatPopGenBenchmark;
use tpcds::TpcDsBenchmark;
use tpch::benchmark::TpcHBenchmark;
pub use utils::file::*;
pub use utils::logging::*;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::error::VortexExpect;
use vortex::error::vortex_err;
use vortex::file::VortexWriteOptions;
use vortex::file::WriteStrategyBuilder;
use vortex::utils::aliases::hash_map::HashMap;

pub mod appian;
pub mod benchmark;
pub mod clickbench;
pub mod compress;
pub mod conversions;
pub mod datasets;
pub mod display;
pub mod downloadable_dataset;
pub mod fineweb;
pub mod measurements;
pub mod memory;
pub mod output;
pub mod polarsignals;
pub mod public_bi;
pub mod random_access;
pub mod realnest;
pub mod runner;
pub mod statpopgen;
pub mod tpcds;
pub mod tpch;
pub mod utils;
pub mod v3;
pub mod vector_dataset;

pub use benchmark::Benchmark;
pub use benchmark::TableSpec;
pub use datasets::BenchmarkDataset;
pub use output::BenchmarkOutput;
pub use output::create_output_writer;
use vortex::VortexSessionDefault;
pub use vortex::error::vortex_panic;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;

// All benchmarks run with mimalloc for consistency.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::default().with_tokio());

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
    #[clap(name = "csv")]
    Csv,
    #[clap(name = "arrow")]
    Arrow,
    #[clap(name = "parquet")]
    Parquet,
    #[clap(name = "vortex")]
    #[serde(rename = "vortex")]
    OnDiskVortex,
    #[clap(name = "vortex-compact")]
    #[serde(rename = "vortex-compact")]
    VortexCompact,
    #[clap(name = "duckdb")]
    #[serde(rename = "duckdb")]
    OnDiskDuckDB,
    #[clap(name = "lance")]
    #[serde(rename = "lance")]
    Lance,
}

impl Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Allowed formats for benchmark CLI arguments.
pub const ALLOWED_FORMATS: &[Format] = &[Format::Parquet, Format::OnDiskVortex, Format::Lance];

impl Format {
    /// Clap value parser that only accepts parquet, vortex, and lance.
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
            Format::Csv => "csv",
            Format::Arrow => "arrow",
            Format::Parquet => "parquet",
            Format::OnDiskVortex => "vortex-file-compressed",
            Format::VortexCompact => "vortex-compact",
            Format::OnDiskDuckDB => "duckdb",
            Format::Lance => "lance",
        }
    }

    pub fn ext(&self) -> &'static str {
        match self {
            Format::Csv => "csv",
            Format::Arrow => "arrow",
            Format::Parquet => "parquet",
            Format::OnDiskVortex => "vortex",
            Format::VortexCompact => "vortex",
            Format::OnDiskDuckDB => "duckdb",
            Format::Lance => "lance",
        }
    }
}

#[derive(ValueEnum, Clone, Copy, Debug, Hash, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Engine {
    #[default]
    Vortex,
    Arrow,
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
            Engine::Arrow => write!(f, "arrow"),
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
                WriteStrategyBuilder::default()
                    .with_btrblocks_builder(BtrBlocksCompressorBuilder::default().with_compact())
                    .build(),
            ),
            CompactionStrategy::Default => options,
        }
    }
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
    #[clap(name = "statpopgen")]
    StatPopGen,
    #[clap(name = "fineweb")]
    Fineweb,
    #[clap(name = "gharchive")]
    GhArchive,
    #[clap(name = "polarsignals")]
    PolarSignals,
    #[clap(name = "public-bi")]
    PublicBi,
}

/// Default scale factor for TPC-related benchmarks
const DEFAULT_SCALE_FACTOR: &str = "1.0";

const SCALE_FACTOR_KEY: &str = "scale-factor";
const REMOTE_DATA_KEY: &str = "remote-data-dir";

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
            let remote_data_dir = opts.get_as::<String>(REMOTE_DATA_KEY);
            let benchmark = ClickBenchBenchmark::new(flavor, None, remote_data_dir)?;
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
            let benchmark = TpcHBenchmark::new(scale_factor.to_string(), remote_data_dir)?;
            Ok(Box::new(benchmark) as _)
        }
        BenchmarkArg::TpcDS => {
            let scale_factor = opts.get(SCALE_FACTOR_KEY).unwrap_or(DEFAULT_SCALE_FACTOR);
            let remote_data_dir = opts.get_as::<String>(REMOTE_DATA_KEY);
            let benchmark = TpcDsBenchmark::new(scale_factor.to_string(), remote_data_dir)?;
            Ok(Box::new(benchmark) as _)
        }
        BenchmarkArg::StatPopGen => {
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
        BenchmarkArg::PolarSignals => {
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

        tracing::info!(
            name,
            base_dir,
            pattern,
            format = load_format.name(),
            "Registering DuckDB {}",
            object_type.to_lowercase()
        );

        sql_statements.push(format!(
            "CREATE {object_type} IF NOT EXISTS {name} AS SELECT * FROM read_{extension}('{base_dir}/{pattern}');\n",
        ));
    }

    sql_statements
}
