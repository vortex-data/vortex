#![feature(exit_status_error)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]

use std::clone::Clone;
use std::fmt::Display;
use std::str::FromStr;

use clap::ValueEnum;
use itertools::Itertools;
use serde::Serialize;

pub mod bench_run;
pub mod clickbench;
pub mod compress;
pub mod conversions;
pub mod datasets;
pub mod display;
pub mod engines;
pub mod measurements;
pub mod metrics;
pub mod public_bi;
pub mod random_access;
pub mod tpch;
pub mod utils;

pub use datasets::{BenchmarkDataset, file};
pub use engines::{ddb, df};
pub use vortex::error::vortex_panic;

// All benchmarks run with mimalloc for consistency.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize)]
pub struct Target {
    engine: Engine,
    format: Format,
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
                .vortex_unwrap(),
            format: Format::from_str(format_str, true)
                .map_err(|e| {
                    vortex_err!(
                        "cannot convert str ({}) to a Format oneof([{}]), got error {}",
                        *format_str,
                        Format::value_variants().iter().join(","),
                        e
                    )
                })
                .vortex_unwrap(),
        })
    }
}

impl Target {
    pub fn new(engine: Engine, format: Format) -> Self {
        Self { engine, format }
    }

    pub fn engine(&self) -> Engine {
        self.engine
    }

    pub fn format(&self) -> Format {
        self.format
    }
}

impl Display for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.engine, self.format)
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Format {
    #[clap(name = "csv")]
    Csv,
    #[clap(name = "arrow")]
    Arrow,
    #[clap(name = "parquet")]
    Parquet,
    #[clap(name = "in-memory-vortex")]
    #[serde(rename = "in-memory-vortex")]
    InMemoryVortex,
    #[clap(name = "vortex")]
    #[serde(rename = "vortex")]
    OnDiskVortex,
    #[clap(name = "duckdb")]
    #[serde(rename = "duckdb")]
    OnDiskDuckDB,
}

impl Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl Format {
    pub fn name(&self) -> &'static str {
        match self {
            Format::Csv => "csv",
            Format::Arrow => "arrow",
            Format::Parquet => "parquet",
            Format::InMemoryVortex => "vortex-in-memory",
            Format::OnDiskVortex => "vortex-file-compressed",
            Format::OnDiskDuckDB => "duckdb",
        }
    }
}

#[derive(ValueEnum, Clone, Copy, Debug, Hash, Default, PartialEq, Eq, Serialize)]
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
            Engine::DataFusion => write!(f, "DataFusion"),
            Engine::DuckDB => write!(f, "DuckDB"),
            Engine::Vortex => write!(f, "Vortex"),
            Engine::Arrow => write!(f, "Arrow"),
        }
    }
}

pub use utils::file_utils::*;
pub use utils::logging::*;
use vortex::error::{VortexUnwrap, vortex_err};
