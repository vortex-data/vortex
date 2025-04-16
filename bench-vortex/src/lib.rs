#![feature(exit_status_error)]

use std::clone::Clone;
use std::fmt::Display;

use clap::ValueEnum;

pub mod bench_run;
pub mod blob;
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

#[macro_export]
macro_rules! feature_flagged_allocator {
    () => {
        cfg_if::cfg_if! {
            if #[cfg(feature = "mimalloc")] {
                #[global_allocator]
                static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
            } else if #[cfg(feature = "jemalloc")] {
                #[global_allocator]
                static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
            }
        }
    };
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, ValueEnum)]
pub enum Format {
    #[clap(name = "csv")]
    Csv,
    #[clap(name = "arrow")]
    Arrow,
    #[clap(name = "parquet")]
    Parquet,
    #[clap(name = "in-memory-vortex")]
    InMemoryVortex,
    #[clap(name = "vortex")]
    OnDiskVortex,
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
        }
    }
}

#[derive(ValueEnum, Clone, Copy, Debug, Hash, Default, PartialEq, Eq)]
pub enum Engine {
    #[default]
    Vortex,
    #[clap(name = "datafusion")]
    DataFusion,
    #[clap(name = "duckdb")]
    DuckDB,
}

impl std::fmt::Display for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Engine::DataFusion => write!(f, "DataFusion"),
            Engine::DuckDB => write!(f, "DuckDB"),
            Engine::Vortex => write!(f, "Vortex"),
        }
    }
}

pub use utils::file_utils::*;
pub use utils::logging::*;
