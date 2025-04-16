use std::fmt::Display;

use anyhow::Result;
use async_trait::async_trait;
use datafusion::prelude::SessionContext;
use url::Url;
use vortex::ArrayRef;

pub mod data_downloads;
pub mod file;
pub mod struct_list_of_ints;
pub mod taxi_data;
pub mod tpch_l_comment;

#[async_trait]
pub trait Dataset {
    fn name(&self) -> &str;

    async fn to_vortex_array(&self) -> ArrayRef;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkDataset {
    TpcH,
    ClickBench { single_file: bool },
}

impl Display for BenchmarkDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BenchmarkDataset::TpcH => write!(f, "tpch"),
            BenchmarkDataset::ClickBench { single_file } => {
                if *single_file {
                    write!(f, "clickbench-single")
                } else {
                    write!(f, "clickbench-partitioned")
                }
            }
        }
    }
}

impl BenchmarkDataset {
    pub fn parquet_path(&self, base_url: &Url) -> Result<Url> {
        match self {
            BenchmarkDataset::TpcH => {
                // TPC-H parquet files are stored alongside the TBL files
                Ok(base_url.clone())
            }
            BenchmarkDataset::ClickBench { .. } => {
                // ClickBench parquet files are stored in "parquet/" subdirectory
                Ok(base_url.join("parquet/")?)
            }
        }
    }

    pub fn vortex_path(&self, base_url: &Url) -> Result<Url> {
        match self {
            BenchmarkDataset::TpcH => {
                // TPC-H vortex files are stored in "vortex_compressed/" subdirectory
                let vortex_dir_path = format!("{}vortex_compressed/", base_url.path());
                let mut vortex_dir = base_url.clone();
                vortex_dir.set_path(&vortex_dir_path);
                Ok(vortex_dir)
            }
            BenchmarkDataset::ClickBench { .. } => {
                // ClickBench vortex files are stored in "vortex/" subdirectory
                Ok(base_url.join("vortex/")?)
            }
        }
    }

    pub fn register_tables(
        &self,
        session: &SessionContext,
        base_url: &Url,
        format: crate::Format,
    ) -> Result<()> {
        // Register tables synchronously to avoid nested runtime issues
        match (self, format) {
            (BenchmarkDataset::TpcH, _) => {
                // TPC-H tables are handled separately
            }
            (BenchmarkDataset::ClickBench { single_file }, crate::Format::Parquet) => {
                crate::clickbench::register_parquet_files(
                    session,
                    "hits",
                    base_url,
                    &crate::clickbench::HITS_SCHEMA,
                    *single_file,
                )?;
            }
            (BenchmarkDataset::ClickBench { single_file }, crate::Format::OnDiskVortex) => {
                crate::clickbench::register_vortex_files(
                    session.clone(),
                    "hits",
                    base_url,
                    &crate::clickbench::HITS_SCHEMA,
                    *single_file,
                )?;
            }
            (BenchmarkDataset::ClickBench { .. }, _) => {
                anyhow::bail!("Unsupported format for ClickBench: {}", format);
            }
        }

        Ok(())
    }
}
