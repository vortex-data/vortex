// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! SQLStorm `Benchmark` implementation, parameterized by origin.

use anyhow::Result;
use anyhow::anyhow;
use glob::Pattern;
use url::Url;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::Format;
use crate::IdempotentPath;
use crate::TableSpec;
use crate::sqlstorm::SqlstormOrigin;
use crate::sqlstorm::data;
use crate::sqlstorm::sqlstorm_queries;
use crate::tpcds::TpcDsBenchmark;
use crate::tpch::benchmark::TpcHBenchmark;

/// SQLStorm benchmark over one origin's vendored query sample.
pub struct SqlstormBenchmark {
    origin: SqlstormOrigin,
    data_url: Url,
}

impl SqlstormBenchmark {
    /// Create a benchmark for `origin`, resolving its data directory (or a remote override).
    pub fn new(origin: SqlstormOrigin, use_remote_data_dir: Option<String>) -> Result<Self> {
        Ok(Self {
            origin,
            data_url: Self::create_data_url(use_remote_data_dir.as_deref(), origin)?,
        })
    }

    /// Resolve the base data URL for `origin`.
    ///
    /// TPC-H and TPC-DS reuse the existing local datasets at the default scale-factor
    /// directory (`DEFAULT_SCALE_FACTOR` = `"1.0"` in `lib.rs`, so both live under
    /// `<dataset>/1.0`). StackOverflow and JOB get their own `sqlstorm-<origin>` directories.
    fn create_data_url(remote_data_dir: Option<&str>, origin: SqlstormOrigin) -> Result<Url> {
        if let Some(remote) = remote_data_dir {
            let mut url = Url::parse(remote)?;
            if !url.path().ends_with('/') {
                url.set_path(&format!("{}/", url.path()));
            }
            return Ok(url);
        }
        let dir = match origin {
            SqlstormOrigin::TpcH => "tpch".to_data_path().join("1.0"),
            SqlstormOrigin::TpcDs => "tpcds".to_data_path().join("1.0"),
            SqlstormOrigin::StackOverflow => "sqlstorm-stackoverflow".to_data_path(),
            SqlstormOrigin::Job => "sqlstorm-job".to_data_path(),
        };
        Url::from_directory_path(&dir)
            .map_err(|_| anyhow!("Failed to create URL from directory path: {:?}", dir))
    }
}

#[async_trait::async_trait]
impl Benchmark for SqlstormBenchmark {
    fn queries(&self) -> Result<Vec<(usize, String)>> {
        sqlstorm_queries(self.origin)
    }

    async fn generate_base_data(&self) -> Result<()> {
        match self.origin {
            SqlstormOrigin::TpcH => {
                TpcHBenchmark::new("1.0".to_string(), None)?
                    .generate_base_data()
                    .await
            }
            SqlstormOrigin::TpcDs => {
                TpcDsBenchmark::new("1.0".to_string(), None)?
                    .generate_base_data()
                    .await
            }
            SqlstormOrigin::StackOverflow => data::generate_stackoverflow(&self.data_url).await,
            SqlstormOrigin::Job => data::generate_job(&self.data_url).await,
        }
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::Sqlstorm {
            origin: self.origin.name().to_string(),
        }
    }

    fn dataset_name(&self) -> &str {
        "sqlstorm"
    }

    fn dataset_display(&self) -> String {
        format!("sqlstorm({})", self.origin.name())
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }

    fn table_specs(&self) -> Vec<TableSpec> {
        data::table_specs(self.origin)
    }

    #[expect(clippy::expect_used)]
    fn pattern(&self, table_name: &str, format: Format) -> Option<Pattern> {
        // Match each origin's on-disk layout: the reused TPC-H dataset shards large
        // tables as `<table>_<n>.parquet` (mirroring `TpcHBenchmark`), while TPC-DS and
        // our single-file StackOverflow/JOB exports use `<table>.<ext>`.
        let glob = match self.origin {
            SqlstormOrigin::TpcH => format!("{}_*.{}", table_name, format.ext()),
            _ => format!("{}.{}", table_name, format.ext()),
        };
        Some(glob.parse().expect("valid glob pattern"))
    }
}
