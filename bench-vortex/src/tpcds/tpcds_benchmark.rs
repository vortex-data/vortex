// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TPC-DS benchmark implementation

use anyhow::Result;
use anyhow::anyhow;
use datafusion::prelude::SessionContext;
use log::info;
use url::Url;

use crate::BenchmarkDataset;
use crate::Format;
use crate::IdempotentPath;
use crate::Target;
use crate::benchmark_trait::Benchmark;
use crate::engines::EngineCtx;
use crate::tpcds::duckdb::generate_tpcds;
use crate::tpcds::tpcds_queries;

/// TPC-DS benchmark implementation
pub struct TpcDsBenchmark {
    pub scale_factor: String,
    pub data_url: Url,
}

impl TpcDsBenchmark {
    pub fn new(scale_factor: String, use_remote_data_dir: Option<String>) -> Result<Self> {
        Ok(Self {
            scale_factor: scale_factor.clone(),
            data_url: Self::create_data_url(&use_remote_data_dir, &scale_factor)?,
        })
    }

    fn create_data_url(remote_data_dir: &Option<String>, scale_factor: &str) -> Result<Url> {
        match remote_data_dir {
            None => {
                let data_dir = "tpcds".to_data_path();
                let data_dir_with_sf = data_dir.join(scale_factor);
                Url::from_directory_path(&data_dir_with_sf).map_err(|_| {
                    anyhow!(
                        "Failed to create URL from directory path: {:?}",
                        &data_dir_with_sf
                    )
                })
            }
            Some(remote_data_dir) => {
                let mut url = Url::parse(remote_data_dir)?;
                if !url.path().ends_with('/') {
                    url.set_path(&format!("{}/", url.path()));
                }
                Ok(url)
            }
        }
    }
}

impl Benchmark for TpcDsBenchmark {
    fn queries(&self) -> Result<Vec<(usize, String)>> {
        Ok(tpcds_queries().collect())
    }

    fn generate_data(&self, target: &Target) -> Result<()> {
        // Use connection-based TPC-DS data generation
        let base_data_dir = self
            .data_url
            .to_file_path()
            .map_err(|_| anyhow!("Invalid file URL: {}", self.data_url))?;

        info!(
            "Generating TPC-DS data with scale factor {} for format {:?}",
            self.scale_factor,
            target.format()
        );

        generate_tpcds(base_data_dir, self.scale_factor.clone(), target.format())?;
        Ok(())
    }

    async fn register_tables(&self, engine_ctx: &EngineCtx, format: Format) -> Result<()> {
        match engine_ctx {
            EngineCtx::DataFusion(ctx) => {
                // Register TPC-DS tables similar to how TPC-H does it
                self.register_tpcds_tables(&ctx.session, &self.data_url, format)
                    .await
            }
            EngineCtx::DuckDB(ctx) => {
                ctx.register_tables(&self.data_url, format, &self.dataset())?;
                Ok(())
            }
        }
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::TpcDS {
            scale_factor: self.scale_factor.clone(),
        }
    }

    fn dataset_name(&self) -> &str {
        "tpcds"
    }

    fn dataset_display(&self) -> String {
        format!("tpcds(sf={})", self.scale_factor)
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }

    fn expected_row_counts(&self) -> Option<&[usize]> {
        None
    }

    fn tables(&self) -> &[&'static str] {
        // TPC-DS has 24 tables
        &[
            "call_center",
            "catalog_page",
            "catalog_returns",
            "catalog_sales",
            "customer",
            "customer_address",
            "customer_demographics",
            "date_dim",
            "household_demographics",
            "income_band",
            "inventory",
            "item",
            "promotion",
            "reason",
            "ship_mode",
            "store",
            "store_returns",
            "store_sales",
            "time_dim",
            "warehouse",
            "web_page",
            "web_returns",
            "web_sales",
            "web_site",
        ]
    }

    fn validate_result(&self, _queries: Vec<usize>) -> Result<()> {
        Ok(())
    }
}

impl TpcDsBenchmark {
    async fn register_tpcds_tables(
        &self,
        session: &SessionContext,
        base_dir: &Url,
        format: Format,
    ) -> Result<()> {
        use crate::tpch::register_arrow;
        use crate::tpch::register_parquet;
        use crate::tpch::register_vortex_compact_file;
        use crate::tpch::register_vortex_file;

        let dataset = self.dataset();
        let files = dataset
            .tables()
            .iter()
            .map(|f| (*f, None))
            .collect::<Vec<_>>();

        // For TPC-DS, files are stored in a subdirectory named after the format
        let format_dir = base_dir.join(&format!("{}/", format.name()))?;

        for (name, schema) in files {
            let format = if format == Format::Arrow {
                Format::Parquet
            } else {
                format
            };

            let path = format_dir.join(&format!("{name}.{}", format.ext()))?;

            match format {
                Format::Arrow => register_arrow(session, name, &path, None).await?,
                Format::Parquet => {
                    register_parquet(session, name, &path, None, schema, &dataset).await?
                }
                Format::OnDiskVortex => {
                    register_vortex_file(session, name, &path, None, schema, &dataset).await?
                }
                Format::VortexCompact => {
                    register_vortex_compact_file(session, name, &path, None, schema, &dataset)
                        .await?
                }
                Format::OnDiskDuckDB => unreachable!("duckdb never supported with datafusion"),
                Format::Csv => todo!(),
                #[cfg(feature = "lance")]
                Format::Lance => unimplemented!(),
            }
        }

        Ok(())
    }
}
