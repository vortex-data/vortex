// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TPC-DS benchmark implementation

use anyhow::Result;
use anyhow::anyhow;
use glob::Pattern;
use tracing::info;
use url::Url;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::Format;
use crate::IdempotentPath;
use crate::TableSpec;
use crate::tpcds::tpcds_queries;
use crate::tpcds::tpcdsgen::generate_tpcds;
use crate::utils::split;

/// TPC-DS benchmark implementation
pub struct TpcDsBenchmark {
    pub scale_factor: String,
    /// Whether the fact tables are clustered by their date and item keys. Clustered data lives in
    /// its own `{scale_factor}-clustered` directory, next to the unclustered data.
    pub clustered: bool,
    pub data_url: Url,
}

impl TpcDsBenchmark {
    pub fn new(
        scale_factor: String,
        use_remote_data_dir: Option<String>,
        clustered: bool,
    ) -> Result<Self> {
        Ok(Self {
            data_url: Self::create_data_url(
                use_remote_data_dir.as_deref(),
                &scale_factor,
                clustered,
            )?,
            scale_factor,
            clustered,
        })
    }

    fn create_data_url(
        remote_data_dir: Option<&str>,
        scale_factor: &str,
        clustered: bool,
    ) -> Result<Url> {
        match remote_data_dir {
            None => {
                let data_dir = "tpcds".to_data_path();
                let data_dir_with_sf = if clustered {
                    data_dir.join(format!("{scale_factor}-clustered"))
                } else {
                    data_dir.join(scale_factor)
                };
                Url::from_directory_path(&data_dir_with_sf).map_err(|_| {
                    anyhow!(
                        "Failed to create URL from directory path: {:?}",
                        data_dir_with_sf
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

#[async_trait::async_trait]
impl Benchmark for TpcDsBenchmark {
    fn doc_path(&self) -> &'static str {
        "vortex-bench/sql/tpcds/README.md"
    }

    fn queries(&self) -> Result<Vec<(usize, String)>> {
        Ok(tpcds_queries().collect())
    }

    async fn generate_base_data(&self) -> Result<()> {
        let base_data_dir = self
            .data_url
            .to_file_path()
            .map_err(|_| anyhow!("Invalid file URL: {}", self.data_url))?;

        info!(
            "Generating TPC-DS data with scale factor {} (clustered: {}) for format {:?}",
            self.scale_factor,
            self.clustered,
            Format::Parquet
        );

        generate_tpcds(base_data_dir, self.scale_factor.clone(), self.clustered).await?;

        Ok(())
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::TpcDS {
            scale_factor: self.scale_factor.clone(),
            clustered: self.clustered,
        }
    }

    fn dataset_name(&self) -> &str {
        "tpcds"
    }

    fn dataset_display(&self) -> String {
        self.dataset().to_string()
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }

    fn table_specs(&self) -> Vec<TableSpec> {
        vec![
            TableSpec::new("call_center", None),
            TableSpec::new("catalog_page", None),
            TableSpec::new("catalog_returns", None),
            TableSpec::new("catalog_sales", None),
            TableSpec::new("customer", None),
            TableSpec::new("customer_address", None),
            TableSpec::new("customer_demographics", None),
            TableSpec::new("date_dim", None),
            TableSpec::new("household_demographics", None),
            TableSpec::new("income_band", None),
            TableSpec::new("inventory", None),
            TableSpec::new("item", None),
            TableSpec::new("promotion", None),
            TableSpec::new("reason", None),
            TableSpec::new("ship_mode", None),
            TableSpec::new("store", None),
            TableSpec::new("store_returns", None),
            TableSpec::new("store_sales", None),
            TableSpec::new("time_dim", None),
            TableSpec::new("warehouse", None),
            TableSpec::new("web_page", None),
            TableSpec::new("web_returns", None),
            TableSpec::new("web_sales", None),
            TableSpec::new("web_site", None),
        ]
    }

    fn pattern(&self, table_name: &str, format: Format) -> Option<Pattern> {
        Some(split::part_pattern(table_name, format.ext()))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::BenchmarkArg;
    use crate::Opt;
    use crate::Opts;
    use crate::create_benchmark;
    use crate::v3::benchmark_dataset_dims;

    /// `--opt tpcds-cluster=true` selects the clustered data directory and reports a distinct
    /// dataset, while the default keeps the unclustered data.
    #[rstest]
    #[case::default(&[], "1.0", "tpcds(sf=1.0)", None)]
    #[case::unclustered(&["tpcds-cluster=false"], "1.0", "tpcds(sf=1.0)", None)]
    #[case::clustered(
        &["scale-factor=10", "tpcds-cluster=true"],
        "10-clustered",
        "tpcds(sf=10,clustered)",
        Some("clustered")
    )]
    fn test_create_benchmark_tpcds_cluster_opt(
        #[case] opts: &[&str],
        #[case] data_dir: &str,
        #[case] display: &str,
        #[case] variant: Option<&str>,
    ) -> Result<()> {
        let opts = opts
            .iter()
            .map(|opt| opt.parse::<Opt>())
            .collect::<Result<Opts>>()?;
        let benchmark = create_benchmark(BenchmarkArg::TpcDS, &opts)?;
        let expected = Url::from_directory_path("tpcds".to_data_path().join(data_dir))
            .map_err(|_| anyhow!("invalid data path"))?;
        assert_eq!(benchmark.data_url(), &expected);
        assert_eq!(benchmark.dataset_display(), display);
        let (_, dataset_variant, _) = benchmark_dataset_dims(&benchmark.dataset());
        assert_eq!(dataset_variant.as_deref(), variant);
        Ok(())
    }
}
