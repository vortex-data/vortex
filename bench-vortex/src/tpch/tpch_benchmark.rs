// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TPCH benchmark implementation

use std::path::{Path, PathBuf};
use std::{env, fs};

use anyhow::{Result, anyhow};
use datafusion::prelude::SessionContext;
use ddb::DuckDBCtx;
use similar::{ChangeTag, TextDiff};
use tokio::runtime::Runtime;
use url::Url;

use crate::benchmark_trait::Benchmark;
use crate::datasets::configs::TpcHDataset;
use crate::datasets::unified_registration::register_dataset_tables;
use crate::engines::{EngineCtx, ddb};
use crate::helpers::urls::{benchmark_data_url, url_to_path};
use crate::tpch::tpchgen::TpchGenOptions;
use crate::tpch::{EXPECTED_ROW_COUNTS_SF1, EXPECTED_ROW_COUNTS_SF10, tpch_queries, tpchgen};
use crate::{BenchmarkDataset, Format, Target};

/// TPCH benchmark implementation
pub struct TpcHBenchmark {
    pub scale_factor: String,
    pub data_url: Url,
}

impl TpcHBenchmark {
    pub fn new(scale_factor: String, remote_data_dir: Option<String>) -> Result<Self> {
        Ok(Self {
            scale_factor: scale_factor.clone(),
            data_url: benchmark_data_url("tpch", Some(&scale_factor), &remote_data_dir)?,
        })
    }
}

impl Benchmark for TpcHBenchmark {
    fn queries(&self) -> Result<Vec<(usize, String)>> {
        Ok(tpch_queries().collect())
    }

    fn generate_data(&self, target: &Target) -> Result<()> {
        match self.data_url.scheme() {
            "file" => {
                let base_data_dir = url_to_path(&self.data_url)?;

                match target.format() {
                    #[cfg(feature = "lance")]
                    Format::Lance => {
                        // For Lance: first generate Parquet, then convert using our converter
                        let options =
                            TpchGenOptions::new(self.scale_factor.clone(), &base_data_dir)
                                .with_format(Format::Parquet)
                                .with_max_file_size_mb(Some(600));

                        let runtime = Runtime::new()?;
                        runtime.block_on(async {
                            // Generate Parquet
                            tpchgen::generate_tpch_tables(options).await?;

                            // Use our format converter infrastructure
                            let parquet_dir = base_data_dir.join("parquet");
                            let lance_dir = base_data_dir.join("lance");

                            use crate::conversion::{ConversionOptions, convert_format};
                            convert_format(
                                &parquet_dir,
                                &lance_dir,
                                Format::Parquet,
                                Format::Lance,
                                ConversionOptions::default(),
                            )
                            .await
                        })?;
                    }
                    Format::Arrow => {
                        // For Arrow: load Parquet files into memory
                        let options =
                            TpchGenOptions::new(self.scale_factor.clone(), &base_data_dir)
                                .with_format(Format::Parquet)
                                .with_max_file_size_mb(Some(600));

                        let runtime = Runtime::new()?;
                        runtime.block_on(async { tpchgen::generate_tpch_tables(options).await })?;
                    }
                    Format::Csv => {
                        anyhow::bail!("CSV format is not supported by tpchgen");
                    }
                    _ => {
                        // Other formats generate directly
                        let options =
                            TpchGenOptions::new(self.scale_factor.clone(), &base_data_dir)
                                .with_format(target.format())
                                .with_max_file_size_mb(Some(600));

                        let runtime = Runtime::new()?;
                        runtime.block_on(async { tpchgen::generate_tpch_tables(options).await })?;
                    }
                }

                Ok(())
            }
            _ => Ok(()),
        }
    }

    #[allow(async_fn_in_trait)]
    async fn register_tables(&self, engine_ctx: &EngineCtx, format: Format) -> Result<()> {
        let dataset = self.dataset();

        match engine_ctx {
            EngineCtx::DataFusion(ctx) => {
                // Register TPCH tables using the same logic as load_datasets
                self.register_tpch_tables(&ctx.session, &self.data_url, format)
                    .await
            }
            EngineCtx::DuckDB(ctx) => {
                ctx.register_tables(&self.data_url, format, &dataset)?;
                Ok(())
            }
        }
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::TpcH {
            scale_factor: self.scale_factor.clone(),
        }
    }

    fn expected_row_counts(&self) -> Option<&[usize]> {
        match self.scale_factor.as_str() {
            "1.0" => Some(&EXPECTED_ROW_COUNTS_SF1),
            "10.0" => Some(&EXPECTED_ROW_COUNTS_SF10),
            _ => None, // Unsupported scale factor
        }
    }

    fn dataset_name(&self) -> &str {
        "tpch"
    }

    fn tables(&self) -> &[&'static str] {
        &[
            "customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier",
        ]
    }

    fn dataset_display(&self) -> String {
        format!("tpch(sf={})", self.scale_factor)
    }

    fn validate_result(&self, queries: Vec<usize>) -> Result<()> {
        self.verify_duckdb_tpch_results(queries)
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }
}

impl TpcHBenchmark {
    /// Register TPCH tables with DataFusion session using unified registration
    async fn register_tpch_tables(
        &self,
        session: &SessionContext,
        base_dir: &Url,
        format: Format,
    ) -> Result<()> {
        let dataset = TpcHDataset {
            scale_factor: self.scale_factor.clone(),
        };

        register_dataset_tables(session, &dataset, base_dir, format).await
    }

    /// Verify DuckDB TPCH results against reference data
    pub fn verify_duckdb_tpch_results(&self, queries: Vec<usize>) -> Result<()> {
        // omit validation for sf != 1.
        if self.scale_factor != "1.0" {
            return Ok(());
        }
        let query_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../vortex-duckdb/duckdb/extension/tpch/dbgen/queries");

        let tmp_dir = format!(
            "{}/spiral-tpch",
            // $RUNNER_TEMP is defined by GitHub Actions.
            env::var("TMPDIR").or_else(|_| env::var("RUNNER_TEMP"))?
        );

        if PathBuf::from(&tmp_dir).exists() {
            fs::remove_dir_all(&tmp_dir)?;
        }
        fs::create_dir(&tmp_dir)?;
        let duckdb_ctx = DuckDBCtx::new_in_memory()?;
        duckdb_ctx.register_tables(
            self.data_url(),
            Format::OnDiskVortex,
            &BenchmarkDataset::TpcH {
                scale_factor: self.scale_factor.clone(),
            },
        )?;

        let mut query_files = fs::read_dir(query_dir)?
            .filter_map(Result::ok)
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "sql"))
            .collect::<Vec<_>>();
        query_files.sort_by_key(|entry| entry.file_name());

        let mut is_matching_ref_result = true;

        for query_file in query_files
            .iter()
            .enumerate()
            .filter(|entry| queries.contains(&(entry.0 + 1)))
            .map(|query_file| query_file.1)
        {
            let query_file_path = query_file.path();
            let query_name = query_file_path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .ok_or_else(|| anyhow!("Invalid query filename"))?;

            let create_table = format!(
                "CREATE OR REPLACE TABLE {query_name}_result AS {};",
                fs::read_to_string(&query_file_path)?
            );

            let csv_actual = format!("{tmp_dir}/{query_name}.csv");
            let write_csv =
                format!("COPY {query_name}_result TO '{csv_actual}' (HEADER, DELIMITER '|');",);

            duckdb_ctx.execute_query_internal(&create_table)?;
            duckdb_ctx.execute_query_internal(&write_csv)?;

            let csv_expected = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("tpch/results/duckdb/{query_name}.csv"));
            let expected = fs::read_to_string(csv_expected)?;
            let actual = fs::read_to_string(csv_actual)?;

            if expected != actual {
                let diff = TextDiff::from_lines(&expected, &actual);

                for change in diff.iter_all_changes() {
                    let sign = match change.tag() {
                        ChangeTag::Delete => "-",
                        ChangeTag::Insert => "+",
                        ChangeTag::Equal => " ",
                    };
                    print!("{sign}{change}");
                }

                eprintln!("query output does not match the reference for {query_name}");
                is_matching_ref_result = false;
            }
        }

        if !is_matching_ref_result {
            return Err(anyhow!("not all queries matched the reference"));
        }

        Ok(())
    }
}
