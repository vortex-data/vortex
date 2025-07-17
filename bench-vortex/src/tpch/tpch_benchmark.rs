// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TPCH benchmark implementation

use std::path::{Path, PathBuf};
use std::{env, fs};

use anyhow::{Result, anyhow};
use datafusion::prelude::SessionContext;
use ddb::DuckDBCtx;
use glob::Pattern;
use log::{info, warn};
use similar::{ChangeTag, TextDiff};
use url::Url;

use crate::benchmark_trait::Benchmark;
use crate::engines::{EngineCtx, ddb};
use crate::tpch::schema::{CUSTOMER, LINEITEM, NATION, ORDERS, PART, PARTSUPP, REGION, SUPPLIER};
use crate::tpch::{
    EXPECTED_ROW_COUNTS_SF1, EXPECTED_ROW_COUNTS_SF10, register_arrow, register_parquet,
    register_vortex_file, tpch_queries, tpchgen,
};
use crate::{BenchmarkDataset, Format, IdempotentPath, Target};

/// TPCH benchmark implementation
pub struct TpcHBenchmark {
    pub scale_factor: String,
    pub data_url: Url,
}

impl TpcHBenchmark {
    pub fn new(scale_factor: String, use_remote_data_dir: Option<String>) -> Result<Self> {
        Ok(Self {
            scale_factor: scale_factor.clone(),
            data_url: Self::create_data_url(&use_remote_data_dir, &scale_factor)?,
        })
    }

    fn create_data_url(remote_data_dir: &Option<String>, scale_factor: &str) -> Result<Url> {
        match remote_data_dir {
            None => {
                let data_dir = "tpch".to_data_path();
                let data_dir_with_sf = data_dir.join(scale_factor);
                Url::from_directory_path(&data_dir_with_sf).map_err(|_| {
                    anyhow!(
                        "Failed to create URL from directory path: {:?}",
                        &data_dir_with_sf
                    )
                })
            }
            Some(remote_data_dir) => {
                if !remote_data_dir.ends_with("/") {
                    warn!(
                        "Supply a --use-remote-data-dir argument which ends in a slash e.g. s3://vortex-bench-dev-eu/parquet/"
                    );
                }
                info!(
                    concat!(
                        "Assuming data already exists at this remote (e.g. S3, GCS) URL: {}.\n",
                        "If it does not, you should kill this command, locally generate the files (by running without\n",
                        "--use-remote-data-dir) and upload data/tpch/{}/ to some remote location.",
                    ),
                    remote_data_dir, scale_factor,
                );
                Ok(Url::parse(remote_data_dir)?)
            }
        }
    }
}

impl Benchmark for TpcHBenchmark {
    fn queries(&self) -> Result<Vec<(usize, String)>> {
        Ok(tpch_queries().collect())
    }

    fn generate_data(&self, target: &Target) -> Result<()> {
        match self.data_url.scheme() {
            "file" => {
                // Generate data for the specific target format (idempotent)
                let format = if target.format() == Format::Arrow {
                    // For Arrow format, we need Parquet files to load into memory
                    Format::Parquet
                } else {
                    target.format()
                };

                // Skip CSV generation as it's not supported by tpchgen
                if format == Format::Csv {
                    anyhow::bail!("CSV format is not supported by tpchgen");
                }

                let base_data_dir = self
                    .data_url
                    .to_file_path()
                    .map_err(|_| anyhow!("Invalid file URL: {}", self.data_url))?;

                // Use tpchgen for data generation
                let options =
                    tpchgen::TpchGenOptions::new(self.scale_factor.clone(), &base_data_dir)
                        .with_format(format)
                        .with_max_file_size_mb(Some(600));

                // Generate data using our streaming tpchgen module
                let runtime = tokio::runtime::Runtime::new()?;
                runtime.block_on(async { tpchgen::generate_tpch_tables(&options).await })?;

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
    /// Register TPCH tables with DataFusion session - extracted from load_datasets
    async fn register_tpch_tables(
        &self,
        session: &SessionContext,
        base_dir: &Url,
        format: Format,
    ) -> Result<()> {
        let files = vec![
            ("customer", Some(CUSTOMER.clone())),
            ("lineitem", Some(LINEITEM.clone())),
            ("nation", Some(NATION.clone())),
            ("orders", Some(ORDERS.clone())),
            ("part", Some(PART.clone())),
            ("partsupp", Some(PARTSUPP.clone())),
            ("region", Some(REGION.clone())),
            ("supplier", Some(SUPPLIER.clone())),
        ];

        for (name, schema) in files {
            let file_format = if format == Format::Arrow {
                // Arrow format loads Parquet files into memory
                Format::Parquet
            } else {
                format
            };

            let path = base_dir.join(&(file_format.name().to_string() + "/"))?;
            let glob = Some(Pattern::new(&format!("{name}_*.{}", file_format.ext()))?);

            match format {
                Format::Arrow => register_arrow(session, name, &path, glob).await?,
                Format::Parquet => {
                    register_parquet(session, name, &path, glob, schema, &self.dataset()).await?
                }
                Format::OnDiskVortex => {
                    register_vortex_file(session, name, &path, glob, schema, &self.dataset())
                        .await?
                }
                Format::OnDiskDuckDB => unreachable!("duckdb never supported with datafusion"),
                Format::Csv => todo!("csv unsupported for tpch benchmark"),
            }
        }

        Ok(())
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

            duckdb_ctx.execute_query(&create_table)?;
            duckdb_ctx.execute_query(&write_csv)?;

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
