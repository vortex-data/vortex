// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::env;
use std::path::Path;

use anyhow::Result;
use tokio::runtime::Runtime;
use url::Url;
use vortex::error::VortexExpect;

use crate::benchmark_trait::Benchmark;
use crate::clickbench::*;
use crate::engines::EngineCtx;
use crate::{BenchmarkDataset, CompactionStrategy, Format, IdempotentPath, Target};

/// ClickBench benchmark implementation
pub struct ClickBenchBenchmark {
    pub flavor: Flavor,
    pub queries_file: Option<String>,
    pub data_url: Url,
}

impl ClickBenchBenchmark {
    pub fn new(
        flavor: Flavor,
        queries_file: Option<String>,
        use_remote_data_dir: Option<String>,
    ) -> Result<Self> {
        let url = Self::create_data_url(&use_remote_data_dir, flavor)?;
        Ok(Self {
            flavor,
            queries_file,
            data_url: url,
        })
    }

    fn create_data_url(remote_data_dir: &Option<String>, flavor: Flavor) -> Result<Url> {
        match remote_data_dir {
            None => {
                let basepath = format!("clickbench_{flavor}").to_data_path();
                Ok(Url::parse(&format!(
                    "file:{}/",
                    basepath.to_str().vortex_expect("path should be utf8")
                ))?)
            }
            Some(remote_data_dir) => {
                if !remote_data_dir.ends_with("/") {
                    log::warn!(
                        "Supply a --use-remote-data-dir argument which ends in a slash e.g. s3://vortex-bench-dev-eu/parquet/"
                    );
                }
                log::info!(
                    concat!(
                        "Assuming data already exists at this remote (e.g. S3, GCS) URL: {}.\\n",
                        "If it does not, you should kill this command, locally generate the files (by running without\\n",
                        "--use-remote-data-dir) and upload data/clickbench/ to some remote location.",
                    ),
                    remote_data_dir,
                );
                Ok(Url::parse(remote_data_dir)?)
            }
        }
    }
}

impl Benchmark for ClickBenchBenchmark {
    fn queries(&self) -> Result<Vec<(usize, String)>> {
        let queries_filepath = match &self.queries_file {
            Some(file) => file.into(),
            None => Path::new(env!("CARGO_MANIFEST_DIR")).join("clickbench_queries.sql"),
        };

        Ok(clickbench_queries(queries_filepath))
    }

    fn generate_data(&self, target: &Target) -> Result<()> {
        match self.data_url.scheme() {
            "file" => {
                let basepath = clickbench_flavor(self.flavor).to_data_path();
                let client = reqwest::blocking::Client::default();

                match target.format() {
                    Format::Parquet | Format::OnDiskDuckDB => {
                        // Download Parquet files (idempotent - won't re-download if already present)
                        // For DuckDB format, we typically start with Parquet and let DuckDB handle it
                        self.flavor.download(&client, basepath.as_path())?;
                    }
                    Format::OnDiskVortex | Format::VortexCompact => {
                        // First ensure Parquet files exist
                        self.flavor.download(&client, basepath.as_path())?;

                        // Then convert to Vortex format (idempotent)
                        if self.data_url.scheme() == "file" {
                            let file_path = self.data_url.to_file_path().map_err(|_| {
                                anyhow::anyhow!("invalid file URL: {}", self.data_url)
                            })?;

                            // Use tokio runtime to handle async conversion
                            let rt = Runtime::new()?;
                            rt.block_on(async {
                                match target.format {
                                    Format::OnDiskVortex => {
                                        convert_parquet_to_vortex(
                                            &file_path,
                                            CompactionStrategy::Default,
                                        )
                                        .await
                                    }
                                    Format::VortexCompact => {
                                        convert_parquet_to_vortex(
                                            &file_path,
                                            CompactionStrategy::Compact,
                                        )
                                        .await
                                    }
                                    _ => unreachable!(),
                                }
                            })?
                        }
                    }
                    Format::Lance => {
                        // Lance manages its own partitioning internally, so flavor doesn't matter.
                        if self.flavor == Flavor::Single {
                            eprintln!(
                                "Note: Lance manages its own internal partitioning. There is no \
                                difference between Single and Partitioned flavors for Lance format."
                            );
                        }

                        // Download Parquet files (either Single or Partitioned).
                        self.flavor.download(&client, basepath.as_path())?;

                        // Then convert to Lance format (idempotent).
                        if self.data_url.scheme() == "file" {
                            let file_path = self.data_url.to_file_path().map_err(|_| {
                                anyhow::anyhow!("invalid file URL: {}", self.data_url)
                            })?;

                            let rt = Runtime::new()?;
                            rt.block_on(async { convert_parquet_to_lance(&file_path).await })?
                        }
                    }
                    f => {
                        todo!("format {f} unsupported in clickbench")
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
                dataset
                    .register_tables(&ctx.session, &self.data_url, format)
                    .await?;
            }
            EngineCtx::DuckDB(ctx) => {
                ctx.register_tables(&self.data_url, format, &dataset)?;
            }
        }

        Ok(())
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::ClickBench {
            flavor: self.flavor,
        }
    }

    fn expected_row_counts(&self) -> Option<&[usize]> {
        // ClickBench reference row counts
        static REFERENCE_ROW_COUNTS: [usize; 43] = [
            1, 1, 1, 1, 1, 1, 1, 18, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 4, 1, 10, 10, 10,
            10, 10, 10, 25, 25, 1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
        ];
        Some(&REFERENCE_ROW_COUNTS)
    }

    // Dataset-specific methods (inlined from BenchmarkDataset)

    fn dataset_name(&self) -> &str {
        "clickbench"
    }

    fn dataset_display(&self) -> String {
        format!("clickbench_{}", self.flavor)
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }
}

fn clickbench_flavor(flavor: Flavor) -> String {
    format!("clickbench_{flavor}")
}
