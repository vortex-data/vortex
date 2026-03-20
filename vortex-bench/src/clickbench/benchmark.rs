// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use anyhow::Result;
use reqwest::Client;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::BenchmarkDescriptor;
use crate::IdempotentPath;
use crate::QuerySource;
use crate::clickbench::*;
use crate::resolve_data_url;

/// ClickBench benchmark implementation
pub struct ClickBenchBenchmark {
    descriptor: BenchmarkDescriptor,
    pub flavor: Flavor,
}

impl ClickBenchBenchmark {
    pub fn new(
        flavor: Flavor,
        queries_file: Option<String>,
        use_remote_data_dir: Option<String>,
    ) -> Result<Self> {
        let data_url = resolve_data_url(
            &format!("clickbench_{flavor}"),
            use_remote_data_dir.as_deref(),
        )?;

        let queries = match queries_file {
            Some(file) => QuerySource::SemicolonDelimited(file.into()),
            None => QuerySource::sql_file("clickbench_queries.sql"),
        };

        Ok(Self {
            descriptor: BenchmarkDescriptor::new(
                "clickbench",
                data_url,
                BenchmarkDataset::ClickBench { flavor },
            )
            .with_display(format!("clickbench_{flavor}"))
            .with_queries(queries)
            .with_table("hits", Some(HITS_SCHEMA.clone()))
            .with_expected_row_counts(vec![
                1, 1, 1, 1, 1, 1, 1, 18, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 4, 1, 10, 10,
                10, 10, 10, 10, 25, 25, 1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
            ]),
            flavor,
        })
    }
}

#[async_trait::async_trait]
impl Benchmark for ClickBenchBenchmark {
    fn descriptor(&self) -> &BenchmarkDescriptor {
        &self.descriptor
    }

    async fn generate_base_data(&self) -> Result<()> {
        if self.descriptor.data_url.scheme() != "file" {
            return Ok(());
        }

        let basepath = clickbench_flavor(self.flavor).to_data_path();
        self.flavor.download(Client::default(), basepath).await?;

        Ok(())
    }
}

fn clickbench_flavor(flavor: Flavor) -> String {
    format!("clickbench_{flavor}")
}
