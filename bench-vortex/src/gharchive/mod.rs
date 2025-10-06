// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! GitHub Archive dataset.

use url::Url;

use crate::benchmark_trait::Benchmark;
use crate::engines::EngineCtx;
use crate::{BenchmarkDataset, Format, Target};

fn github_archive_urls() -> impl Iterator<Item = String> {
    (1..=24).map(|hour| format!("https://data.gharchive.org/2025-10-05-{hour}.json.gz"))
}

pub struct GitHubArchive {
    data_url: Url,
}

impl Benchmark for GitHubArchive {
    fn queries(&self) -> anyhow::Result<Vec<(usize, String)>> {
        todo!()
    }

    fn generate_data(&self, target: &Target) -> anyhow::Result<()> {
        anyhow::ensure!(
            matches!(
                target.format,
                Format::Parquet | Format::OnDiskVortex | Format::VortexCompact
            ),
            "unsupported format for `fineweb` bench: {}",
            target.format
        );

        todo!()
    }

    async fn register_tables(&self, engine_ctx: &EngineCtx, format: Format) -> anyhow::Result<()> {
        todo!()
    }

    fn dataset(&self) -> BenchmarkDataset {
        todo!()
    }

    fn dataset_name(&self) -> &str {
        todo!()
    }

    fn dataset_display(&self) -> String {
        todo!()
    }

    fn data_url(&self) -> &Url {
        todo!()
    }
}
