// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::path::Path;

use datafusion::prelude::SessionContext;
use itertools::Itertools;
use url::Url;

use crate::df::get_session_context;
use crate::tpch::{
    register_arrow, register_parquet, register_vortex_compact_file, register_vortex_file,
};
use crate::{BenchmarkDataset, Format};

pub mod tpcds_benchmark;

pub use tpcds_benchmark::TpcDsBenchmark;

pub fn tpcds_queries() -> impl Iterator<Item = (usize, String)> {
    (1..=99).map(|idx| (idx, tpcds_query(idx)))
}

// A few tpcds queries have multiple statements, this handles that
fn tpcds_query(query_idx: usize) -> String {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tpcds")
        .join(format!("{query_idx:02}"))
        .with_extension("sql");
    fs::read_to_string(manifest_dir).unwrap()
}

/// Generate table dataset.
pub async fn load_datasets(
    base_dir: &Url,
    format: Format,
    dataset: &BenchmarkDataset,
    disable_datafusion_cache: bool,
) -> anyhow::Result<SessionContext> {
    let context = get_session_context(disable_datafusion_cache);

    let files = match dataset {
        dataset @ BenchmarkDataset::TpcDS { .. } => {
            dataset.tables().iter().map(|f| (*f, None)).collect_vec()
        }
        _ => todo!(),
    };

    for (name, path, schema) in files.into_iter().map(|(name, schema)| {
        let format = if format == Format::Arrow {
            Format::Parquet
        } else {
            format
        };
        (
            name,
            base_dir.join(&format!("{}/{name}.{}", format.name(), format.ext())),
            schema,
        )
    }) {
        let path = path?;
        match format {
            Format::Arrow => register_arrow(&context, name, &path, None).await?,
            Format::Parquet => {
                register_parquet(&context, name, &path, None, schema, dataset).await?
            }
            Format::OnDiskVortex => {
                register_vortex_file(&context, name, &path, None, schema, dataset).await?
            }
            Format::VortexCompact => {
                register_vortex_compact_file(&context, name, &path, None, schema, dataset).await?
            }
            Format::OnDiskDuckDB => unreachable!("duckdb never supported with datafusion"),
            Format::Csv => todo!(),
        }
    }

    Ok(context)
}
