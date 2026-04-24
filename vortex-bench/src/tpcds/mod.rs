// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::path::Path;

pub mod duckdb;
pub mod schema;
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
