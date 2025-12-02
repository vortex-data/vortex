// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Test binary for reading benchmark entries from S3 and printing them.

#![allow(clippy::expect_used)]

use vortex::VortexSessionDefault;
use vortex::session::VortexSession;
use vortex_wasm::website::names::NAMES;
use vortex_wasm::website::read_s3::read_benchmark_entries;

const KEY: &str = "test/random_access.vortex";

fn main() {
    let runtime = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
    runtime.block_on(async_main());
}

async fn async_main() {
    let session = VortexSession::default();

    println!("Reading benchmark entries from {}...\n", KEY);

    let entries = read_benchmark_entries(&session, KEY)
        .await
        .expect("Failed to read benchmark entries");

    println!("Read {} entries:\n", entries.len());

    // Print header.
    println!(
        "{:<44} {:>15} {:>15} {:>15} {:>12}",
        "commit_id", "benchmark_group", "chart_name", "series_name", "value"
    );
    println!("{}", "-".repeat(105));

    // Print entries (limit to first 20 and last 5 for readability).
    let show_first = 20;
    let show_last = 5;

    for (i, entry) in entries.iter().enumerate() {
        if i < show_first || i >= entries.len() - show_last {
            let benchmark_group = NAMES.get(&entry.benchmark_group.0).unwrap_or(&"unknown");
            let chart_name = NAMES.get(&entry.chart_name.0).unwrap_or(&"unknown");
            let series_name = NAMES.get(&entry.series_name.0).unwrap_or(&"unknown");

            println!(
                "{} {:>15} {:>15} {:>15} {:>12}",
                entry.commit_id, benchmark_group, chart_name, series_name, entry.value
            );
        } else if i == show_first {
            println!(
                "... ({} more entries) ...",
                entries.len() - show_first - show_last
            );
        }
    }

    println!("\nTotal: {} entries", entries.len());
}
