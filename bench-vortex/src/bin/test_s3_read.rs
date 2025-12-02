// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Test binary for reading benchmark entries from S3 and printing them.

#![allow(clippy::expect_used)]

use aws_config::BehaviorVersion;
use aws_sdk_s3::Client;
use bench_vortex::website::names::NAMES;
use bench_vortex::website::read_s3::read_benchmark_entries;
use vortex::VortexSessionDefault;
use vortex::session::VortexSession;

const BUCKET: &str = "vortex-benchmark-results-database";
const KEY: &str = "test/random_access.vortex";

fn main() {
    let runtime = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
    runtime.block_on(async_main());
}

async fn async_main() {
    let session = VortexSession::default();

    // Load AWS config with SSO profile.
    let config = aws_config::defaults(BehaviorVersion::latest())
        .profile_name("PowerUserAccess-375504701696")
        .load()
        .await;
    let client = Client::new(&config);

    println!(
        "Reading benchmark entries from s3://{}/{}...\n",
        BUCKET, KEY
    );

    let entries = read_benchmark_entries(&client, &session, BUCKET, KEY)
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
