// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Binary to migrate the random-access JSON benchmark data to a Vortex file.
//!
//! This reads the JSON file from `organized_data/random-access/random_access.json` and converts
//! it to a Vortex file with the [`BenchmarkEntry`] schema.

#![allow(clippy::expect_used, clippy::panic)]

use std::env;
use std::fs;

use serde::Deserialize;
use vortex::VortexSessionDefault;
use vortex::array::IntoArray;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::compressor::CompactCompressor;
use vortex::dtype::FieldNames;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex::session::VortexSession;

/// Name ID constants from `bench-vortex/src/website/names.rs`.
mod name_ids {
    pub const RANDOM_ACCESS: u32 = 2;
    pub const VORTEX_NVME: u32 = 3;
    pub const PARQUET_NVME: u32 = 4;
    pub const LANCE_NVME: u32 = 5;
}

/// Represents a benchmark entry from the JSON file.
#[derive(Debug, Deserialize)]
struct JsonEntry {
    name: String,
    value: u64,
    commit_id: String,
    // Ignore other fields from JSON.
    #[serde(flatten)]
    _extra: serde_json::Value,
}

/// Maps the JSON `name` field to a series name ID.
fn series_name_id(name: &str) -> u32 {
    match name {
        "random-access/vortex-tokio-local-disk" => name_ids::VORTEX_NVME,
        "random-access/parquet-tokio-local-disk" => name_ids::PARQUET_NVME,
        "random-access/lance-tokio-local-disk" => name_ids::LANCE_NVME,
        _ => panic!("Unknown benchmark name: {}", name),
    }
}

fn main() {
    let runtime = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");

    runtime.block_on(async_main());
}

async fn async_main() {
    let session = VortexSession::default();

    let args: Vec<String> = env::args().collect();
    let input_path = args
        .get(1)
        .expect("Usage: migrate_random_access <input_json> <output_file>");
    let output_path = args
        .get(2)
        .map(String::as_str)
        .expect("Usage: migrate_random_access <input_json> <output_file>");

    // Parse JSON.
    let contents = fs::read_to_string(input_path).expect("Failed to read file");
    let entries: Vec<JsonEntry> = serde_json::from_str(&contents).expect("Failed to parse JSON");

    let num_entries = entries.len();
    println!("Parsing {} entries from JSON...", num_entries);

    // Extract fields into separate vectors.
    let mut commit_id_bytes: Vec<u8> = Vec::with_capacity(num_entries * 20);
    let mut benchmark_groups: Vec<u32> = Vec::with_capacity(num_entries);
    let mut chart_names: Vec<u32> = Vec::with_capacity(num_entries);
    let mut series_names: Vec<u32> = Vec::with_capacity(num_entries);
    let mut values: Vec<u64> = Vec::with_capacity(num_entries);

    for entry in &entries {
        // Decode hex commit_id to 20 binary bytes.
        let bytes = hex::decode(&entry.commit_id).expect("Invalid hex in commit_id");
        assert_eq!(bytes.len(), 20, "commit_id must decode to 20 bytes");
        commit_id_bytes.extend_from_slice(&bytes);

        // All entries have the same benchmark_group and chart_name.
        benchmark_groups.push(name_ids::RANDOM_ACCESS);
        chart_names.push(name_ids::RANDOM_ACCESS);

        // Map name to series_name ID.
        series_names.push(series_name_id(&entry.name));

        values.push(entry.value);
    }

    // Create arrays.

    // commit_id: FixedSizeList<u8, 20>
    let commit_id_elements =
        PrimitiveArray::new(Buffer::from(commit_id_bytes), Validity::NonNullable);
    let commit_id_array = FixedSizeListArray::try_new(
        commit_id_elements.into_array(),
        20,
        Validity::NonNullable,
        num_entries,
    )
    .expect("Failed to create commit_id array");

    // benchmark_group: u32
    let benchmark_group_array =
        PrimitiveArray::new(Buffer::from(benchmark_groups), Validity::NonNullable);

    // chart_name: u32
    let chart_name_array = PrimitiveArray::new(Buffer::from(chart_names), Validity::NonNullable);

    // series_name: u32
    let series_name_array = PrimitiveArray::new(Buffer::from(series_names), Validity::NonNullable);

    // value: u64
    let value_array = PrimitiveArray::new(Buffer::from(values), Validity::NonNullable);

    // Create struct array with all fields.
    let struct_array = StructArray::try_new(
        FieldNames::from([
            "commit_id",
            "benchmark_group",
            "chart_name",
            "series_name",
            "value",
        ]),
        vec![
            commit_id_array.into_array(),
            benchmark_group_array.into_array(),
            chart_name_array.into_array(),
            series_name_array.into_array(),
            value_array.into_array(),
        ],
        num_entries,
        Validity::NonNullable,
    )
    .expect("Failed to create struct array");

    println!("Created struct array with {} entries", num_entries);
    println!("Schema: {}", struct_array.dtype());

    // Write to Vortex file with compression.
    let file = tokio::fs::File::create(output_path)
        .await
        .expect("Failed to create output file");

    session
        .write_options()
        .with_strategy(
            WriteStrategyBuilder::new()
                .with_compressor(CompactCompressor::default())
                .build(),
        )
        .write(file, struct_array.to_array_stream())
        .await
        .expect("Failed to write Vortex file");

    println!("Wrote {} entries to {}", num_entries, output_path);
}
