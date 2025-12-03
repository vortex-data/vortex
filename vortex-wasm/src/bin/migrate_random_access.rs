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
use vortex::array::arrays::VarBinArray;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::compressor::CompactCompressor;
use vortex::dtype::DType;
use vortex::dtype::FieldNames;
use vortex::dtype::Nullability;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex::session::VortexSession;

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

/// Maps the JSON `name` field to the series name string.
fn series_name(name: &str) -> &'static str {
    match name {
        "random-access/vortex-tokio-local-disk" => "vortex-nvme",
        "random-access/parquet-tokio-local-disk" => "parquet-nvme",
        "random-access/lance-tokio-local-disk" => "lance-nvme",
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
    let mut benchmark_groups: Vec<&str> = Vec::with_capacity(num_entries);
    let mut chart_names: Vec<&str> = Vec::with_capacity(num_entries);
    let mut series_names: Vec<&str> = Vec::with_capacity(num_entries);
    let mut values: Vec<u64> = Vec::with_capacity(num_entries);

    for entry in &entries {
        // Decode hex commit_id to 20 binary bytes.
        let bytes = hex::decode(&entry.commit_id).expect("Invalid hex in commit_id");
        assert_eq!(bytes.len(), 20, "commit_id must decode to 20 bytes");
        commit_id_bytes.extend_from_slice(&bytes);

        // All entries have the same benchmark_group and chart_name.
        benchmark_groups.push("random-access");
        chart_names.push("random-access");

        // Map name to series_name string.
        series_names.push(series_name(&entry.name));

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

    // benchmark_group: utf8
    let benchmark_group_array = VarBinArray::from_iter(
        benchmark_groups.iter().map(|s| Some(*s)),
        DType::Utf8(Nullability::NonNullable),
    );

    // chart_name: utf8
    let chart_name_array = VarBinArray::from_iter(
        chart_names.iter().map(|s| Some(*s)),
        DType::Utf8(Nullability::NonNullable),
    );

    // series_name: utf8
    let series_name_array = VarBinArray::from_iter(
        series_names.iter().map(|s| Some(*s)),
        DType::Utf8(Nullability::NonNullable),
    );

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
