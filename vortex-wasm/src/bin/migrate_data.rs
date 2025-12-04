// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Binary to migrate a JSON array of [`BenchmarkEntry`] objects to a Vortex file.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p vortex-wasm --bin migrate_data -- <input.json> <output.vortex>
//! ```

#![allow(clippy::expect_used, clippy::panic)]

use std::env;
use std::fs;

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
use vortex_wasm::website::entry::BenchmarkEntry;

#[tokio::main]
async fn main() {
    let session = VortexSession::default();

    let args: Vec<String> = env::args().collect();
    let input_path = args
        .get(1)
        .expect("Usage: migrate_all <input_json> <output_file>");
    let output_path = args
        .get(2)
        .map(String::as_str)
        .expect("Usage: migrate_all <input_json> <output_file>");

    let contents = fs::read_to_string(input_path).expect("Failed to read file");
    let entries: Vec<BenchmarkEntry> =
        serde_json::from_str(&contents).expect("Failed to parse JSON");

    let num_entries = entries.len();
    println!("Parsed {num_entries} entries from JSON");

    // Extract fields into columnar vectors.
    let mut commit_id_bytes: Vec<u8> = Vec::with_capacity(num_entries * 20);
    let mut benchmark_groups: Vec<&str> = Vec::with_capacity(num_entries);
    let mut chart_names: Vec<&str> = Vec::with_capacity(num_entries);
    let mut series_names: Vec<&str> = Vec::with_capacity(num_entries);
    let mut values: Vec<u64> = Vec::with_capacity(num_entries);

    for entry in &entries {
        commit_id_bytes.extend_from_slice(&entry.commit_id.0);
        benchmark_groups.push(&entry.benchmark_group);
        chart_names.push(&entry.chart_name);
        series_names.push(&entry.series_name);
        values.push(entry.value);
    }

    // Build Vortex arrays.
    let commit_id_elements =
        PrimitiveArray::new(Buffer::from(commit_id_bytes), Validity::NonNullable);
    let commit_id_array = FixedSizeListArray::try_new(
        commit_id_elements.into_array(),
        20,
        Validity::NonNullable,
        num_entries,
    )
    .expect("Failed to create commit_id array");

    let benchmark_group_array = VarBinArray::from_iter(
        benchmark_groups.iter().map(|s| Some(*s)),
        DType::Utf8(Nullability::NonNullable),
    );
    let chart_name_array = VarBinArray::from_iter(
        chart_names.iter().map(|s| Some(*s)),
        DType::Utf8(Nullability::NonNullable),
    );
    let series_name_array = VarBinArray::from_iter(
        series_names.iter().map(|s| Some(*s)),
        DType::Utf8(Nullability::NonNullable),
    );
    let value_array = PrimitiveArray::new(Buffer::from(values), Validity::NonNullable);

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

    println!("Wrote {num_entries} entries to {output_path}");
}
