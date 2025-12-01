// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::expect_used)]

use std::env;
use std::fs;

use serde::Deserialize;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::expr::session::ExprSession;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::current::CurrentThreadRuntime;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::session::LayoutSession;
use vortex_metrics::VortexMetrics;
use vortex_session::VortexSession;

/// Represents a benchmark entry with value and commit ID.
#[derive(Debug, Deserialize)]
struct BenchmarkEntry {
    value: u64,
    commit_id: String,
}

fn main() {
    let runtime = CurrentThreadRuntime::new();

    let session = VortexSession::empty()
        .with::<VortexMetrics>()
        .with::<ArraySession>()
        .with::<LayoutSession>()
        .with::<ExprSession>()
        .with::<RuntimeSession>()
        .with_handle(runtime.handle());

    vortex_file::register_default_encodings(&session);

    runtime.block_on(async_main(session));
}

async fn async_main(session: VortexSession) {
    let args: Vec<String> = env::args().collect();
    let input_path = args
        .get(1)
        .expect("Usage: migrate <input_json> <output_vortex>");
    let output_path = args
        .get(2)
        .expect("Usage: migrate <input_json> <output_vortex>");

    // Parse JSON.
    let contents = fs::read_to_string(input_path).expect("Failed to read file");
    let entries: Vec<BenchmarkEntry> =
        serde_json::from_str(&contents).expect("Failed to parse JSON");

    // Extract values and commit_ids into separate vectors.
    let values: Vec<u64> = entries.iter().map(|e| e.value).collect();
    let commit_ids: Vec<&str> = entries.iter().map(|e| e.commit_id.as_str()).collect();
    let num_entries = entries.len();

    // Create primitive array for values.
    let values_array = PrimitiveArray::new(Buffer::from(values), Validity::NonNullable);

    // Create VarBin array for commit_ids (UTF8 strings).
    let commit_ids_array = VarBinArray::from_iter(
        commit_ids.into_iter().map(Some),
        DType::Utf8(Nullability::NonNullable),
    );

    // Create struct array with both fields.
    let struct_array = StructArray::from_fields(&[
        ("value", values_array.into_array()),
        ("commit_id", commit_ids_array.into_array()),
    ])
    .expect("Failed to create struct array");

    // Write to Vortex file using push-based API.
    let file = async_fs::File::create(output_path)
        .await
        .expect("Failed to create output file");

    let mut writer = session
        .write_options()
        .writer(file, struct_array.dtype().clone());

    writer
        .push(struct_array.into_array())
        .await
        .expect("Failed to push array");

    writer.finish().await.expect("Failed to finish writing");

    println!("Wrote {} entries to {}", num_entries, output_path);
}
