// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Binary to append a single benchmark entry to a Vortex file.

#![allow(clippy::expect_used)]

use std::env;

use vortex_dtype::DType;
use vortex_dtype::FieldNames;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_dtype::StructFields;
use vortex_scalar::Scalar;
use vortex_wasm::benchmark_website::append_entry;

/// Returns the expected DType for the benchmark data file.
///
/// The schema is a struct with two fields:
/// - `value`: u64 (non-nullable)
/// - `commit_id`: utf8 string (non-nullable)
fn benchmark_dtype() -> DType {
    DType::Struct(
        StructFields::new(
            FieldNames::from(["value", "commit_id"]),
            vec![
                DType::Primitive(PType::U64, Nullability::NonNullable),
                DType::Utf8(Nullability::NonNullable),
            ],
        ),
        Nullability::NonNullable,
    )
}

/// Creates a benchmark scalar from a value and commit ID.
fn benchmark_scalar(value: u64, commit_id: &str) -> Scalar {
    Scalar::struct_(
        benchmark_dtype(),
        vec![
            Scalar::primitive(value, Nullability::NonNullable),
            Scalar::utf8(commit_id, Nullability::NonNullable),
        ],
    )
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 4 {
        eprintln!("Usage: append_entries <vortex_file> <value> <commit_id>");
        eprintln!();
        eprintln!("Appends a single benchmark entry to the Vortex file.");
        return;
    }

    let vortex_path = &args[1];
    let value: u64 = args[2].parse().expect("Failed to parse value as u64");
    let commit_id = &args[3];

    let scalar = benchmark_scalar(value, commit_id);

    let total = append_entry(vortex_path, vortex_path, scalar)
        .expect("Failed to append entry to Vortex file");

    println!(
        "Appended entry (value={}, commit_id={}) to {} (total: {} entries)",
        value, commit_id, vortex_path, total
    );
}
