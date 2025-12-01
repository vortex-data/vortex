// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Binary to append a single benchmark entry to a Vortex file.

#![allow(clippy::expect_used)]

use std::env;
use std::path::Path;

use vortex_array::builders::builder_with_capacity;
use vortex_dtype::DType;
use vortex_dtype::FieldNames;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_dtype::StructFields;
use vortex_error::VortexResult;
use vortex_file::update_file;
use vortex_scalar::Scalar;

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

/// Reads a Vortex file and appends a single scalar entry, writing the result back.
///
/// This function:
/// 1. Reads the existing Vortex file (using the scalar's dtype)
/// 2. Appends the new scalar to the existing data using a builder
/// 3. Writes the combined data back to the output path
///
/// The input and output paths can be the same to overwrite the existing file.
///
/// # Arguments
///
/// * `input_path` - Path to the existing Vortex file to read.
/// * `output_path` - Path to write the updated Vortex file (can be same as input).
/// * `new_entry` - The scalar to append. Its dtype is used for reading/writing the file.
///
/// # Returns
///
/// The total number of entries in the resulting file.
pub fn append_entry(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    new_entry: Scalar,
) -> VortexResult<usize> {
    let dtype = new_entry.dtype().clone();

    let summary = update_file(input_path, output_path, |existing_array| async move {
        let existing_len = existing_array.len();

        // Create a builder and extend with existing data, then append the new entry.
        let total_capacity = existing_len + 1;
        let mut builder = builder_with_capacity(&dtype, total_capacity);

        // Add existing data.
        builder.extend_from_array(&existing_array);

        // Append the new entry.
        builder.append_scalar(&new_entry)?;

        Ok(builder.finish())
    })?;

    #[expect(clippy::cast_possible_truncation)]
    Ok(summary.row_count() as usize)
}
