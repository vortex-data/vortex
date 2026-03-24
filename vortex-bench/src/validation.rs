// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Validation utilities for comparing query results against `.slt.no` reference files.
//!
//! Each engine has its own set of reference files under `{results_dir}/{engine}/`;
//! files that are identical across engines use `include` directives to point at
//! a shared file in the parent directory.

use std::path::Path;

use datafusion_sqllogictest::value_normalizer;
use sqllogictest::DefaultColumnType;
use sqllogictest::Record;
use sqllogictest::default_validator;
use sqllogictest::parse_file;

/// Validate actual query rows against a `.slt.no` reference file.
///
/// Parses the file using `sqllogictest::parse_file` (which resolves `include`
/// directives), extracts the expected rows from the first `Query` record, and
/// compares using `sqllogictest::default_validator` with
/// `datafusion_sqllogictest::value_normalizer`.
///
/// Returns `Ok(())` if the results match, or `Err` with a diff description.
pub fn validate_against_slt(
    slt_path: &Path,
    actual_rows: &mut [Vec<String>],
) -> Result<(), String> {
    let records = parse_file::<DefaultColumnType>(slt_path)
        .map_err(|e| format!("Failed to parse {}: {e}", slt_path.display()))?;

    let expected_lines = records
        .into_iter()
        .find_map(|rec| {
            if let Record::Query {
                expected: sqllogictest::QueryExpect::Results { results, .. },
                ..
            } = rec
            {
                return Some(results);
            }
            None
        })
        .ok_or_else(|| format!("No query record found in {}", slt_path.display()))?;

    // Apply rowsort to actual rows (same as the slt file specifies)
    actual_rows.sort();

    let matches = default_validator(value_normalizer, actual_rows, &expected_lines);

    if matches {
        Ok(())
    } else {
        // Build a human-readable diff for the error message
        let actual_flat: Vec<String> = actual_rows.iter().map(|row| row.join(" ")).collect();

        let mut diff_msg = String::new();
        diff_msg.push_str(&format!("Mismatch against {}\n", slt_path.display()));
        diff_msg.push_str(&format!(
            "Expected {} lines, got {} lines\n",
            expected_lines.len(),
            actual_flat.len()
        ));

        let max_lines = expected_lines.len().max(actual_flat.len()).min(20);
        for i in 0..max_lines {
            let exp = expected_lines
                .get(i)
                .map(String::as_str)
                .unwrap_or("<missing>");
            let act = actual_flat
                .get(i)
                .map(String::as_str)
                .unwrap_or("<missing>");
            if exp != act {
                diff_msg.push_str(&format!("  line {i}: expected: {exp}\n"));
                diff_msg.push_str(&format!("  line {i}:   actual: {act}\n"));
            }
        }
        if expected_lines.len().max(actual_flat.len()) > 20 {
            diff_msg.push_str("  ... (truncated)\n");
        }

        Err(diff_msg)
    }
}
