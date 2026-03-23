// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared utilities for normalizing query results for cross-engine comparison.
//!
//! Both DataFusion and DuckDB use these functions to produce normalized string
//! representations of query results. Each engine has its own set of reference
//! files under `{results_dir}/{engine}/`; files that are identical across
//! engines use `include` directives to point at a shared file in the parent
//! directory.

use std::path::Path;
use std::str::FromStr;

use bigdecimal::BigDecimal;
use datafusion_sqllogictest::value_normalizer;
use sqllogictest::DefaultColumnType;
use sqllogictest::Record;
use sqllogictest::default_validator;
use sqllogictest::parse_file;

/// Normalize a `f64` value to a canonical string representation.
///
/// Matches the rounding behavior of `datafusion-sqllogictest` and
/// `vortex-sqllogictest`: rounds via `BigDecimal` to 12 decimal places.
pub fn normalize_f64(value: f64) -> String {
    if value.is_nan() {
        "NaN".to_string()
    } else if value == f64::INFINITY {
        "Infinity".to_string()
    } else if value == f64::NEG_INFINITY {
        "-Infinity".to_string()
    } else {
        big_decimal_to_str(
            BigDecimal::from_str(&value.to_string()).expect("f64 always parses to BigDecimal"),
        )
    }
}

/// Normalize a `f32` value to a canonical string representation.
pub fn normalize_f32(value: f32) -> String {
    if value.is_nan() {
        "NaN".to_string()
    } else if value == f32::INFINITY {
        "Infinity".to_string()
    } else if value == f32::NEG_INFINITY {
        "-Infinity".to_string()
    } else {
        big_decimal_to_str(
            BigDecimal::from_str(&value.to_string()).expect("f32 always parses to BigDecimal"),
        )
    }
}

/// Normalize a decimal value (i128 with scale) to a canonical string.
pub fn normalize_decimal(value: i128, scale: i8) -> String {
    let bd = BigDecimal::new(value.into(), scale as i64);
    big_decimal_to_str(bd)
}

/// Normalize a string value, matching sqllogictest conventions.
pub fn normalize_string(value: &str) -> String {
    if value.is_empty() {
        "(empty)".to_string()
    } else {
        value.trim_end_matches('\n').replace('\0', "\\0")
    }
}

/// Normalize a timestamp string to a canonical format.
///
/// Replaces the ISO 8601 `T` separator with a space so that
/// `2013-07-02T00:00:00` and `2013-07-02 00:00:00` produce the same output.
pub fn normalize_timestamp(value: &str) -> String {
    // Only replace T that sits between a date and time pattern to avoid
    // mangling unrelated strings.
    if let Some(t_pos) = value.find('T')
        && t_pos >= 10
        && value.len() > t_pos + 1
    {
        let before = &value[t_pos - 1..t_pos];
        let after = &value[t_pos + 1..t_pos + 2];
        if before.as_bytes()[0].is_ascii_digit() && after.as_bytes()[0].is_ascii_digit() {
            let mut s = value.to_string();
            s.replace_range(t_pos..t_pos + 1, " ");
            return s;
        }
    }
    value.to_string()
}

fn big_decimal_to_str(value: BigDecimal) -> String {
    value.round(12).normalized().to_plain_string()
}

/// Serialize query results into sqllogictest `.slt.no` format.
///
/// Produces a complete sqllogictest record with the form:
/// ```text
/// query {types} rowsort
/// {sql}
/// ----
/// {value1} {value2} ...
/// ```
///
/// Rows are sorted lexicographically (via `rowsort`) so that output is
/// deterministic regardless of query execution order.
pub fn rows_to_slt(query_sql: &str, column_types: &str, rows: &mut Vec<Vec<String>>) -> String {
    rows.sort();

    let mut out = String::new();
    out.push_str(&format!("query {column_types} rowsort\n"));
    out.push_str(query_sql.trim());
    out.push('\n');
    out.push_str("----\n");
    for row in rows {
        out.push_str(&row.join(" "));
        out.push('\n');
    }
    out
}

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
