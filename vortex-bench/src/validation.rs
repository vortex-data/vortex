// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared utilities for normalizing query results for cross-engine comparison.
//!
//! Both DataFusion and DuckDB use these functions to produce identical string
//! representations of the same values, enabling a single set of reference
//! files to validate results from either engine.

use std::str::FromStr;

use bigdecimal::BigDecimal;

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

fn big_decimal_to_str(value: BigDecimal) -> String {
    value.round(12).normalized().to_plain_string()
}

/// Serialize a set of column names and rows into a normalized TSV string.
///
/// Rows are sorted lexicographically before serialization so that the output
/// is deterministic regardless of query execution order.
pub fn rows_to_normalized_tsv(column_names: &[String], rows: &mut Vec<Vec<String>>) -> String {
    rows.sort();

    let mut out = String::new();
    out.push_str(&column_names.join("\t"));
    out.push('\n');
    for row in rows {
        out.push_str(&row.join("\t"));
        out.push('\n');
    }
    out
}
