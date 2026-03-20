// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared utilities for normalizing query results for cross-engine comparison.
//!
//! Both DataFusion and DuckDB use these functions to produce identical string
//! representations of the same values, enabling a single set of reference
//! files to validate results from either engine.

use std::path::Path;
use std::str::FromStr;

use bigdecimal::BigDecimal;
use datafusion_sqllogictest::value_normalizer;
use sqllogictest::default_validator;

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

/// Normalize a column name for cross-engine comparison.
///
/// Applies the following transformations:
/// - Lowercase the name
/// - Strip table-qualifier prefixes (e.g. `sum(hits.AdvEngineID)` → `sum(advengineid)`)
/// - Map DuckDB's `count_star()` to `count(*)`
pub fn normalize_column_name(name: &str) -> String {
    let mut n = name.to_lowercase();
    // DuckDB uses count_star() where DataFusion uses count(*)
    n = n.replace("count_star()", "count(*)");
    // Strip DuckDB type-cast functions: int64(1) → 1, int32(x) → x, etc.
    for cast_fn in &["int64(", "int32(", "int16(", "int8(", "uint64(", "uint32("] {
        while let Some(start) = n.find(cast_fn) {
            let after = start + cast_fn.len();
            // Find the matching closing paren
            if let Some(close) = find_matching_paren(&n[after..]) {
                let inner = &n[after..after + close];
                n = format!("{}{}{}", &n[..start], inner, &n[after + close + 1..]);
            } else {
                break;
            }
        }
    }
    // Collapse redundant parentheses: ((expr)) → (expr)
    while n.contains("((") {
        let prev = n.clone();
        n = collapse_double_parens(&n);
        if n == prev {
            break;
        }
    }
    // Strip table-qualifier prefixes like "hits." or "lineitem."
    // These appear as an identifier (alphanumeric/underscore) followed by a dot
    // before another identifier, e.g. "sum(hits.col)" → "sum(col)".
    loop {
        let bytes = n.as_bytes();
        let dot_pos = match n.find('.') {
            Some(p) => p,
            None => break,
        };
        // Check there's an identifier char before the dot
        if dot_pos == 0 || !bytes[dot_pos - 1].is_ascii_alphanumeric() {
            break;
        }
        // Check there's an identifier char after the dot
        if dot_pos + 1 >= bytes.len() || !bytes[dot_pos + 1].is_ascii_alphanumeric() {
            break;
        }
        // Find the start of the table-name word before the dot
        let word_start = n[..dot_pos]
            .rfind(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .map(|p| p + 1)
            .unwrap_or(0);
        // Remove "tablename."
        n = format!("{}{}", &n[..word_start], &n[dot_pos + 1..]);
    }
    // Strip outer parentheses wrapping the entire name, e.g. "(clientip - 1)" → "clientip - 1"
    while n.starts_with('(') && n.ends_with(')') {
        if let Some(close) = find_matching_paren(&n[1..]) {
            if close + 1 == n.len() - 1 {
                n = n[1..n.len() - 1].to_string();
            } else {
                break;
            }
        } else {
            break;
        }
    }
    n
}

/// Find the position of the matching closing paren (relative to input start).
/// Input starts right after the opening paren.
fn find_matching_paren(s: &str) -> Option<usize> {
    let mut depth = 1u32;
    for (i, ch) in s.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Collapse `((expr))` → `(expr)` for the outermost double-paren pair.
fn collapse_double_parens(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut i = 0;
    let mut result = String::with_capacity(s.len());
    while i < bytes.len() {
        if i + 1 < bytes.len() && bytes[i] == b'(' && bytes[i + 1] == b'(' {
            // Find the matching close for the outer paren
            if let Some(outer_close) = find_matching_paren(&s[i + 1..]) {
                let outer_close = i + 1 + outer_close;
                // Check if the char before outer_close is also ')'
                if outer_close > 0 && bytes[outer_close - 1] == b')' {
                    // Collapse: skip outer ( and outer )
                    result.push_str(&s[i + 1..outer_close]);
                    i = outer_close + 1;
                    continue;
                }
            }
        }
        result.push(bytes[i] as char);
        i += 1;
    }
    result
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

/// Serialize a set of column names and rows into a normalized TSV string.
///
/// Column names are normalized via [`normalize_column_name`] so that
/// engine-specific differences (casing, table prefixes, function names)
/// are eliminated. Rows are sorted lexicographically before serialization
/// so that the output is deterministic regardless of query execution order.
pub fn rows_to_normalized_tsv(column_names: &[String], rows: &mut Vec<Vec<String>>) -> String {
    rows.sort();

    let normalized_names: Vec<String> = column_names
        .iter()
        .map(|n| normalize_column_name(n))
        .collect();

    let mut out = String::new();
    out.push_str(&normalized_names.join("\t"));
    out.push('\n');
    for row in rows {
        out.push_str(&row.join("\t"));
        out.push('\n');
    }
    out
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

/// Parse expected result rows from a `.slt.no` file.
///
/// Extracts all lines after the first `----` separator until the end of the
/// file (or until a blank line followed by another record header). Each line
/// becomes one entry in the returned `Vec<String>`.
fn parse_slt_expected_rows(content: &str) -> Vec<String> {
    let mut in_results = false;
    let mut expected = Vec::new();

    for line in content.lines() {
        if !in_results {
            if line == "----" {
                in_results = true;
            }
            continue;
        }
        expected.push(line.to_string());
    }

    // Trim trailing empty lines
    while expected.last().is_some_and(|l| l.is_empty()) {
        expected.pop();
    }

    expected
}

/// Validate actual query rows against a `.slt.no` reference file.
///
/// Reads the file at `slt_path`, parses the expected rows after `----`,
/// applies `rowsort` to the actual rows, and compares using
/// `sqllogictest::default_validator` with `datafusion_sqllogictest::value_normalizer`.
///
/// Returns `Ok(())` if the results match, or `Err` with a diff description.
pub fn validate_against_slt(
    slt_path: &Path,
    actual_rows: &mut [Vec<String>],
) -> Result<(), String> {
    let content = std::fs::read_to_string(slt_path)
        .map_err(|e| format!("Failed to read {}: {e}", slt_path.display()))?;

    let expected_lines = parse_slt_expected_rows(&content);

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
