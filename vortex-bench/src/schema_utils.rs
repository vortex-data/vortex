// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Utility to build Arrow schemas from SQL DDL column definitions.
//!
//! Instead of writing verbose Arrow `Field::new(...)` calls, you can write:
//!
//! ```
//! use vortex_bench::schema_from_ddl;
//! let schema = schema_from_ddl("
//!     id BIGINT NOT NULL,
//!     name VARCHAR NOT NULL,
//!     score DECIMAL(15,2),
//!     created_at TIMESTAMP NOT NULL,
//! ");
//! assert_eq!(schema.fields().len(), 4);
//! ```

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use arrow_schema::TimeUnit;

use crate::vortex_panic;

/// Build an Arrow [`Schema`] from SQL DDL column definitions.
///
/// Each line should be `COLUMN_NAME TYPE [NOT NULL]`, separated by commas or
/// newlines. Lines that are empty or start with `--` are skipped.
///
/// Supported SQL types:
/// - `BIGINT` / `INT8` → `Int64`
/// - `INTEGER` / `INT` / `INT4` → `Int32`
/// - `SMALLINT` / `INT2` → `Int16`
/// - `TINYINT` / `INT1` → `Int8`
/// - `BOOLEAN` / `BOOL` → `Boolean`
/// - `FLOAT` / `REAL` / `FLOAT4` → `Float32`
/// - `DOUBLE` / `FLOAT8` → `Float64`
/// - `VARCHAR` / `TEXT` / `STRING` → `Utf8View`
/// - `TIMESTAMP` → `Timestamp(Microsecond, None)`
/// - `DATE` → `Date32`
/// - `DECIMAL(p,s)` → `Decimal128(p,s)`
pub fn schema_from_ddl(ddl: &str) -> Schema {
    // Strip comment lines before splitting by commas
    let cleaned: String = ddl
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.starts_with("--"))
        .collect::<Vec<_>>()
        .join("\n");

    let fields: Vec<Field> = split_columns(&cleaned)
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|col_def| {
            parse_column_def(col_def)
                .unwrap_or_else(|| vortex_panic!("failed to parse column definition: {col_def:?}"))
        })
        .collect();

    assert!(!fields.is_empty(), "schema_from_ddl: no columns parsed");
    Schema::new(fields)
}

/// Split column definitions by commas, but respect parentheses (e.g. `DECIMAL(15,2)`).
fn split_columns(ddl: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut depth = 0;
    let mut start = 0;
    for (i, ch) in ddl.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                result.push(&ddl[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    // Don't forget the last segment
    if start < ddl.len() {
        result.push(&ddl[start..]);
    }
    result
}

fn parse_column_def(def: &str) -> Option<Field> {
    let tokens: Vec<&str> = def.split_whitespace().collect();
    if tokens.len() < 2 {
        return None;
    }

    let name = tokens[0];
    let type_str = tokens[1].to_uppercase();
    let rest = tokens[2..].join(" ").to_uppercase();

    let nullable = !rest.contains("NOT NULL");

    let data_type = parse_sql_type(&type_str, &rest)?;

    Some(Field::new(name, data_type, nullable))
}

fn parse_sql_type(type_str: &str, rest: &str) -> Option<DataType> {
    // Handle DECIMAL(p,s) — the precision/scale may be in type_str or rest
    if type_str.starts_with("DECIMAL") {
        return parse_decimal(type_str, rest);
    }

    match type_str {
        // Integer types
        "BIGINT" | "INT8" => Some(DataType::Int64),
        "INTEGER" | "INT" | "INT4" => Some(DataType::Int32),
        "SMALLINT" | "INT2" => Some(DataType::Int16),
        "TINYINT" | "INT1" => Some(DataType::Int8),

        // Boolean
        "BOOLEAN" | "BOOL" => Some(DataType::Boolean),

        // Float types
        "FLOAT" | "REAL" | "FLOAT4" => Some(DataType::Float32),
        "DOUBLE" | "FLOAT8" => Some(DataType::Float64),

        // String types
        "VARCHAR" | "TEXT" | "STRING" => Some(DataType::Utf8View),

        // Date/time types
        "TIMESTAMP" => Some(DataType::Timestamp(TimeUnit::Microsecond, None)),
        "DATE" => Some(DataType::Date32),

        _ => None,
    }
}

fn parse_decimal(type_str: &str, rest: &str) -> Option<DataType> {
    // Try parsing from "DECIMAL(p,s)" all in one token
    let combined = format!("{type_str} {rest}");
    let start = combined.find('(')?;
    let end = combined.find(')')?;
    let inner = &combined[start + 1..end];
    let parts: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();

    let precision: u8 = parts.first()?.parse().ok()?;
    let scale: i8 = parts.get(1).unwrap_or(&"0").parse().ok()?;
    Some(DataType::Decimal128(precision, scale))
}

#[cfg(test)]
mod tests {
    use arrow_schema::DataType;
    use arrow_schema::TimeUnit;

    use super::*;

    #[test]
    fn test_basic_types() {
        let schema = schema_from_ddl(
            "
            id BIGINT NOT NULL,
            name VARCHAR NOT NULL,
            age SMALLINT,
            active BOOLEAN NOT NULL,
        ",
        );

        assert_eq!(schema.fields().len(), 4);

        assert_eq!(schema.field(0).name(), "id");
        assert_eq!(schema.field(0).data_type(), &DataType::Int64);
        assert!(!schema.field(0).is_nullable());

        assert_eq!(schema.field(1).name(), "name");
        assert_eq!(schema.field(1).data_type(), &DataType::Utf8View);
        assert!(!schema.field(1).is_nullable());

        assert_eq!(schema.field(2).name(), "age");
        assert_eq!(schema.field(2).data_type(), &DataType::Int16);
        assert!(schema.field(2).is_nullable());

        assert_eq!(schema.field(3).name(), "active");
        assert_eq!(schema.field(3).data_type(), &DataType::Boolean);
        assert!(!schema.field(3).is_nullable());
    }

    #[test]
    fn test_decimal_and_timestamps() {
        let schema = schema_from_ddl(
            "
            price DECIMAL(15,2) NOT NULL,
            created_at TIMESTAMP NOT NULL,
            birth_date DATE,
        ",
        );

        assert_eq!(schema.field(0).data_type(), &DataType::Decimal128(15, 2));
        assert_eq!(
            schema.field(1).data_type(),
            &DataType::Timestamp(TimeUnit::Microsecond, None)
        );
        assert_eq!(schema.field(2).data_type(), &DataType::Date32);
    }

    #[test]
    fn test_comments_and_blank_lines() {
        let schema = schema_from_ddl(
            "
            -- this is a comment
            id BIGINT NOT NULL,

            -- another comment
            name VARCHAR NOT NULL,
        ",
        );

        assert_eq!(schema.fields().len(), 2);
    }
}
