// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::process::Command;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bigdecimal::BigDecimal;
use datafusion_sqllogictest::DFColumnType;
use indicatif::ProgressBar;
use sqllogictest::DBOutput;
use sqllogictest::Normalizer;
use sqllogictest::runner::AsyncDB;
use vortex::error::VortexError;
use vortex::error::VortexExpect;
use vortex_duckdb::duckdb::Connection;
use vortex_duckdb::duckdb::Database;
use vortex_duckdb::duckdb::ExtractedValue;
use vortex_duckdb::duckdb::LogicalType;
use vortex_duckdb::duckdb::LogicalTypeRef;
use vortex_duckdb::duckdb::Value;
use vortex_duckdb::initialize;

#[derive(Debug, thiserror::Error)]
pub enum DuckDBTestError {
    Other(String),
    Vortex(#[from] VortexError),
}

impl std::fmt::Display for DuckDBTestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DuckDBTestError::Other(msg) => write!(f, "Other: {msg}"),
            DuckDBTestError::Vortex(inner) => write!(f, "Vortex: {inner}"),
        }
    }
}

struct Inner {
    conn: Connection,
    _db: Database,
}

unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

pub struct DuckDB {
    inner: Arc<Inner>,
    pb: ProgressBar,
}

impl DuckDB {
    pub fn try_new(pb: ProgressBar) -> Result<Self, DuckDBTestError> {
        let db = Database::open_in_memory()?;
        db.register_vortex_scan_replacement()?;
        initialize(&db)?;

        let conn = db.connect()?;

        Ok(Self {
            pb,
            inner: Arc::new(Inner { conn, _db: db }),
        })
    }

    /// Turn the DuckDB logical type into a `DFColumnType`, which
    /// tells the runner what types they are. We use the one from DataFusion
    /// as its richer than the default one.
    fn normalize_column_type(logical_type: &LogicalTypeRef) -> DFColumnType {
        let type_id = logical_type.as_type_id();

        if type_id == LogicalType::int32().as_type_id()
            || type_id == LogicalType::int64().as_type_id()
            || type_id == LogicalType::uint64().as_type_id()
            || type_id == LogicalType::int128().as_type_id()
            || type_id == LogicalType::uint128().as_type_id()
        {
            DFColumnType::Integer
        } else if type_id == LogicalType::varchar().as_type_id() {
            DFColumnType::Text
        } else if type_id == LogicalType::bool().as_type_id() {
            DFColumnType::Boolean
        } else if type_id == LogicalType::float32().as_type_id()
            || type_id == LogicalType::float64().as_type_id()
            || logical_type.is_decimal()
        {
            DFColumnType::Float
        } else if type_id == LogicalType::timestamp().as_type_id()
            || type_id == LogicalType::timestamp_tz().as_type_id()
        {
            DFColumnType::Timestamp
        } else if type_id == LogicalType::date().as_type_id() {
            DFColumnType::DateTime
        } else {
            DFColumnType::Another
        }
    }
}

pub fn duckdb_validator(
    normalizer: Normalizer,
    actual: &[Vec<String>],
    expected: &[String],
) -> bool {
    let actual = actual.iter().flat_map(|strings| {
        strings
            .join(" ")
            .trim_end()
            .split('\n')
            .map(|line| line.trim_end().to_string())
            .collect::<Vec<_>>()
    });
    Iterator::eq(actual, expected.iter().map(normalizer))
}

#[async_trait]
impl AsyncDB for DuckDB {
    type Error = DuckDBTestError;
    type ColumnType = DFColumnType;

    async fn run(&mut self, sql: &str) -> Result<DBOutput<Self::ColumnType>, Self::Error> {
        let result = {
            let r = self.inner.conn.query(sql)?;

            if r.column_count() == 0 && r.row_count() == 0 {
                Ok(DBOutput::StatementComplete(0))
            } else {
                let mut types = Vec::default();
                let mut rows = Vec::default();

                for col_idx in 0..r.column_count() {
                    let col_idx = usize::try_from(col_idx).map_err(VortexError::from)?;
                    let dtype = r.column_type(col_idx);
                    types.push(Self::normalize_column_type(&dtype));
                }

                for chunk in r.into_iter() {
                    for row_idx in 0..chunk.len() {
                        let mut current_row = Vec::new();
                        for col_idx in 0..chunk.column_count() {
                            let vector = chunk.get_vector(col_idx);
                            match vector.get_value(row_idx, chunk.len()) {
                                Some(value) => {
                                    current_row.push(ValueDisplayAdapter(value).to_string())
                                }
                                None => current_row
                                    .push(Value::null(&vector.logical_type()).to_string()),
                            }
                        }

                        rows.push(current_row);
                    }
                }

                Ok(DBOutput::Rows { types, rows })
            }
        };

        self.pb.inc(1);

        result
    }

    async fn shutdown(&mut self) {}

    fn engine_name(&self) -> &str {
        "DuckDB"
    }

    async fn sleep(dur: Duration) {
        tokio::time::sleep(dur).await
    }

    async fn run_command(command: Command) -> std::io::Result<std::process::Output> {
        tokio::process::Command::from(command).output().await
    }
}

/// Rounds a floating-point value via BigDecimal to 12 decimal places,
/// matching the behavior of `big_decimal_to_str` from `datafusion_sqllogictest`.
fn big_decimal_to_str(value: BigDecimal) -> String {
    value.round(12).normalized().to_plain_string()
}

fn f32_to_str(value: f32) -> String {
    if value.is_nan() {
        "NaN".to_string()
    } else if value == f32::INFINITY {
        "Infinity".to_string()
    } else if value == f32::NEG_INFINITY {
        "-Infinity".to_string()
    } else {
        big_decimal_to_str(
            BigDecimal::from_str(&value.to_string())
                .ok()
                .vortex_expect("value can't be parsed to decimal"),
        )
    }
}

fn f64_to_str(value: f64) -> String {
    if value.is_nan() {
        "NaN".to_string()
    } else if value == f64::INFINITY {
        "Infinity".to_string()
    } else if value == f64::NEG_INFINITY {
        "-Infinity".to_string()
    } else {
        big_decimal_to_str(
            BigDecimal::from_str(&value.to_string())
                .ok()
                .vortex_expect("value can't be parsed to decimal"),
        )
    }
}

fn decimal_to_str(value: i128, scale: i8) -> String {
    let bd = BigDecimal::new(value.into(), scale as i64);
    big_decimal_to_str(bd)
}

fn varchar_to_str(value: &str) -> String {
    if value.is_empty() {
        "(empty)".to_string()
    } else {
        value.trim_end_matches('\n').replace('\0', "\\0")
    }
}

/// Adapter type to control how duckdb values are displayed.
/// Matches the behavior of `cell_to_string` from `datafusion_sqllogictest`.
struct ValueDisplayAdapter(Value);

impl std::fmt::Display for ValueDisplayAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0.extract() {
            ExtractedValue::Null => write!(f, "NULL"),
            ExtractedValue::TinyInt(v) => write!(f, "{v}"),
            ExtractedValue::SmallInt(v) => write!(f, "{v}"),
            ExtractedValue::Integer(v) => write!(f, "{v}"),
            ExtractedValue::BigInt(v) => write!(f, "{v}"),
            ExtractedValue::HugeInt(v) => write!(f, "{v}"),
            ExtractedValue::UTinyInt(v) => write!(f, "{v}"),
            ExtractedValue::USmallInt(v) => write!(f, "{v}"),
            ExtractedValue::UInteger(v) => write!(f, "{v}"),
            ExtractedValue::UBigInt(v) => write!(f, "{v}"),
            ExtractedValue::UHugeInt(v) => write!(f, "{v}"),
            ExtractedValue::Float(v) => write!(f, "{}", f32_to_str(v)),
            ExtractedValue::Double(v) => write!(f, "{}", f64_to_str(v)),
            ExtractedValue::Boolean(v) => write!(f, "{v}"),
            ExtractedValue::Varchar(s) => write!(f, "{}", varchar_to_str(s.as_str())),
            ExtractedValue::Decimal(_, scale, value) => {
                write!(f, "{}", decimal_to_str(value, scale))
            }
            // For types not specially handled by cell_to_string (dates, times, timestamps,
            // blobs, lists), delegate to DuckDB's native string representation.
            ExtractedValue::Blob(_)
            | ExtractedValue::Date(_)
            | ExtractedValue::Time(_)
            | ExtractedValue::TimestampNs(_)
            | ExtractedValue::Timestamp(_)
            | ExtractedValue::TimestampMs(_)
            | ExtractedValue::TimestampS(_)
            | ExtractedValue::List(_) => write!(f, "{}", self.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_duckdb::duckdb::Value;

    use super::ValueDisplayAdapter;

    fn display(value: Value) -> String {
        ValueDisplayAdapter(value).to_string()
    }

    #[test]
    fn test_null() {
        assert_eq!(display(Value::sql_null()), "NULL");
    }

    #[rstest]
    #[case(true, "true")]
    #[case(false, "false")]
    fn test_bool(#[case] input: bool, #[case] expected: &str) {
        assert_eq!(display(Value::from(input)), expected);
    }

    #[rstest]
    #[case(0i32, "0")]
    #[case(42i32, "42")]
    #[case(-1i32, "-1")]
    fn test_integer(#[case] input: i32, #[case] expected: &str) {
        assert_eq!(display(Value::from(input)), expected);
    }

    #[rstest]
    #[case(0i64, "0")]
    #[case(i64::MAX, "9223372036854775807")]
    #[case(i64::MIN, "-9223372036854775808")]
    fn test_bigint(#[case] input: i64, #[case] expected: &str) {
        assert_eq!(display(Value::from(input)), expected);
    }

    #[rstest]
    #[case(0.0f64, "0")]
    #[case(0.1, "0.1")]
    #[case(1.0 / 3.0, "0.333333333333")]
    #[case(f64::NAN, "NaN")]
    #[case(f64::INFINITY, "Infinity")]
    #[case(f64::NEG_INFINITY, "-Infinity")]
    #[case(-0.5, "-0.5")]
    fn test_double(#[case] input: f64, #[case] expected: &str) {
        assert_eq!(display(Value::from(input)), expected);
    }

    #[rstest]
    #[case(0.0f32, "0")]
    #[case(0.1f32, "0.1")]
    #[case(f32::NAN, "NaN")]
    #[case(f32::INFINITY, "Infinity")]
    #[case(f32::NEG_INFINITY, "-Infinity")]
    fn test_float(#[case] input: f32, #[case] expected: &str) {
        assert_eq!(display(Value::from(input)), expected);
    }

    #[rstest]
    #[case("hello", "hello")]
    #[case("", "(empty)")]
    #[case("trailing\n", "trailing")]
    fn test_varchar(#[case] input: &str, #[case] expected: &str) {
        assert_eq!(display(Value::from(input)), expected);
    }

    #[rstest]
    #[case(12345, 2, "123.45")]
    #[case(100, 0, "100")]
    #[case(1, 1, "0.1")]
    #[case(-12345, 2, "-123.45")]
    fn test_decimal(#[case] value: i128, #[case] scale: i8, #[case] expected: &str) {
        assert_eq!(display(Value::new_decimal(18, scale, value)), expected);
    }
}
