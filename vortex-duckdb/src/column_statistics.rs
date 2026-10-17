// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::match_each_float_ptype;
use vortex::array::stats::StatsSet;
use vortex::dtype::DType;
use vortex::dtype::PType;
use vortex::error::VortexExpect as _;
use vortex::error::VortexResult;
use vortex::expr::stats::Precision;
use vortex::expr::stats::Stat;
use vortex::scalar::Scalar;
use vortex::scalar::ScalarValue;

use crate::convert::ToDuckDBScalar as _;
use crate::duckdb::LogicalType;
use crate::duckdb::Value;

#[derive(Debug)]
pub struct ColumnStatistics {
    pub min: Option<Value>,
    pub max: Option<Value>,
    pub max_string_length: u64,
    pub has_null: bool,
    /// The estimated number of distinct non-null values, or 0 when unknown.
    pub distinct_count: u64,
    pub logical_type: LogicalType,
}

impl ColumnStatistics {
    pub fn try_from(stats: ColumnStatisticsAggregate, dtype: DType) -> VortexResult<Self> {
        // DuckDB treats a column without a distinct count as having one distinct value per row,
        // which makes it overestimate joins on columns whose values repeat, such as foreign keys.
        let distinct_count = stats.distinct_count.unwrap_or(0);
        let to_value = |value: ScalarValue| {
            Scalar::try_new(dtype.clone(), Some(value))
                .and_then(|scalar| scalar.try_to_duckdb_scalar())
                .ok()
        };
        let min = stats.min.and_then(to_value);
        // Vortex's max skips NaN, while DuckDB orders NaN above every other value.
        let max = if !dtype.is_float() || stats.nan_count == Some(0) {
            stats.max.and_then(to_value)
        } else if stats.nan_count.is_some() {
            to_value(match_each_float_ptype!(dtype.as_ptype(), |F| {
                ScalarValue::from(F::NAN)
            }))
        } else {
            None
        };

        let max_string_length = stats
            .max_string_length
            .map_or(0, |len| (1u64 << 63) | (len as u64));

        // Useful estimate if we didn't get null count stats
        let has_null = stats.has_null && dtype.is_nullable();

        let logical_type = LogicalType::try_from(dtype)?;

        Ok(Self {
            min,
            max,
            max_string_length,
            has_null,
            distinct_count,
            logical_type,
        })
    }
}

#[derive(Default)]
pub struct ColumnStatisticsAggregate {
    pub min: Option<ScalarValue>,
    pub max: Option<ScalarValue>,
    pub max_string_length: Option<u32>,
    /// May be true if null count stat isn't present
    pub has_null: bool,
    pub nan_count: Option<u64>,
    /// The estimated number of distinct non-null values.
    pub distinct_count: Option<u64>,
}

impl ColumnStatisticsAggregate {
    pub fn new(stats: &StatsSet, distinct_count: Option<u64>, max_length: Option<u64>) -> Self {
        let min = match stats.get(Stat::Min) {
            Precision::Exact(min) => Some(min),
            _ => None,
        };
        let max = match stats.get(Stat::Max) {
            Precision::Exact(max) => Some(max),
            _ => None,
        };

        // DuckDB's string length is u32, and a longer one is no use to it.
        let max_string_length = max_length.and_then(|length| u32::try_from(length).ok());

        let has_null = match stats.get(Stat::NullCount) {
            Precision::Exact(cnt) => cnt.as_primitive().as_u64().vortex_expect("not a u64") > 0,
            _ => true,
        };

        let nan_count = stats
            .get_as::<u64>(Stat::NaNCount, &PType::U64.into())
            .as_exact();

        Self {
            min,
            max,
            max_string_length,
            has_null,
            nan_count,
            distinct_count,
        }
    }
}
