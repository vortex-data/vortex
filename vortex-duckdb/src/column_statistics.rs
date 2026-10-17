// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::aggregate_fn::AggregateFnRef;
use vortex::aggregate_fn::AggregateFnVTableExt;
use vortex::aggregate_fn::EmptyOptions;
use vortex::aggregate_fn::NumericalAggregateOpts;
use vortex::aggregate_fn::fns::max::Max;
use vortex::aggregate_fn::fns::min::Min;
use vortex::aggregate_fn::fns::nan_count::NanCount;
use vortex::aggregate_fn::fns::null_count::NullCount;
use vortex::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use vortex::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeOpts;
use vortex::array::match_each_float_ptype;
use vortex::dtype::DType;
use vortex::error::VortexExpect as _;
use vortex::error::VortexResult;
use vortex::expr::stats::Precision;
use vortex::layout::layouts::file_stats::AggregateStats;
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
    /// An estimate of the number of distinct values, or 0 when unknown.
    pub distinct_count: u64,
    pub logical_type: LogicalType,
}

impl ColumnStatistics {
    /// Converts the statistics of a column of `row_count` rows.
    pub fn try_from(
        stats: ColumnStatisticsAggregate,
        dtype: DType,
        row_count: u64,
    ) -> VortexResult<Self> {
        let distinct_count = reported_distinct_count(stats.distinct_count, row_count);
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

/// The distinct count to give DuckDB for a column of `row_count` rows with `distinct_count`
/// estimated distinct values, or 0 to leave it unknown.
///
/// DuckDB treats a column without a distinct count as having one distinct value per row left
/// after filters, which makes it overestimate joins on foreign keys, whose values repeat. For a
/// column that is unique, or close to it, such as the key of a dimension table, that default is
/// exact even after filters, where the distinct count of the whole column would make DuckDB
/// underestimate joins with a filtered dimension table.
fn reported_distinct_count(distinct_count: Option<u64>, row_count: u64) -> u64 {
    // Distinct counts are sketched with an error of about 1%.
    let near_unique = row_count - row_count / 20;
    distinct_count
        .filter(|&count| count > 0 && count < near_unique)
        .unwrap_or(0)
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
    pub fn new(aggregates: &AggregateStats, distinct_count: Option<u64>) -> Self {
        let exact = |aggregate_fn: AggregateFnRef| match aggregates.get(&aggregate_fn) {
            Precision::Exact(value) => value.into_value(),
            _ => None,
        };

        let min = exact(Min.bind(NumericalAggregateOpts::skip_nans()));
        let max = exact(Max.bind(NumericalAggregateOpts::skip_nans()));

        // DuckDB's string length is u32
        #[allow(clippy::cast_possible_truncation)]
        let max_string_length = exact(UncompressedSizeInBytes.bind(UncompressedSizeOpts::exact()))
            .map(|value| value.as_primitive().as_u64().vortex_expect("not a u64") as u32);

        let has_null = exact(NullCount.bind(EmptyOptions))
            .is_none_or(|count| count.as_primitive().as_u64().vortex_expect("not a u64") > 0);

        let nan_count = exact(NanCount.bind(EmptyOptions))
            .map(|count| count.as_primitive().as_u64().vortex_expect("not a u64"));

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

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::reported_distinct_count;

    #[rstest]
    // A foreign key repeats its values.
    #[case(Some(1_823), 28_800_991, 1_823)]
    #[case(Some(12), 1_000, 12)]
    // A unique column, as estimated by a sketch, is left to DuckDB's default.
    #[case(Some(73_049), 73_049, 0)]
    #[case(Some(73_300), 73_049, 0)]
    #[case(Some(70_000), 73_049, 0)]
    #[case(None, 100, 0)]
    #[case(Some(0), 100, 0)]
    fn reports_distinct_counts_of_repeating_columns(
        #[case] distinct_count: Option<u64>,
        #[case] row_count: u64,
        #[case] expected: u64,
    ) {
        assert_eq!(reported_distinct_count(distinct_count, row_count), expected);
    }
}
