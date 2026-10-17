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
        let distinct_count = integer_distinct_count(&stats, &dtype, row_count);
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

/// Estimates the distinct values of an integer column, including dates and times stored as
/// integers, as the number of integers between its minimum and maximum, or 0 when that does not
/// show fewer distinct values than rows.
///
/// DuckDB treats a column without a distinct count as having one distinct value per row left
/// after filters, which makes it overestimate joins on foreign keys, whose values span a much
/// smaller range than the rows referencing them. For a column that may be unique, such as the key
/// of a dimension table, that default is exact even after filters, where the distinct count of
/// the whole column would make DuckDB underestimate joins with a filtered dimension table.
fn integer_distinct_count(stats: &ColumnStatisticsAggregate, dtype: &DType, row_count: u64) -> u64 {
    let storage = match dtype {
        DType::Extension(ext) => ext.storage_dtype(),
        dtype => dtype,
    };
    let DType::Primitive(ptype, _) = storage else {
        return 0;
    };
    if !ptype.is_int() {
        return 0;
    }
    let (Some(ScalarValue::Primitive(min)), Some(ScalarValue::Primitive(max))) =
        (&stats.min, &stats.max)
    else {
        return 0;
    };
    let span = if ptype.is_unsigned_int() {
        match (min.cast::<u64>(), max.cast::<u64>()) {
            (Ok(min), Ok(max)) => i128::from(max) - i128::from(min),
            _ => return 0,
        }
    } else {
        match (min.cast::<i64>(), max.cast::<i64>()) {
            (Ok(min), Ok(max)) => i128::from(max) - i128::from(min),
            _ => return 0,
        }
    };
    if span < 0 {
        return 0;
    }
    u64::try_from(span + 1)
        .ok()
        .filter(|&values| values < row_count)
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
}

impl ColumnStatisticsAggregate {
    pub fn new(stats: &StatsSet) -> Self {
        let min = match stats.get(Stat::Min) {
            Precision::Exact(min) => Some(min),
            _ => None,
        };
        let max = match stats.get(Stat::Max) {
            Precision::Exact(max) => Some(max),
            _ => None,
        };

        let max_string_length =
            if let Precision::Exact(value) = stats.get(Stat::UncompressedSizeInBytes) {
                // DuckDB's string length is u32
                #[allow(clippy::cast_possible_truncation)]
                Some(value.as_primitive().as_u64().vortex_expect("not a u64") as u32)
            } else {
                None
            };

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
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex::dtype::DType;
    use vortex::dtype::Nullability;
    use vortex::dtype::PType;
    use vortex::scalar::ScalarValue;

    use super::ColumnStatisticsAggregate;
    use super::integer_distinct_count;

    fn stats(min: ScalarValue, max: ScalarValue) -> ColumnStatisticsAggregate {
        ColumnStatisticsAggregate {
            min: Some(min),
            max: Some(max),
            ..Default::default()
        }
    }

    #[rstest]
    // A foreign key spans fewer values than it has rows.
    #[case(PType::I64, 2_450_816i64.into(), 2_452_642i64.into(), 28_800_991, 1_827)]
    #[case(PType::U32, 1u32.into(), 12u32.into(), 1_000, 12)]
    // A column that may be unique is left to DuckDB's default.
    #[case(PType::I32, 1i32.into(), 73_049i32.into(), 73_049, 0)]
    #[case(PType::I32, 1i32.into(), 1_000_000i32.into(), 100, 0)]
    // Floats have no countable range.
    #[case(PType::F64, 1.0f64.into(), 2.0f64.into(), 100, 0)]
    fn distinct_count_from_range(
        #[case] ptype: PType,
        #[case] min: ScalarValue,
        #[case] max: ScalarValue,
        #[case] row_count: u64,
        #[case] expected: u64,
    ) {
        let dtype = DType::Primitive(ptype, Nullability::Nullable);
        assert_eq!(
            integer_distinct_count(&stats(min, max), &dtype, row_count),
            expected
        );
    }

    #[test]
    fn distinct_count_without_bounds_is_unknown() {
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let stats = ColumnStatisticsAggregate::default();
        assert_eq!(integer_distinct_count(&stats, &dtype, 100), 0);
    }
}
