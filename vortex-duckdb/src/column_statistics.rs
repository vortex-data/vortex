// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::aggregate_fn::AggregateFnVTableExt;
use vortex::array::aggregate_fn::EmptyOptions;
use vortex::array::aggregate_fn::NumericalAggregateOpts;
use vortex::array::aggregate_fn::fns::max::Max;
use vortex::array::aggregate_fn::fns::min::Min;
use vortex::array::aggregate_fn::fns::null_count::NullCount;
use vortex::array::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use vortex::array::stats::AggregateResults;
use vortex::dtype::DType;
use vortex::error::VortexExpect as _;
use vortex::error::VortexResult;
use vortex::expr::stats::Precision;
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
    pub logical_type: LogicalType,
}

impl ColumnStatistics {
    pub fn try_from(stats: ColumnStatisticsAggregate, dtype: DType) -> VortexResult<Self> {
        let to_value = |value: ScalarValue| {
            Scalar::try_new(dtype.clone(), Some(value))
                .and_then(|scalar| scalar.try_to_duckdb_scalar())
                .ok()
        };
        let min = stats.min.and_then(to_value);
        let max = stats.max.and_then(to_value);

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
}

impl ColumnStatisticsAggregate {
    pub fn new(stats: &AggregateResults) -> Self {
        let min = match stats.get(&Min.bind(NumericalAggregateOpts::skip_nans())) {
            Precision::Exact(min) => min.into_value(),
            _ => None,
        };
        let max = match stats.get(&Max.bind(NumericalAggregateOpts::skip_nans())) {
            Precision::Exact(max) => max.into_value(),
            _ => None,
        };

        let max_string_length = match stats
            .get(&UncompressedSizeInBytes.bind(EmptyOptions))
            .as_exact()
        {
            Some(value) => {
                // DuckDB's string length is u32
                #[allow(clippy::cast_possible_truncation)]
                let size = value.as_primitive().as_::<u64>().vortex_expect("not a u64") as u32;
                Some(size)
            }
            None => None,
        };

        let has_null = match stats.get(&NullCount.bind(EmptyOptions)) {
            Precision::Exact(cnt) => cnt.as_primitive().as_::<u64>().vortex_expect("not a u64") > 0,
            _ => true,
        };

        Self {
            min,
            max,
            max_string_length,
            has_null,
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex::array::aggregate_fn::AggregateFnVTableExt;
    use vortex::array::aggregate_fn::EmptyOptions;
    use vortex::array::aggregate_fn::NumericalAggregateOpts;
    use vortex::array::aggregate_fn::fns::max::Max;
    use vortex::array::aggregate_fn::fns::min::Min;
    use vortex::array::aggregate_fn::fns::null_count::NullCount;
    use vortex::array::stats::AggregateResults;
    use vortex::dtype::DType;
    use vortex::dtype::Nullability;
    use vortex::dtype::PType;
    use vortex::error::VortexResult;
    use vortex::expr::stats::Precision;
    use vortex::scalar::Scalar;

    use super::ColumnStatisticsAggregate;

    #[test]
    fn planning_uses_only_exact_results() -> VortexResult<()> {
        let results = AggregateResults::try_new(
            &DType::from(PType::I32),
            [
                (
                    Min.bind(NumericalAggregateOpts::skip_nans()),
                    Precision::Exact(Scalar::primitive(1i32, Nullability::Nullable)),
                ),
                (
                    Max.bind(NumericalAggregateOpts::skip_nans()),
                    Precision::Inexact(Scalar::primitive(9i32, Nullability::Nullable)),
                ),
                (NullCount.bind(EmptyOptions), Precision::Exact(0u64.into())),
            ],
        )?;
        let stats = ColumnStatisticsAggregate::new(&results);
        assert_eq!(stats.min, Some(1i32.into()));
        assert!(stats.max.is_none());
        assert!(!stats.has_null);

        let missing = ColumnStatisticsAggregate::new(&AggregateResults::default());
        assert!(missing.has_null);
        assert!(missing.min.is_none());
        assert!(missing.max.is_none());
        Ok(())
    }
}
