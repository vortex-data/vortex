// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::aggregate_fn::AggregateFnRef;
use vortex::array::aggregate_fn::AggregateFnVTableExt;
use vortex::array::aggregate_fn::EmptyOptions;
use vortex::array::aggregate_fn::NumericalAggregateOpts;
use vortex::array::aggregate_fn::fns::max::Max;
use vortex::array::aggregate_fn::fns::min::Min;
use vortex::array::aggregate_fn::fns::null_count::NullCount;
use vortex::array::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use vortex::array::stats::StatsSet;
use vortex::dtype::DType;
use vortex::error::VortexExpect as _;
use vortex::expr::stats::Precision;
use vortex::expr::stats::Stat;
use vortex::scalar::Scalar;
use vortex::scalar::ScalarValue;

use crate::convert::ToDuckDBScalar as _;
use crate::duckdb::Value;

const MIN_INDEX: usize = 0;
const MAX_INDEX: usize = 1;
const NULL_COUNT_INDEX: usize = 2;
const BYTE_SIZE_INDEX: usize = 3;

pub fn column_statistics_aggregate_fns() -> Vec<AggregateFnRef> {
    vec![
        Min.bind(NumericalAggregateOpts::default()),
        Max.bind(NumericalAggregateOpts::default()),
        NullCount.bind(EmptyOptions),
        UncompressedSizeInBytes.bind(EmptyOptions),
    ]
}

#[derive(Debug, Default)]
pub struct ColumnStatistics {
    pub min: Option<Value>,
    pub max: Option<Value>,
    pub max_string_length: u64,
    pub has_null: bool,
}

impl ColumnStatistics {
    pub fn from(stats: &ColumnStatisticsAggregate, dtype: DType) -> Self {
        let min = stats.min.as_ref().map(|value| {
            let value = value.clone();
            Scalar::try_new(dtype.clone(), Some(value))
                .vortex_expect("scalar dtype and value are incompatible")
                .try_to_duckdb_scalar()
                .vortex_expect("can't convert Scalar to duckdb Value")
        });
        let max = stats.max.as_ref().map(|value| {
            Scalar::try_new(dtype.clone(), Some(value.clone()))
                .vortex_expect("scalar dtype and value are incompatible")
                .try_to_duckdb_scalar()
                .vortex_expect("can't convert Scalar to duckdb Value")
        });

        let max_string_length = stats
            .max_string_length
            .map_or(0, |len| (1u64 << 63) | (len as u64));

        // Useful estimate if we didn't get null count stats
        let has_null = stats.has_null && dtype.is_nullable();

        Self {
            min,
            max,
            max_string_length,
            has_null,
        }
    }
}

#[derive(Clone, Default)]
pub struct ColumnStatisticsAggregate {
    pub min: Option<ScalarValue>,
    pub max: Option<ScalarValue>,
    pub max_string_length: Option<u32>,
    /// May be true if null count stat isn't present
    pub has_null: bool,
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

        Self {
            min,
            max,
            max_string_length,
            has_null,
        }
    }

    pub fn from_aggregate_stats(stats: &[Precision<Scalar>]) -> Self {
        let min = exact_scalar_value(stats.get(MIN_INDEX));
        let max = exact_scalar_value(stats.get(MAX_INDEX));
        let max_string_length = stats
            .get(BYTE_SIZE_INDEX)
            .and_then(exact_scalar_u64)
            .map(|value| u32::try_from(value).unwrap_or(u32::MAX));
        let has_null = stats
            .get(NULL_COUNT_INDEX)
            .and_then(exact_scalar_u64)
            .is_none_or(|count| count > 0);

        Self {
            min,
            max,
            max_string_length,
            has_null,
        }
    }
}

fn exact_scalar_value(stat: Option<&Precision<Scalar>>) -> Option<ScalarValue> {
    match stat {
        Some(Precision::Exact(value)) => value.clone().into_value(),
        _ => None,
    }
}

fn exact_scalar_u64(stat: &Precision<Scalar>) -> Option<u64> {
    match stat {
        Precision::Exact(value) => value.as_primitive().typed_value::<u64>(),
        _ => None,
    }
}
