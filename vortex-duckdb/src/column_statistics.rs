// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex::array::aggregate_fn::AggregateFnRef;
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
use vortex::scalar::Scalar;
use vortex::scalar::ScalarValue;

use crate::convert::ToDuckDBScalar as _;
use crate::duckdb::LogicalType;
use crate::duckdb::Value;

static MIN: LazyLock<AggregateFnRef> =
    LazyLock::new(|| Min.bind(NumericalAggregateOpts::skip_nans()));
static MAX: LazyLock<AggregateFnRef> =
    LazyLock::new(|| Max.bind(NumericalAggregateOpts::skip_nans()));
static NULL_COUNT: LazyLock<AggregateFnRef> = LazyLock::new(|| NullCount.bind(EmptyOptions));
static UNCOMPRESSED_SIZE: LazyLock<AggregateFnRef> =
    LazyLock::new(|| UncompressedSizeInBytes.bind(EmptyOptions));

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
        let min = stats
            .get_result(&MIN)
            .as_exact()
            .and_then(|v| v.value().cloned());
        let max = stats
            .get_result(&MAX)
            .as_exact()
            .and_then(|v| v.value().cloned());

        let max_string_length = stats
            .get_result(&UNCOMPRESSED_SIZE)
            .as_exact()
            .map(|value| {
                // DuckDB's string length is u32.
                #[allow(clippy::cast_possible_truncation)]
                {
                    u64::try_from(&value).vortex_expect("not a u64") as u32
                }
            });
        let has_null = stats
            .get_result(&NULL_COUNT)
            .as_exact()
            .is_none_or(|value| u64::try_from(&value).vortex_expect("not a u64") > 0);

        Self {
            min,
            max,
            max_string_length,
            has_null,
        }
    }
}
