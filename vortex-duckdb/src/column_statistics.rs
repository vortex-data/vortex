// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::stats::StatsSet;
use vortex::dtype::DType;
use vortex::error::VortexExpect as _;
use vortex::expr::stats::Precision;
use vortex::expr::stats::Stat;
use vortex::scalar::Scalar;
use vortex::scalar::ScalarValue;

use crate::convert::ToDuckDBScalar as _;
use crate::duckdb::Value;

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

#[derive(Default)]
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
}
