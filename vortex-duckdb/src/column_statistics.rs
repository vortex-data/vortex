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
use vortex::array::match_each_float_ptype;
use vortex::dtype::Claim;
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
    pub logical_type: LogicalType,
}

impl ColumnStatistics {
    pub fn try_from(stats: ColumnStatisticsAggregate, dtype: DType) -> VortexResult<Self> {
        let to_value = |value: ScalarValue| {
            Scalar::try_new(dtype.clone(), Some(value))
                .and_then(|scalar| scalar.try_to_duckdb_scalar())
                .ok()
        };
        let (min, max) = if dtype.holds(Claim::Ord) {
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
            (min, max)
        } else {
            (None, None)
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
}

impl ColumnStatisticsAggregate {
    pub fn new(aggregates: &AggregateStats) -> Self {
        let exact = |aggregate_fn: AggregateFnRef| match aggregates.get(&aggregate_fn) {
            Precision::Exact(value) => value.into_value(),
            _ => None,
        };

        let min = exact(Min.bind(NumericalAggregateOpts::skip_nans()));
        let max = exact(Max.bind(NumericalAggregateOpts::skip_nans()));

        // DuckDB's string length is u32
        let max_string_length =
            exact(UncompressedSizeInBytes.bind(EmptyOptions)).and_then(|value| {
                u32::try_from(value.as_primitive().as_u64().vortex_expect("not a u64")).ok()
            });

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
        }
    }
}

#[cfg(test)]
mod tests {
    use geo_types::Point;
    use vortex::dtype::Nullability;
    use vortex::dtype::extension::ExtDType;
    use vortex::error::vortex_err;
    use vortex::layout::layouts::file_stats::AggregateStat;
    use vortex_spatial::extension::SpatialMetadata;
    use vortex_spatial::extension::WellKnownBinary;
    use wkb::writer::WriteOptions;
    use wkb::writer::write_point;

    use super::*;

    #[test]
    fn column_statistics_skip_types_without_ord() -> VortexResult<()> {
        let mut wkb = Vec::new();
        write_point(&mut wkb, &Point::new(1.0, 2.0), &WriteOptions::default())
            .map_err(|error| vortex_err!("writing WKB point: {error}"))?;
        let ext_dtype = ExtDType::<WellKnownBinary>::try_new(
            SpatialMetadata::new(None),
            Nullability::NonNullable,
        )?
        .erased();
        let value = Scalar::try_extension_ref(
            ext_dtype.clone(),
            Scalar::binary(wkb, Nullability::NonNullable),
        )?;
        let aggregates = AggregateStats::new(vec![
            AggregateStat::from_value(
                Min.bind(NumericalAggregateOpts::skip_nans()),
                Precision::exact(value.clone()),
            ),
            AggregateStat::from_value(
                Max.bind(NumericalAggregateOpts::skip_nans()),
                Precision::exact(value),
            ),
        ]);

        let stats = ColumnStatistics::try_from(
            ColumnStatisticsAggregate::new(&aggregates),
            DType::Extension(ext_dtype),
        )?;
        assert!(stats.min.is_none());
        assert!(stats.max.is_none());
        Ok(())
    }
}
