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
    use rstest::rstest;
    use vortex::aggregate_fn::AggregateFnVTableExt;
    use vortex::aggregate_fn::EmptyOptions;
    use vortex::aggregate_fn::NumericalAggregateOpts;
    use vortex::aggregate_fn::fns::max::Max;
    use vortex::aggregate_fn::fns::min::Min;
    use vortex::aggregate_fn::fns::nan_count::NanCount;
    use vortex::aggregate_fn::fns::null_count::NullCount;
    use vortex::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
    use vortex::dtype::DType;
    use vortex::dtype::Nullability;
    use vortex::dtype::PType;
    use vortex::error::VortexResult;
    use vortex::expr::stats::Precision;
    use vortex::layout::layouts::file_stats::AggregateStat;
    use vortex::layout::layouts::file_stats::AggregateStats;
    use vortex::scalar::Scalar;
    use vortex::scalar::ScalarValue;

    use super::ColumnStatistics;
    use super::ColumnStatisticsAggregate;
    use crate::cpp::DUCKDB_TYPE;
    use crate::duckdb::ExtractedValue;
    use crate::duckdb::Value;

    fn extract_f64(value: &Value) -> f64 {
        match value.extract() {
            ExtractedValue::Float(v) => f64::from(v),
            ExtractedValue::Double(v) => v,
            other => panic!("expected float value, got {other:?}"),
        }
    }

    #[test]
    fn try_from_integer_preserves_min_max_regardless_of_nan_count() -> VortexResult<()> {
        for nan_count in [None, Some(0), Some(3)] {
            let stats = ColumnStatistics::try_from(
                ColumnStatisticsAggregate {
                    min: Some(ScalarValue::from(-10i32)),
                    max: Some(ScalarValue::from(42i32)),
                    max_string_length: None,
                    has_null: false,
                    nan_count,
                },
                DType::Primitive(PType::I32, Nullability::Nullable),
            )?;

            assert!(matches!(
                stats.min.as_ref().map(|v| v.extract()),
                Some(ExtractedValue::Integer(-10))
            ));
            assert!(matches!(
                stats.max.as_ref().map(|v| v.extract()),
                Some(ExtractedValue::Integer(42))
            ));
            assert_eq!(stats.max_string_length, 0);
            assert!(!stats.has_null);
            assert_eq!(
                stats.logical_type.as_type_id(),
                DUCKDB_TYPE::DUCKDB_TYPE_INTEGER
            );
        }
        Ok(())
    }

    #[rstest]
    #[case::f32(PType::F32, ScalarValue::from(-1.5f32), ScalarValue::from(3.5f32))]
    #[case::f64(PType::F64, ScalarValue::from(-1.5f64), ScalarValue::from(3.5f64))]
    fn try_from_float_nan_count_controls_max_bound(
        #[case] ptype: PType,
        #[case] min: ScalarValue,
        #[case] max: ScalarValue,
    ) -> VortexResult<()> {
        let dtype = DType::Primitive(ptype, Nullability::Nullable);

        // Zero NaNs preserves the finite max bound.
        let no_nans = ColumnStatistics::try_from(
            ColumnStatisticsAggregate {
                min: Some(min.clone()),
                max: Some(max.clone()),
                nan_count: Some(0),
                ..Default::default()
            },
            dtype.clone(),
        )?;
        assert_eq!(extract_f64(no_nans.min.as_ref().unwrap()), -1.5);
        assert_eq!(extract_f64(no_nans.max.as_ref().unwrap()), 3.5);

        // Any NaNs (with or without a finite max) sets max to NaN because DuckDB orders NaN highest.
        for finite_max in [Some(max.clone()), None] {
            let with_nans = ColumnStatistics::try_from(
                ColumnStatisticsAggregate {
                    min: Some(min.clone()),
                    max: finite_max,
                    nan_count: Some(2),
                    ..Default::default()
                },
                dtype.clone(),
            )?;
            assert_eq!(extract_f64(with_nans.min.as_ref().unwrap()), -1.5);
            assert!(extract_f64(with_nans.max.as_ref().unwrap()).is_nan());
        }

        // Unknown nan_count drops max conservatively.
        let unknown_nans = ColumnStatistics::try_from(
            ColumnStatisticsAggregate {
                min: Some(min),
                max: Some(max),
                nan_count: None,
                ..Default::default()
            },
            dtype,
        )?;
        assert_eq!(extract_f64(unknown_nans.min.as_ref().unwrap()), -1.5);
        assert!(unknown_nans.max.is_none());
        Ok(())
    }

    #[rstest]
    #[case::absent(None, 0)]
    #[case::zero(Some(0), 1u64 << 63)]
    #[case::typical(Some(42), (1u64 << 63) | 42)]
    #[case::u32_max(Some(u32::MAX), (1u64 << 63) | u64::from(u32::MAX))]
    fn try_from_max_string_length_bitmask(
        #[case] max_string_length: Option<u32>,
        #[case] expected: u64,
    ) -> VortexResult<()> {
        let stats = ColumnStatistics::try_from(
            ColumnStatisticsAggregate {
                max_string_length,
                ..Default::default()
            },
            DType::Utf8(Nullability::Nullable),
        )?;

        assert_eq!(stats.max_string_length, expected);
        assert_eq!(
            stats.logical_type.as_type_id(),
            DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR
        );
        Ok(())
    }

    #[rstest]
    #[case::has_null_nullable(true, Nullability::Nullable, true)]
    #[case::has_null_non_nullable(true, Nullability::NonNullable, false)]
    #[case::no_null_nullable(false, Nullability::Nullable, false)]
    #[case::no_null_non_nullable(false, Nullability::NonNullable, false)]
    fn try_from_has_null_respects_dtype_nullability(
        #[case] has_null: bool,
        #[case] nullability: Nullability,
        #[case] expected: bool,
    ) -> VortexResult<()> {
        let stats = ColumnStatistics::try_from(
            ColumnStatisticsAggregate {
                has_null,
                ..Default::default()
            },
            DType::Primitive(PType::I64, nullability),
        )?;

        assert_eq!(stats.has_null, expected);
        Ok(())
    }

    #[test]
    fn try_from_drops_mismatched_bound_scalar_and_errors_on_unsupported_dtype() -> VortexResult<()>
    {
        let stats = ColumnStatistics::try_from(
            ColumnStatisticsAggregate {
                min: Some(ScalarValue::from("not an int")),
                max: Some(ScalarValue::from("still not an int")),
                ..Default::default()
            },
            DType::Primitive(PType::I32, Nullability::Nullable),
        )?;
        assert!(stats.min.is_none());
        assert!(stats.max.is_none());

        let variant_err = ColumnStatistics::try_from(
            ColumnStatisticsAggregate::default(),
            DType::Variant(Nullability::Nullable),
        )
        .unwrap_err();
        assert!(
            variant_err.to_string().contains("Variant"),
            "unexpected error: {variant_err}"
        );
        let f16_err = ColumnStatistics::try_from(
            ColumnStatisticsAggregate {
                nan_count: Some(1),
                ..Default::default()
            },
            DType::Primitive(PType::F16, Nullability::Nullable),
        )
        .unwrap_err();
        assert!(
            f16_err.to_string().contains("F16"),
            "unexpected error: {f16_err}"
        );
        Ok(())
    }

    #[test]
    fn aggregate_new_empty_falls_back_to_has_null() {
        let agg = ColumnStatisticsAggregate::new(&AggregateStats::default());
        assert!(agg.min.is_none());
        assert!(agg.max.is_none());
        assert!(agg.max_string_length.is_none());
        assert!(agg.has_null);
        assert!(agg.nan_count.is_none());
    }

    #[test]
    fn aggregate_new_reads_exact_stats_and_ignores_inexact() {
        let exact = AggregateStats::new(vec![
            AggregateStat::from_value(
                Min.bind(NumericalAggregateOpts::skip_nans()),
                Precision::exact(Scalar::from(1.25f64)),
            ),
            AggregateStat::from_value(
                Max.bind(NumericalAggregateOpts::skip_nans()),
                Precision::exact(Scalar::from(9.5f64)),
            ),
            AggregateStat::from_value(
                UncompressedSizeInBytes.bind(EmptyOptions),
                Precision::exact(Scalar::from(128u64)),
            ),
            AggregateStat::from_value(
                NullCount.bind(EmptyOptions),
                Precision::exact(Scalar::from(0u64)),
            ),
            AggregateStat::from_value(
                NanCount.bind(EmptyOptions),
                Precision::exact(Scalar::from(2u64)),
            ),
        ]);
        let agg = ColumnStatisticsAggregate::new(&exact);
        assert_eq!(agg.min, Some(ScalarValue::from(1.25f64)));
        assert_eq!(agg.max, Some(ScalarValue::from(9.5f64)));
        assert_eq!(agg.max_string_length, Some(128));
        assert!(!agg.has_null);
        assert_eq!(agg.nan_count, Some(2));

        let inexact = AggregateStats::new(vec![
            AggregateStat::from_value(
                Min.bind(NumericalAggregateOpts::skip_nans()),
                Precision::inexact(Scalar::from(1.25f64)),
            ),
            AggregateStat::from_value(
                Max.bind(NumericalAggregateOpts::skip_nans()),
                Precision::inexact(Scalar::from(9.5f64)),
            ),
            AggregateStat::from_value(
                UncompressedSizeInBytes.bind(EmptyOptions),
                Precision::inexact(Scalar::from(128u64)),
            ),
            AggregateStat::from_value(
                NullCount.bind(EmptyOptions),
                Precision::inexact(Scalar::from(0u64)),
            ),
            AggregateStat::from_value(
                NanCount.bind(EmptyOptions),
                Precision::inexact(Scalar::from(0u64)),
            ),
        ]);
        let agg = ColumnStatisticsAggregate::new(&inexact);
        assert!(agg.min.is_none());
        assert!(agg.max.is_none());
        assert!(agg.max_string_length.is_none());
        // Inexact null count must fall back to assuming nulls may be present.
        assert!(agg.has_null);
        assert!(agg.nan_count.is_none());
    }

    #[test]
    fn aggregate_new_handles_nonzero_null_count_and_oversized_string_length() {
        let stats = AggregateStats::new(vec![
            AggregateStat::from_value(
                NullCount.bind(EmptyOptions),
                Precision::exact(Scalar::from(7u64)),
            ),
            AggregateStat::from_value(
                UncompressedSizeInBytes.bind(EmptyOptions),
                Precision::exact(Scalar::from(u64::from(u32::MAX) + 1)),
            ),
        ]);
        let agg = ColumnStatisticsAggregate::new(&stats);
        assert!(agg.has_null);
        assert!(agg.max_string_length.is_none());
    }
}
