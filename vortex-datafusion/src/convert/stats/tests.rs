// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex::array::stats::StatsSet;
use vortex::array::stats::compat::legacy_stats_to_results;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::expr::stats::Stat;

use super::*;

#[rstest]
#[case::constant(VortexPrecision::Exact(true), Precision::Exact(1))]
#[case::non_constant(VortexPrecision::Exact(false), Precision::Absent)]
#[case::inexact(VortexPrecision::Inexact(true), Precision::Absent)]
#[case::missing(VortexPrecision::Absent, Precision::Absent)]
fn constant_distinct_count(
    #[case] constant: VortexPrecision<bool>,
    #[case] expected: Precision<usize>,
) -> VortexResult<()> {
    let dtype = DType::Bool(Nullability::NonNullable);
    let results =
        AggregateResults::try_new(&dtype, [(IS_CONSTANT.clone(), constant.map(Scalar::from))])?;

    assert_eq!(
        aggregate_results_to_df(&results, &dtype)?.distinct_count,
        expected
    );
    Ok(())
}

#[rstest]
#[case::exact(
    VortexPrecision::Exact(7),
    Precision::Exact(ScalarValue::Int32(Some(7)))
)]
#[case::inexact(
    VortexPrecision::Inexact(7),
    Precision::Inexact(ScalarValue::Int32(Some(7)))
)]
#[case::missing(VortexPrecision::Absent, Precision::Absent)]
fn extrema_keep_precision(
    #[case] min: VortexPrecision<i32>,
    #[case] expected: Precision<ScalarValue>,
) -> VortexResult<()> {
    let dtype = DType::from(PType::I32);
    let results = AggregateResults::try_new(
        &dtype,
        [(
            MIN.clone(),
            min.map(|value| Scalar::primitive(value, Nullability::Nullable)),
        )],
    )?;
    let stats = aggregate_results_to_df(&results, &dtype)?;

    assert_eq!(stats.min_value, expected);
    assert_eq!(stats.max_value, Precision::Absent);
    assert_eq!(stats.null_count, Precision::Absent);
    Ok(())
}

#[test]
fn historical_non_nullable_extrema_keep_precision() -> VortexResult<()> {
    let dtype = DType::from(PType::I32);
    let legacy = StatsSet::of(Stat::Min, VortexPrecision::Inexact(7i32.into()));
    let results = legacy_stats_to_results(&dtype, &legacy)?;

    assert_eq!(
        results
            .get_result(&MIN)
            .into_inner()
            .map(|value| value.dtype().clone()),
        Some(dtype.clone())
    );
    assert_eq!(
        aggregate_results_to_df(&results, &dtype)?.min_value,
        Precision::Inexact(ScalarValue::Int32(Some(7)))
    );
    Ok(())
}

#[test]
fn exact_null_sum_differs_from_missing_metadata() -> VortexResult<()> {
    let dtype = DType::from(PType::I64);
    let results = AggregateResults::try_new(
        &dtype,
        [(
            SUM.clone(),
            VortexPrecision::Exact(Scalar::null(dtype.as_nullable())),
        )],
    )?;
    let stats = aggregate_results_to_df(&results, &dtype)?;

    assert_eq!(stats.sum_value, Precision::Exact(ScalarValue::Int64(None)));
    assert_eq!(stats.min_value, Precision::Absent);
    assert_eq!(
        aggregate_results_to_df(&AggregateResults::default(), &dtype)?.sum_value,
        Precision::Absent
    );
    Ok(())
}

#[rstest]
#[case::exact(VortexPrecision::Exact(Scalar::null(DType::from(PType::I32).as_nullable())))]
#[case::inexact(VortexPrecision::Inexact(Scalar::null(DType::from(PType::I32).as_nullable())))]
fn null_extrema_do_not_provide_bounds(#[case] value: VortexPrecision<Scalar>) -> VortexResult<()> {
    let dtype = DType::from(PType::I32);
    let results =
        AggregateResults::try_new(&dtype, [(MIN.clone(), value.clone()), (MAX.clone(), value)])?;
    let stats = aggregate_results_to_df(&results, &dtype)?;

    assert_eq!(stats.min_value, Precision::Absent);
    assert_eq!(stats.max_value, Precision::Absent);
    Ok(())
}

#[rstest]
#[case::null_column(DType::Null)]
#[case::wrong_type(DType::Bool(Nullability::NonNullable))]
fn incompatible_column_type_has_no_extrema(#[case] dtype: DType) -> VortexResult<()> {
    let results = AggregateResults::try_new(
        &DType::from(PType::I32),
        [(
            MIN.clone(),
            VortexPrecision::Exact(Scalar::primitive(7i32, Nullability::Nullable)),
        )],
    )?;

    assert_eq!(
        aggregate_results_to_df(&results, &dtype)?.min_value,
        Precision::Absent
    );
    Ok(())
}

#[test]
fn unsupported_column_type_has_no_sum() -> VortexResult<()> {
    let results = AggregateResults::try_new(
        &DType::from(PType::I32),
        [(
            SUM.clone(),
            VortexPrecision::Exact(Scalar::primitive(7i64, Nullability::Nullable)),
        )],
    )?;

    assert_eq!(
        aggregate_results_to_df(&results, &DType::Utf8(Nullability::NonNullable))?.sum_value,
        Precision::Absent
    );
    Ok(())
}

#[test]
fn including_nan_results_do_not_satisfy_skipping_nan_requests() -> VortexResult<()> {
    let dtype = DType::from(PType::F64);
    let options = NumericalAggregateOpts::include_nans();
    let nan = VortexPrecision::Exact(Scalar::primitive(f64::NAN, Nullability::Nullable));
    let results = AggregateResults::try_new(
        &dtype,
        [
            (Min.bind(options), nan.clone()),
            (Max.bind(options), nan.clone()),
            (Sum.bind(options), nan),
        ],
    )?;
    let stats = aggregate_results_to_df(&results, &dtype)?;

    assert_eq!(stats.min_value, Precision::Absent);
    assert_eq!(stats.max_value, Precision::Absent);
    assert_eq!(stats.sum_value, Precision::Absent);
    Ok(())
}

#[rstest]
#[case::zero(VortexPrecision::Exact(0), Precision::Exact(0))]
#[case::inexact(VortexPrecision::Inexact(7), Precision::Inexact(7))]
#[case::missing(VortexPrecision::Absent, Precision::Absent)]
fn count_and_size_keep_precision(
    #[case] value: VortexPrecision<u64>,
    #[case] expected: Precision<usize>,
) -> VortexResult<()> {
    let dtype = DType::from(PType::I32);
    let scalar = value.map(Scalar::from);
    let results = AggregateResults::try_new(
        &dtype,
        [
            (NULL_COUNT.clone(), scalar.clone()),
            (UNCOMPRESSED_SIZE.clone(), scalar),
        ],
    )?;
    let stats = aggregate_results_to_df(&results, &dtype)?;

    assert_eq!(stats.null_count, expected);
    assert_eq!(stats.byte_size, expected);
    Ok(())
}
