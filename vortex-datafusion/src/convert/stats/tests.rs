// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use datafusion_common::ScalarValue as DFScalarValue;
use rstest::rstest;
use vortex::VortexSessionDefault;
use vortex::dtype::half::f16;
use vortex::error::vortex_err;
use vortex::expr::stats::Precision as VortexPrecision;
use vortex::scalar::ScalarValue;
use vortex::session::VortexSession;

use super::*;
use crate::convert::scalar_from_df;

fn bounds_stats(value: &DFScalarValue) -> VortexResult<(StatsSet, DType)> {
    let scalar = scalar_from_df(value, &VortexSession::default())?;
    let value = scalar
        .value()
        .cloned()
        .ok_or_else(|| vortex_err!(AssertionFailed: "expected non-null bounds"))?;
    let mut stats = StatsSet::of(Stat::Min, VortexPrecision::exact(value.clone()));
    stats.set(Stat::Max, VortexPrecision::exact(value));
    stats.set(Stat::NullCount, VortexPrecision::exact(0u64));
    Ok((stats, scalar.dtype().clone()))
}

#[rstest]
#[case::boolean(DFScalarValue::Boolean(Some(true)))]
#[case::signed(DFScalarValue::Int32(Some(7)))]
#[case::unsigned(DFScalarValue::UInt64(Some(7)))]
#[case::string(DFScalarValue::Utf8(Some("TN".into())))]
#[case::binary(DFScalarValue::Binary(Some(vec![0, 255])))]
#[case::decimal(DFScalarValue::Decimal128(Some(123), 12, 2))]
#[case::date(DFScalarValue::Date32(Some(123)))]
#[case::time(DFScalarValue::Time64Microsecond(Some(123)))]
#[case::timestamp(DFScalarValue::TimestampNanosecond(Some(123), None))]
fn exact_non_float_bounds_prove_constant(#[case] value: DFScalarValue) -> VortexResult<()> {
    let (stats, dtype) = bounds_stats(&value)?;
    assert_eq!(
        stats_set_to_df(&stats, &dtype)?.distinct_count,
        Precision::Exact(1)
    );
    Ok(())
}

#[rstest]
#[case::no_nans(VortexPrecision::Exact(0u64), Precision::Exact(1))]
#[case::mixed_nans(VortexPrecision::Exact(1u64), Precision::Absent)]
#[case::inexact_nans(VortexPrecision::Inexact(0u64), Precision::Absent)]
#[case::unknown_nans(VortexPrecision::Absent, Precision::Absent)]
fn float_bounds_require_exact_zero_nan_count(
    #[values(
        DFScalarValue::Float16(Some(f16::from_f32(7.0))),
        DFScalarValue::Float32(Some(7.0)),
        DFScalarValue::Float64(Some(7.0))
    )]
    value: DFScalarValue,
    #[case] nan_count: VortexPrecision<u64>,
    #[case] expected: Precision<usize>,
) -> VortexResult<()> {
    let (mut stats, dtype) = bounds_stats(&value)?;
    stats.set(Stat::NaNCount, nan_count.map(ScalarValue::from));
    assert_eq!(stats_set_to_df(&stats, &dtype)?.distinct_count, expected);
    Ok(())
}

#[rstest]
fn unknown_statistics_cannot_prove_constant(
    #[values(Stat::Min, Stat::Max, Stat::NullCount)] stat: Stat,
    #[values(true, false)] missing: bool,
) -> VortexResult<()> {
    let (mut stats, dtype) = bounds_stats(&DFScalarValue::Int32(Some(7)))?;
    if missing {
        stats.clear(stat);
    } else {
        stats.set(stat, stats.get(stat).into_inexact());
    }
    assert_eq!(
        stats_set_to_df(&stats, &dtype)?.distinct_count,
        Precision::Absent
    );
    Ok(())
}

#[rstest]
#[case::nulls(Stat::NullCount, ScalarValue::from(1u64))]
#[case::different_bounds(Stat::Max, ScalarValue::from(8i32))]
fn non_constant_values_are_not_singletons(
    #[case] stat: Stat,
    #[case] value: ScalarValue,
) -> VortexResult<()> {
    let (mut stats, dtype) = bounds_stats(&DFScalarValue::Int32(Some(7)))?;
    stats.set(stat, VortexPrecision::exact(value));
    assert_eq!(
        stats_set_to_df(&stats, &dtype)?.distinct_count,
        Precision::Absent
    );
    Ok(())
}

#[test]
fn float_bounds_preserve_signed_zero_distinction() -> VortexResult<()> {
    let (mut stats, dtype) = bounds_stats(&DFScalarValue::Float64(Some(-0.0)))?;
    stats.set(Stat::Max, VortexPrecision::exact(0.0f64));
    stats.set(Stat::NaNCount, VortexPrecision::exact(0u64));
    assert_eq!(
        stats_set_to_df(&stats, &dtype)?.distinct_count,
        Precision::Absent
    );
    Ok(())
}

#[test]
fn is_constant_false_does_not_imply_one_distinct_value() -> VortexResult<()> {
    let false_constant = StatsSet::of(Stat::IsConstant, VortexPrecision::exact(false));
    let false_stats = stats_set_to_df(&false_constant, &DType::Bool(Nullability::NonNullable))?;

    assert_eq!(false_stats.distinct_count, Precision::Absent);

    let true_constant = StatsSet::of(Stat::IsConstant, VortexPrecision::exact(true));
    let true_stats = stats_set_to_df(&true_constant, &DType::Bool(Nullability::NonNullable))?;

    assert_eq!(true_stats.distinct_count, Precision::Exact(1));

    Ok(())
}

/// A statistic the column dtype cannot carry comes back `Absent`.
#[rstest]
#[case::min_of_null_column(Stat::Min, DType::Null, ScalarValue::from(1i32))]
#[case::max_of_null_column(Stat::Max, DType::Null, ScalarValue::from(1i32))]
#[case::sum_of_utf8_column(
    Stat::Sum,
    DType::Utf8(Nullability::NonNullable),
    ScalarValue::from(1i32)
)]
#[case::value_disagrees_with_dtype(
    Stat::Min,
    DType::Bool(Nullability::NonNullable),
    ScalarValue::from("not a bool")
)]
fn unconvertible_statistics_are_absent(
    #[case] stat: Stat,
    #[case] dtype: DType,
    #[case] value: ScalarValue,
) -> VortexResult<()> {
    let stats = stats_set_to_df(&StatsSet::of(stat, VortexPrecision::exact(value)), &dtype)?;

    for reported in [stats.min_value, stats.max_value, stats.sum_value] {
        assert_eq!(reported, Precision::Absent);
    }
    Ok(())
}

/// A statistic that does convert keeps its precision.
#[rstest]
#[case::exact(
    VortexPrecision::exact(ScalarValue::from(7i32)),
    Precision::Exact(DFScalarValue::Int32(Some(7)))
)]
#[case::inexact(
    VortexPrecision::inexact(ScalarValue::from(7i32)),
    Precision::Inexact(DFScalarValue::Int32(Some(7)))
)]
fn convertible_statistics_keep_their_precision(
    #[case] min: VortexPrecision<ScalarValue>,
    #[case] expected: Precision<DFScalarValue>,
) -> VortexResult<()> {
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let stats = stats_set_to_df(&StatsSet::of(Stat::Min, min), &dtype)?;

    assert_eq!(stats.min_value, expected);
    Ok(())
}

#[rstest]
#[case::nan_free(VortexPrecision::exact(0u64), true)]
#[case::has_nan(VortexPrecision::exact(1u64), false)]
#[case::inexact(VortexPrecision::inexact(0u64), false)]
#[case::unknown(VortexPrecision::Absent, false)]
fn float_statistics_require_no_nans(
    #[case] nan_count: VortexPrecision<u64>,
    #[case] reported: bool,
) -> VortexResult<()> {
    let stats_set = StatsSet::from_iter([
        (Stat::Min, VortexPrecision::exact(ScalarValue::from(1.0f64))),
        (Stat::Max, VortexPrecision::exact(ScalarValue::from(2.0f64))),
        (Stat::Sum, VortexPrecision::exact(ScalarValue::from(3.0f64))),
        (Stat::NaNCount, nan_count.map(ScalarValue::from)),
    ]);
    let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
    let stats = stats_set_to_df(&stats_set, &dtype)?;

    let expected = |value: f64| {
        if reported {
            Precision::Exact(DFScalarValue::Float64(Some(value)))
        } else {
            Precision::Absent
        }
    };
    assert_eq!(stats.min_value, expected(1.0));
    assert_eq!(stats.max_value, expected(2.0));
    assert_eq!(stats.sum_value, expected(3.0));
    Ok(())
}
