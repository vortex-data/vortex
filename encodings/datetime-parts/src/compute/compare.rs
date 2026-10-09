// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::extension::datetime::Timestamp;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::binary::CompareKernel;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use crate::array::DateTimeParts;
use crate::array::DateTimePartsArraySlotsExt;
use crate::timestamp;

impl CompareKernel for DateTimeParts {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(rhs_const) = rhs.as_constant() else {
            return Ok(None);
        };
        let Some(timestamp) = rhs_const
            .as_extension()
            .to_storage_scalar()
            .as_primitive()
            .as_::<i64>()
        else {
            return Ok(None);
        };

        let DType::Extension(ext_dtype) = rhs_const.dtype() else {
            return Ok(None);
        };

        let nullability = lhs.dtype().nullability() | rhs.dtype().nullability();

        let Some(options) = ext_dtype.metadata_opt::<Timestamp>() else {
            return Ok(None);
        };
        let ts_parts = timestamp::split(timestamp, options.unit)?;

        let result = match operator {
            CompareOperator::Eq => compare_eq(lhs, &ts_parts, nullability, ctx)?,
            CompareOperator::NotEq => compare_ne(lhs, &ts_parts, nullability, ctx)?,
            CompareOperator::Lt
            | CompareOperator::Lte
            | CompareOperator::Gt
            | CompareOperator::Gte => compare_ordered(lhs, &ts_parts, operator, nullability, ctx)?,
        };
        with_validity(lhs, result, nullability).map(Some)
    }
}

/// Restores the array's validity on a result combined from its parts.
///
/// The seconds and subseconds are non-nullable, so combining their comparisons with the nullable
/// days comparison under Kleene logic can turn a null row into `false` or `true`.
fn with_validity(
    lhs: ArrayView<DateTimeParts>,
    result: ArrayRef,
    nullability: Nullability,
) -> VortexResult<ArrayRef> {
    Ok(match lhs.validity()?.union_nullability(nullability) {
        Validity::NonNullable | Validity::AllValid => result,
        Validity::AllInvalid => {
            ConstantArray::new(Scalar::null(DType::Bool(nullability)), lhs.len()).into_array()
        }
        Validity::Array(validity) => result.mask(validity)?,
    })
}

fn compare_eq(
    lhs: ArrayView<DateTimeParts>,
    ts_parts: &timestamp::TimestampParts,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let mut comparison = compare_dtp(lhs.days(), ts_parts.days, CompareOperator::Eq, nullability)?;
    if comparison.statistics().compute_max::<bool>(ctx) == Some(false) {
        return Ok(comparison);
    }

    comparison = compare_dtp(
        lhs.seconds(),
        ts_parts.seconds,
        CompareOperator::Eq,
        nullability,
    )?
    .binary(comparison, Operator::And)?;

    if comparison.statistics().compute_max::<bool>(ctx) == Some(false) {
        return Ok(comparison);
    }

    compare_dtp(
        lhs.subseconds(),
        ts_parts.subseconds,
        CompareOperator::Eq,
        nullability,
    )?
    .binary(comparison, Operator::And)
}

fn compare_ne(
    lhs: ArrayView<DateTimeParts>,
    ts_parts: &timestamp::TimestampParts,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let mut comparison = compare_dtp(
        lhs.days(),
        ts_parts.days,
        CompareOperator::NotEq,
        nullability,
    )?;
    if comparison.statistics().compute_min::<bool>(ctx) == Some(true) {
        return Ok(comparison);
    }

    comparison = compare_dtp(
        lhs.seconds(),
        ts_parts.seconds,
        CompareOperator::NotEq,
        nullability,
    )?
    .binary(comparison, Operator::Or)?;

    if comparison.statistics().compute_min::<bool>(ctx) == Some(true) {
        return Ok(comparison);
    }

    compare_dtp(
        lhs.subseconds(),
        ts_parts.subseconds,
        CompareOperator::NotEq,
        nullability,
    )?
    .binary(comparison, Operator::Or)
}

/// Orders timestamps by their parts.
///
/// `split` gives every timestamp a unique `(days, seconds, subseconds)` triple whose remainders
/// share the timestamp's sign, so the parts compare lexicographically: the first part that
/// differs decides, and the last part carries the operator's strictness.
fn compare_ordered(
    lhs: ArrayView<DateTimeParts>,
    ts_parts: &timestamp::TimestampParts,
    operator: CompareOperator,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let strict = match operator {
        CompareOperator::Lt | CompareOperator::Lte => CompareOperator::Lt,
        CompareOperator::Gt | CompareOperator::Gte => CompareOperator::Gt,
        CompareOperator::Eq | CompareOperator::NotEq => {
            unreachable!("equality is answered by compare_eq and compare_ne")
        }
    };

    let days_strict = compare_dtp(lhs.days(), ts_parts.days, strict, nullability)?;
    if days_strict.statistics().compute_min::<bool>(ctx) == Some(true) {
        return Ok(days_strict);
    }
    let days_eq = compare_dtp(lhs.days(), ts_parts.days, CompareOperator::Eq, nullability)?;
    if days_eq.statistics().compute_max::<bool>(ctx) == Some(false) {
        return Ok(days_strict);
    }

    let seconds_strict = compare_dtp(lhs.seconds(), ts_parts.seconds, strict, nullability)?;
    let seconds_eq = compare_dtp(
        lhs.seconds(),
        ts_parts.seconds,
        CompareOperator::Eq,
        nullability,
    )?;
    let subseconds = compare_dtp(lhs.subseconds(), ts_parts.subseconds, operator, nullability)?;

    let within_day =
        seconds_strict.binary(seconds_eq.binary(subseconds, Operator::And)?, Operator::Or)?;
    days_strict.binary(days_eq.binary(within_day, Operator::And)?, Operator::Or)
}

fn compare_dtp(
    lhs: &ArrayRef,
    rhs: i32,
    operator: CompareOperator,
    nullability: Nullability,
) -> VortexResult<ArrayRef> {
    // Since nullability is stripped from RHS and carried forward through nullability argument we want to incorporate it into lhs.dtype() that we cast rhs into
    match Scalar::from(rhs).cast(&lhs.dtype().with_nullability(nullability)) {
        Ok(casted) => lhs.binary(
            ConstantArray::new(casted, lhs.len()).into_array(),
            Operator::from(operator),
        ),
        // The narrowing cast failed, so rhs is either > or < every value in lhs.
        // rhs positive => > all lhs
        // rhs negative => < all lhs
        _ => {
            let all_lhs_smaller = rhs > 0;
            let constant_value = match operator {
                CompareOperator::Eq => false,
                CompareOperator::NotEq => true,
                CompareOperator::Lt | CompareOperator::Lte => all_lhs_smaller,
                CompareOperator::Gt | CompareOperator::Gte => !all_lhs_smaller,
            };
            // If there are nulls in lhs, we need to propagate them
            let validity = match nullability {
                Nullability::NonNullable => Validity::NonNullable,
                // lhs may be non-nullable while rhs contributes the nullability.
                Nullability::Nullable => match lhs.validity()? {
                    Validity::NonNullable => Validity::AllValid,
                    validity => validity,
                },
            };
            Ok(match validity {
                Validity::NonNullable | Validity::AllValid => {
                    ConstantArray::new(Scalar::bool(constant_value, nullability), lhs.len())
                        .into_array()
                }
                Validity::AllInvalid => {
                    ConstantArray::new(Scalar::null(DType::Bool(nullability)), lhs.len())
                        .into_array()
                }
                Validity::Array(validity_array) => {
                    ConstantArray::new(Scalar::bool(constant_value, nullability), lhs.len())
                        .into_array()
                        .mask(validity_array)?
                }
            })
        }
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::ExecutionCtx;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::fns::sum::sum;
    use vortex_array::array_session;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::TemporalArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::IntegerPType;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::extension::datetime::Timestamp;
    use vortex_array::extension::datetime::TimestampOptions;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::fns::binary::CompareKernel;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    use super::*;
    use crate::DateTimeParts;
    use crate::DateTimePartsArray;

    fn dtp_array_from_timestamp<T: IntegerPType>(
        value: T,
        validity: Validity,
    ) -> DateTimePartsArray {
        DateTimeParts::try_from_temporal(
            TemporalArray::new_timestamp(
                PrimitiveArray::new(buffer![value], validity).into_array(),
                TimeUnit::Seconds,
                Some("UTC".into()),
            ),
            &mut array_session().create_execution_ctx(),
        )
        .expect("Failed to construct DateTimePartsArray from TemporalArray")
    }

    /// Count the true values in a boolean array using the provided execution context.
    fn true_count(array: &ArrayRef, ctx: &mut ExecutionCtx) -> usize {
        sum(array, ctx)
            .unwrap()
            .as_primitive()
            .as_::<usize>()
            .unwrap()
    }

    #[rstest]
    #[case(Validity::NonNullable, Validity::NonNullable)]
    #[case(Validity::NonNullable, Validity::AllValid)]
    #[case(Validity::AllValid, Validity::NonNullable)]
    #[case(Validity::AllValid, Validity::AllValid)]
    fn compare_date_time_parts_eq(#[case] lhs_validity: Validity, #[case] rhs_validity: Validity) {
        let mut ctx = array_session().create_execution_ctx();
        let lhs = dtp_array_from_timestamp(86400i64, lhs_validity); // January 2, 1970, 00:00:00 UTC
        let rhs = dtp_array_from_timestamp(86400i64, rhs_validity.clone()); // January 2, 1970, 00:00:00 UTC
        let comparison = lhs
            .clone()
            .into_array()
            .binary(rhs.into_array(), Operator::Eq)
            .unwrap();
        assert_eq!(true_count(&comparison, &mut ctx), 1);

        let rhs = dtp_array_from_timestamp(0i64, rhs_validity); // January 1, 1970, 00:00:00 UTC
        let comparison = lhs
            .into_array()
            .binary(rhs.into_array(), Operator::Eq)
            .unwrap();
        assert_eq!(true_count(&comparison, &mut ctx), 0);
    }

    #[rstest]
    #[case(Validity::NonNullable, Validity::NonNullable)]
    #[case(Validity::NonNullable, Validity::AllValid)]
    #[case(Validity::AllValid, Validity::NonNullable)]
    #[case(Validity::AllValid, Validity::AllValid)]
    fn compare_date_time_parts_ne(#[case] lhs_validity: Validity, #[case] rhs_validity: Validity) {
        let mut ctx = array_session().create_execution_ctx();
        let lhs = dtp_array_from_timestamp(86400i64, lhs_validity); // January 2, 1970, 00:00:00 UTC
        let rhs = dtp_array_from_timestamp(86401i64, rhs_validity.clone()); // January 2, 1970, 00:00:01 UTC
        let comparison = lhs
            .clone()
            .into_array()
            .binary(rhs.into_array(), Operator::NotEq)
            .unwrap();
        assert_eq!(true_count(&comparison, &mut ctx), 1);

        let rhs = dtp_array_from_timestamp(86400i64, rhs_validity); // January 2, 1970, 00:00:00 UTC
        let comparison = lhs
            .into_array()
            .binary(rhs.into_array(), Operator::NotEq)
            .unwrap();
        assert_eq!(true_count(&comparison, &mut ctx), 0);
    }

    #[rstest]
    #[case(Validity::NonNullable, Validity::NonNullable)]
    #[case(Validity::NonNullable, Validity::AllValid)]
    #[case(Validity::AllValid, Validity::NonNullable)]
    #[case(Validity::AllValid, Validity::AllValid)]
    fn compare_date_time_parts_lt(#[case] lhs_validity: Validity, #[case] rhs_validity: Validity) {
        let mut ctx = array_session().create_execution_ctx();
        let lhs = dtp_array_from_timestamp(0i64, lhs_validity); // January 1, 1970, 01:00:00 UTC
        let rhs = dtp_array_from_timestamp(86400i64, rhs_validity); // January 2, 1970, 00:00:00 UTC

        let comparison = lhs
            .into_array()
            .binary(rhs.into_array(), Operator::Lt)
            .unwrap();
        assert_eq!(true_count(&comparison, &mut ctx), 1);
    }

    #[rstest]
    #[case(Validity::NonNullable, Validity::NonNullable)]
    #[case(Validity::NonNullable, Validity::AllValid)]
    #[case(Validity::AllValid, Validity::NonNullable)]
    #[case(Validity::AllValid, Validity::AllValid)]
    fn compare_date_time_parts_gt(#[case] lhs_validity: Validity, #[case] rhs_validity: Validity) {
        let mut ctx = array_session().create_execution_ctx();
        let lhs = dtp_array_from_timestamp(86400i64, lhs_validity); // January 2, 1970, 02:00:00 UTC
        let rhs = dtp_array_from_timestamp(0i64, rhs_validity); // January 1, 1970, 01:00:00 UTC

        let comparison = lhs
            .into_array()
            .binary(rhs.into_array(), Operator::Gt)
            .unwrap();
        assert_eq!(true_count(&comparison, &mut ctx), 1);
    }

    #[rstest]
    #[case(Validity::NonNullable, Validity::NonNullable)]
    #[case(Validity::NonNullable, Validity::AllValid)]
    #[case(Validity::AllValid, Validity::NonNullable)]
    #[case(Validity::AllValid, Validity::AllValid)]
    fn compare_date_time_parts_narrowing(
        #[case] lhs_validity: Validity,
        #[case] rhs_validity: Validity,
    ) {
        let mut ctx = array_session().create_execution_ctx();
        let temporal_array = TemporalArray::new_timestamp(
            PrimitiveArray::new(buffer![0i64], lhs_validity.clone()).into_array(),
            TimeUnit::Seconds,
            Some("UTC".into()),
        );

        let lhs = DateTimeParts::try_new(
            DType::Extension(temporal_array.ext_dtype()),
            PrimitiveArray::new(buffer![0i32], lhs_validity).into_array(),
            PrimitiveArray::new(buffer![0u32], Validity::NonNullable).into_array(),
            PrimitiveArray::new(buffer![0i64], Validity::NonNullable).into_array(),
        )
        .unwrap();

        // Timestamp with a value larger than i32::MAX.
        let rhs = dtp_array_from_timestamp(i64::MAX, rhs_validity);

        let comparison = lhs
            .clone()
            .into_array()
            .binary(rhs.clone().into_array(), Operator::Eq)
            .unwrap();
        assert_eq!(true_count(&comparison, &mut ctx), 0);

        let comparison = lhs
            .clone()
            .into_array()
            .binary(rhs.clone().into_array(), Operator::NotEq)
            .unwrap();
        assert_eq!(true_count(&comparison, &mut ctx), 1);

        let comparison = lhs
            .clone()
            .into_array()
            .binary(rhs.clone().into_array(), Operator::Lt)
            .unwrap();
        assert_eq!(true_count(&comparison, &mut ctx), 1);

        let comparison = lhs
            .into_array()
            .binary(rhs.into_array(), Operator::Lte)
            .unwrap();
        assert_eq!(true_count(&comparison, &mut ctx), 1);

        // `CompareOperator::Gt` and `CompareOperator::Gte` only cover the case of all lhs values
        // being larger. Therefore, these cases are not covered by unit tests.
    }

    #[test]
    fn compare_date_time_parts_eq_out_of_range_seconds_constant() -> VortexResult<()> {
        const QUERY_TIMESTAMP_MS: i64 = 1_786_576_785_621;

        let session = array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();
        let timestamp = Scalar::extension::<Timestamp>(
            TimestampOptions {
                unit: TimeUnit::Milliseconds,
                tz: None,
            },
            QUERY_TIMESTAMP_MS.into(),
        );
        let rhs = ConstantArray::new(timestamp, 1).into_array();
        let lhs = DateTimeParts::try_new(
            rhs.dtype().clone(),
            buffer![20_677i32].into_array(),
            buffer![0u16].into_array(),
            buffer![0u16].into_array(),
        )?;

        let comparison = lhs.into_array().binary(rhs, Operator::Eq)?;

        assert_eq!(true_count(&comparison, &mut ctx), 0);
        Ok(())
    }

    /// A days constant below the range of the narrowed storage type must compare as smaller than
    /// every stored value, not larger.
    #[test]
    fn compare_date_time_parts_below_range_days_constant() -> VortexResult<()> {
        // 1969-12-31, which splits to a day of `-1`: below the minimum of an unsigned days array.
        const BEFORE_EPOCH_MS: i64 = -86_400_000;

        let session = array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();
        let timestamp = Scalar::extension::<Timestamp>(
            TimestampOptions {
                unit: TimeUnit::Milliseconds,
                tz: None,
            },
            BEFORE_EPOCH_MS.into(),
        );
        let rhs = ConstantArray::new(timestamp, 1).into_array();
        // 2026-08-08. Days narrow to `u16` because the column holds no pre-epoch value.
        let lhs = DateTimeParts::try_new(
            rhs.dtype().clone(),
            buffer![20_677u16].into_array(),
            buffer![0u16].into_array(),
            buffer![0u16].into_array(),
        )?
        .into_array();

        let lt = lhs.binary(rhs.clone(), Operator::Lt)?;
        assert_eq!(true_count(&lt, &mut ctx), 0);

        let gt = lhs.binary(rhs, Operator::Gt)?;
        assert_eq!(true_count(&gt, &mut ctx), 1);
        Ok(())
    }

    /// A null lhs value stays null, even when the constant is outside the range of the narrowed
    /// days storage type.
    #[rstest]
    #[case(Validity::AllInvalid, [None, None])]
    #[case(Validity::from_iter([false, true]), [None, Some(true)])]
    fn compare_date_time_parts_null_out_of_range_days_constant(
        #[case] days_validity: Validity,
        #[case] expected: [Option<bool>; 2],
    ) -> VortexResult<()> {
        let session = array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();
        // 1969-12-31, a day of `-1`: below the minimum of an unsigned days array.
        let timestamp = Scalar::extension::<Timestamp>(
            TimestampOptions {
                unit: TimeUnit::Milliseconds,
                tz: None,
            },
            (-86_400_000i64).into(),
        );
        let rhs = ConstantArray::new(timestamp, 2).into_array();
        let lhs = DateTimeParts::try_new(
            rhs.dtype().with_nullability(Nullability::Nullable),
            PrimitiveArray::new(buffer![20_677u16, 20_678u16], days_validity).into_array(),
            buffer![0u16, 0u16].into_array(),
            buffer![0u16, 0u16].into_array(),
        )?
        .into_array();

        let gt = lhs.binary(rhs, Operator::Gt)?;
        assert_arrays_eq!(gt, BoolArray::from_iter(expected), &mut ctx);
        Ok(())
    }

    const ALL_OPERATORS: [CompareOperator; 6] = [
        CompareOperator::Eq,
        CompareOperator::NotEq,
        CompareOperator::Lt,
        CompareOperator::Lte,
        CompareOperator::Gt,
        CompareOperator::Gte,
    ];

    fn timestamp_constant(array: &DateTimePartsArray, timestamp: i64, len: usize) -> ArrayRef {
        let options = array
            .dtype()
            .as_extension()
            .metadata_opt::<Timestamp>()
            .expect("timestamp metadata")
            .clone();
        ConstantArray::new(
            Scalar::extension::<Timestamp>(
                options,
                Scalar::primitive(timestamp, Nullability::NonNullable),
            ),
            len,
        )
        .into_array()
    }

    /// Asserts the kernel engages for every operator and matches the comparison over the
    /// timestamps it was built from.
    fn assert_matches_temporal(
        timestamps: PrimitiveArray,
        unit: TimeUnit,
        constants: impl IntoIterator<Item = i64>,
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let temporal =
            TemporalArray::new_timestamp(timestamps.into_array(), unit, Some("UTC".into()));
        let array = DateTimeParts::try_from_temporal(temporal.clone(), &mut ctx)?;
        for constant in constants {
            let rhs = timestamp_constant(&array, constant, array.len());
            for operator in ALL_OPERATORS {
                let actual = <DateTimeParts as CompareKernel>::compare(
                    array.as_view(),
                    &rhs,
                    operator,
                    &mut ctx,
                )?
                .unwrap_or_else(|| panic!("datetime-parts compare must engage for {operator:?}"));
                let expected = temporal
                    .clone()
                    .into_array()
                    .binary(rhs.clone(), Operator::from(operator))?;
                assert_arrays_eq!(actual, expected, &mut ctx);
            }
        }
        Ok(())
    }

    #[rstest]
    #[case::seconds(TimeUnit::Seconds)]
    #[case::milliseconds(TimeUnit::Milliseconds)]
    #[case::nanoseconds(TimeUnit::Nanoseconds)]
    fn ordering_matches_temporal(#[case] unit: TimeUnit) -> VortexResult<()> {
        let divisor: i64 = match unit {
            TimeUnit::Seconds => 1,
            TimeUnit::Milliseconds => 1_000,
            TimeUnit::Microseconds => 1_000_000,
            TimeUnit::Nanoseconds => 1_000_000_000,
            TimeUnit::Days => unreachable!(),
        };
        let day = 86_400 * divisor;
        // Both signs, rows sharing a day or a second, and rows differing only in subseconds.
        let values: Vec<i64> = vec![
            -2 * day - 1,
            -day - divisor,
            -day,
            -day + 1,
            -divisor - 1,
            -1,
            0,
            1,
            divisor - 1,
            divisor,
            day - 1,
            day,
            day + divisor,
            day + divisor + 1,
            3 * day + 17 * divisor + 5,
        ];
        let mut constants = Vec::new();
        for value in &values {
            constants.extend([value - 1, *value, value + 1]);
        }
        constants.extend([-20_000 * day, 20_000 * day]);

        assert_matches_temporal(
            PrimitiveArray::from_iter(values.clone()),
            unit,
            constants.clone(),
        )?;

        let nullable = PrimitiveArray::from_option_iter(
            values
                .iter()
                .enumerate()
                .map(|(i, value)| (i % 3 != 1).then_some(*value)),
        );
        assert_matches_temporal(nullable, unit, constants)
    }

    #[test]
    fn null_rows_stay_null_when_parts_differ() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        // Row 1 is null and its stored seconds differ from the constant's.
        let values = PrimitiveArray::new(
            buffer![86_400i64 + 5, 86_400i64 + 7, 86_400i64 + 5],
            Validity::from_iter([true, false, true]),
        );
        let array = DateTimeParts::try_from_temporal(
            TemporalArray::new_timestamp(
                values.into_array(),
                TimeUnit::Seconds,
                Some("UTC".into()),
            ),
            &mut ctx,
        )?;
        let rhs = timestamp_constant(&array, 86_400 + 5, array.len());

        for (operator, valid) in [
            (CompareOperator::Eq, true),
            (CompareOperator::NotEq, false),
            (CompareOperator::Lt, false),
            (CompareOperator::Lte, true),
            (CompareOperator::Gt, false),
            (CompareOperator::Gte, true),
        ] {
            let actual = <DateTimeParts as CompareKernel>::compare(
                array.as_view(),
                &rhs,
                operator,
                &mut ctx,
            )?
            .expect("kernel engages");
            let expected = BoolArray::from_iter([Some(valid), None, Some(valid)]);
            assert_arrays_eq!(actual, expected, &mut ctx);
        }
        Ok(())
    }
}
