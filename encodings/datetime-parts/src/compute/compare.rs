// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateFn;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::aggregate_fn::fns::min::Min;
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

        match operator {
            CompareOperator::Eq => compare_eq(lhs, &ts_parts, nullability, ctx),
            CompareOperator::NotEq => compare_ne(lhs, &ts_parts, nullability, ctx),
            // lt and lte have identical behavior, as we optimize
            // for the case that all days on the lhs are smaller.
            // If that special case is not hit, we return `Ok(None)` to
            // signal that the comparison wasn't handled within dtp.
            CompareOperator::Lt => compare_lt(lhs, &ts_parts, nullability, ctx),
            CompareOperator::Lte => compare_lt(lhs, &ts_parts, nullability, ctx),
            // (Like for lt, lte)
            CompareOperator::Gt => compare_gt(lhs, &ts_parts, nullability, ctx),
            CompareOperator::Gte => compare_gt(lhs, &ts_parts, nullability, ctx),
        }
    }
}

fn compare_eq(
    lhs: ArrayView<DateTimeParts>,
    ts_parts: &timestamp::TimestampParts,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let mut comparison = compare_dtp(lhs.days(), ts_parts.days, CompareOperator::Eq, nullability)?;
    if comparison
        .aggregations()
        .compute_as::<bool>(
            &AggregateFn::new(Max, NumericalAggregateOpts::default()).erased(),
            ctx,
        )
        .ok()
        == Some(false)
    {
        // All values are different.
        return Ok(Some(comparison));
    }

    comparison = compare_dtp(
        lhs.seconds(),
        ts_parts.seconds,
        CompareOperator::Eq,
        nullability,
    )?
    .binary(comparison, Operator::And)?;

    if comparison
        .aggregations()
        .compute_as::<bool>(
            &AggregateFn::new(Max, NumericalAggregateOpts::default()).erased(),
            ctx,
        )
        .ok()
        == Some(false)
    {
        // All values are different.
        return Ok(Some(comparison));
    }

    comparison = compare_dtp(
        lhs.subseconds(),
        ts_parts.subseconds,
        CompareOperator::Eq,
        nullability,
    )?
    .binary(comparison, Operator::And)?;

    Ok(Some(comparison))
}

fn compare_ne(
    lhs: ArrayView<DateTimeParts>,
    ts_parts: &timestamp::TimestampParts,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let mut comparison = compare_dtp(
        lhs.days(),
        ts_parts.days,
        CompareOperator::NotEq,
        nullability,
    )?;
    if comparison
        .aggregations()
        .compute_as::<bool>(
            &AggregateFn::new(Min, NumericalAggregateOpts::default()).erased(),
            ctx,
        )
        .ok()
        == Some(true)
    {
        // All values are different.
        return Ok(Some(comparison));
    }

    comparison = compare_dtp(
        lhs.seconds(),
        ts_parts.seconds,
        CompareOperator::NotEq,
        nullability,
    )?
    .binary(comparison, Operator::Or)?;

    if comparison
        .aggregations()
        .compute_as::<bool>(
            &AggregateFn::new(Min, NumericalAggregateOpts::default()).erased(),
            ctx,
        )
        .ok()
        == Some(true)
    {
        // All values are different.
        return Ok(Some(comparison));
    }

    comparison = compare_dtp(
        lhs.subseconds(),
        ts_parts.subseconds,
        CompareOperator::NotEq,
        nullability,
    )?
    .binary(comparison, Operator::Or)?;

    Ok(Some(comparison))
}

fn compare_lt(
    lhs: ArrayView<DateTimeParts>,
    ts_parts: &timestamp::TimestampParts,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let days_lt = compare_dtp(lhs.days(), ts_parts.days, CompareOperator::Lt, nullability)?;
    if days_lt
        .aggregations()
        .compute_as::<bool>(
            &AggregateFn::new(Min, NumericalAggregateOpts::default()).erased(),
            ctx,
        )
        .ok()
        == Some(true)
    {
        // All values on the lhs are smaller.
        return Ok(Some(days_lt));
    }

    Ok(None)
}

fn compare_gt(
    lhs: ArrayView<DateTimeParts>,
    ts_parts: &timestamp::TimestampParts,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let days_gt = compare_dtp(lhs.days(), ts_parts.days, CompareOperator::Gt, nullability)?;
    if days_gt
        .aggregations()
        .compute_as::<bool>(
            &AggregateFn::new(Min, NumericalAggregateOpts::default()).erased(),
            ctx,
        )
        .ok()
        == Some(true)
    {
        // All values on the lhs are larger.
        return Ok(Some(days_gt));
    }

    Ok(None)
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
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::ExecutionCtx;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::AggregateFnRef;
    use vortex_array::aggregate_fn::AggregateFnVTable;
    use vortex_array::aggregate_fn::fns::sum::sum;
    use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
    use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
    use vortex_array::array_session;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::TemporalArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::IntegerPType;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::extension::datetime::Timestamp;
    use vortex_array::extension::datetime::TimestampOptions;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_mask::Mask;

    use super::*;
    use crate::DateTimeParts;
    use crate::DateTimePartsArray;

    #[derive(Debug)]
    struct StatefulMaxKernel(AtomicUsize);

    impl DynAggregateKernel for StatefulMaxKernel {
        fn aggregate(
            &self,
            aggregate_fn: &AggregateFnRef,
            batch: &ArrayRef,
            _ctx: &mut ExecutionCtx,
        ) -> VortexResult<Option<Scalar>> {
            if !aggregate_fn.is::<Max>() || !batch.dtype().is_boolean() {
                return Ok(None);
            }

            Ok(Some(Scalar::bool(
                self.0.fetch_add(1, Ordering::Relaxed) != 0,
                Nullability::Nullable,
            )))
        }
    }

    #[test]
    fn decimal_cast_resolves_stateful_comparison_validity_once() -> VortexResult<()> {
        static KERNEL: StatefulMaxKernel = StatefulMaxKernel(AtomicUsize::new(0));
        KERNEL.0.store(0, Ordering::Relaxed);

        let session = array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();
        let timestamp = Scalar::extension::<Timestamp>(
            TimestampOptions {
                unit: TimeUnit::Seconds,
                tz: None,
            },
            0i64.into(),
        );
        let rhs = ConstantArray::new(timestamp, 2).into_array();
        let lhs = DateTimeParts::try_new(
            rhs.dtype().clone(),
            buffer![0i32, 0].into_array(),
            buffer![1i32, 0].into_array(),
            buffer![0i32, 0].into_array(),
        )?
        .into_array();
        let comparison = lhs.binary(rhs, Operator::Eq)?;
        let expected = Mask::from_iter([false, true]);
        assert_eq!(comparison.clone().execute::<Mask>(&mut ctx)?, expected);

        session.aggregate_fns().register_aggregate_kernel(
            comparison.encoding_id(),
            Some(Max.id()),
            &KERNEL,
        );

        // The first false maximum returns only the all-true days comparison. Later executions
        // compare seconds too, so this safely registered kernel makes the lazy mask change.
        assert_eq!(
            comparison.clone().execute::<Mask>(&mut ctx)?,
            Mask::AllTrue(2)
        );
        assert_eq!(comparison.clone().execute::<Mask>(&mut ctx)?, expected);

        KERNEL.0.store(0, Ordering::Relaxed);
        let source =
            PrimitiveArray::new(buffer![1000i32, 1], Validity::Array(comparison)).into_array();
        let casted = source
            .cast(DType::Decimal(
                DecimalDType::new(2, 0),
                Nullability::NonNullable,
            ))?
            .execute::<DecimalArray>(&mut ctx);
        let error = casted.expect_err("the resolved validity makes 1000 a valid value");
        assert!(error.to_string().contains("does not fit in precision"));
        assert_eq!(KERNEL.0.load(Ordering::Relaxed), 1);
        Ok(())
    }

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
}
