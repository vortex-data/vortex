// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::CompareKernel;
use vortex_array::expr::CompareOperator;
use vortex_array::expr::Operator;
use vortex_array::extension::datetime::Timestamp;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::array::DateTimePartsArray;
use crate::array::DateTimePartsVTable;
use crate::timestamp;

impl CompareKernel for DateTimePartsVTable {
    fn compare(
        lhs: &DateTimePartsArray,
        rhs: &dyn Array,
        operator: CompareOperator,
        _ctx: &mut ExecutionCtx,
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
            CompareOperator::Eq => compare_eq(lhs, &ts_parts, nullability),
            CompareOperator::NotEq => compare_ne(lhs, &ts_parts, nullability),
            // lt and lte have identical behavior, as we optimize
            // for the case that all days on the lhs are smaller.
            // If that special case is not hit, we return `Ok(None)` to
            // signal that the comparison wasn't handled within dtp.
            CompareOperator::Lt => compare_lt(lhs, &ts_parts, nullability),
            CompareOperator::Lte => compare_lt(lhs, &ts_parts, nullability),
            // (Like for lt, lte)
            CompareOperator::Gt => compare_gt(lhs, &ts_parts, nullability),
            CompareOperator::Gte => compare_gt(lhs, &ts_parts, nullability),
        }
    }
}

fn compare_eq(
    lhs: &DateTimePartsArray,
    ts_parts: &timestamp::TimestampParts,
    nullability: Nullability,
) -> VortexResult<Option<ArrayRef>> {
    let mut comparison = compare_dtp(lhs.days(), ts_parts.days, CompareOperator::Eq, nullability)?;
    if comparison.statistics().compute_max::<bool>() == Some(false) {
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

    if comparison.statistics().compute_max::<bool>() == Some(false) {
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
    lhs: &DateTimePartsArray,
    ts_parts: &timestamp::TimestampParts,
    nullability: Nullability,
) -> VortexResult<Option<ArrayRef>> {
    let mut comparison = compare_dtp(
        lhs.days(),
        ts_parts.days,
        CompareOperator::NotEq,
        nullability,
    )?;
    if comparison.statistics().compute_min::<bool>() == Some(true) {
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

    if comparison.statistics().compute_min::<bool>() == Some(true) {
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
    lhs: &DateTimePartsArray,
    ts_parts: &timestamp::TimestampParts,
    nullability: Nullability,
) -> VortexResult<Option<ArrayRef>> {
    let days_lt = compare_dtp(lhs.days(), ts_parts.days, CompareOperator::Lt, nullability)?;
    if days_lt.statistics().compute_min::<bool>() == Some(true) {
        // All values on the lhs are smaller.
        return Ok(Some(days_lt));
    }

    Ok(None)
}

fn compare_gt(
    lhs: &DateTimePartsArray,
    ts_parts: &timestamp::TimestampParts,
    nullability: Nullability,
) -> VortexResult<Option<ArrayRef>> {
    let days_gt = compare_dtp(lhs.days(), ts_parts.days, CompareOperator::Gt, nullability)?;
    if days_gt.statistics().compute_min::<bool>() == Some(true) {
        // All values on the lhs are larger.
        return Ok(Some(days_gt));
    }

    Ok(None)
}

fn compare_dtp(
    lhs: &dyn Array,
    rhs: i64,
    operator: CompareOperator,
    nullability: Nullability,
) -> VortexResult<ArrayRef> {
    // Since nullability is stripped from RHS and carried forward through nullability argument we want to incorporate it into lhs.dtype() that we cast rhs into
    match ConstantArray::new(rhs, lhs.len())
        .into_array()
        .cast(lhs.dtype().with_nullability(nullability))
    {
        Ok(casted) => lhs.to_array().binary(casted, Operator::from(operator)),
        // The narrowing cast failed. Therefore, we know lhs < rhs.
        _ => {
            let constant_value = match operator {
                CompareOperator::Eq | CompareOperator::Gte | CompareOperator::Gt => false,
                CompareOperator::NotEq | CompareOperator::Lte | CompareOperator::Lt => true,
            };
            Ok(
                ConstantArray::new(Scalar::bool(constant_value, nullability), lhs.len())
                    .into_array(),
            )
        }
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::TemporalArray;
    use vortex_array::dtype::IntegerPType;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    use super::*;

    fn dtp_array_from_timestamp<T: IntegerPType>(
        value: T,
        validity: Validity,
    ) -> DateTimePartsArray {
        DateTimePartsArray::try_from(TemporalArray::new_timestamp(
            PrimitiveArray::new(buffer![value], validity).into_array(),
            TimeUnit::Seconds,
            Some("UTC".into()),
        ))
        .expect("Failed to construct DateTimePartsArray from TemporalArray")
    }

    #[rstest]
    #[case(Validity::NonNullable, Validity::NonNullable)]
    #[case(Validity::NonNullable, Validity::AllValid)]
    #[case(Validity::AllValid, Validity::NonNullable)]
    #[case(Validity::AllValid, Validity::AllValid)]
    fn compare_date_time_parts_eq(#[case] lhs_validity: Validity, #[case] rhs_validity: Validity) {
        let lhs = dtp_array_from_timestamp(86400i64, lhs_validity); // January 2, 1970, 00:00:00 UTC
        let rhs = dtp_array_from_timestamp(86400i64, rhs_validity.clone()); // January 2, 1970, 00:00:00 UTC
        let comparison = lhs.to_array().binary(rhs.to_array(), Operator::Eq).unwrap();
        assert_eq!(comparison.as_bool_typed().true_count().unwrap(), 1);

        let rhs = dtp_array_from_timestamp(0i64, rhs_validity); // January 1, 1970, 00:00:00 UTC
        let comparison = lhs.to_array().binary(rhs.to_array(), Operator::Eq).unwrap();
        assert_eq!(comparison.as_bool_typed().true_count().unwrap(), 0);
    }

    #[rstest]
    #[case(Validity::NonNullable, Validity::NonNullable)]
    #[case(Validity::NonNullable, Validity::AllValid)]
    #[case(Validity::AllValid, Validity::NonNullable)]
    #[case(Validity::AllValid, Validity::AllValid)]
    fn compare_date_time_parts_ne(#[case] lhs_validity: Validity, #[case] rhs_validity: Validity) {
        let lhs = dtp_array_from_timestamp(86400i64, lhs_validity); // January 2, 1970, 00:00:00 UTC
        let rhs = dtp_array_from_timestamp(86401i64, rhs_validity.clone()); // January 2, 1970, 00:00:01 UTC
        let comparison = lhs
            .to_array()
            .binary(rhs.to_array(), Operator::NotEq)
            .unwrap();
        assert_eq!(comparison.as_bool_typed().true_count().unwrap(), 1);

        let rhs = dtp_array_from_timestamp(86400i64, rhs_validity); // January 2, 1970, 00:00:00 UTC
        let comparison = lhs
            .to_array()
            .binary(rhs.to_array(), Operator::NotEq)
            .unwrap();
        assert_eq!(comparison.as_bool_typed().true_count().unwrap(), 0);
    }

    #[rstest]
    #[case(Validity::NonNullable, Validity::NonNullable)]
    #[case(Validity::NonNullable, Validity::AllValid)]
    #[case(Validity::AllValid, Validity::NonNullable)]
    #[case(Validity::AllValid, Validity::AllValid)]
    fn compare_date_time_parts_lt(#[case] lhs_validity: Validity, #[case] rhs_validity: Validity) {
        let lhs = dtp_array_from_timestamp(0i64, lhs_validity); // January 1, 1970, 01:00:00 UTC
        let rhs = dtp_array_from_timestamp(86400i64, rhs_validity); // January 2, 1970, 00:00:00 UTC

        let comparison = lhs.to_array().binary(rhs.to_array(), Operator::Lt).unwrap();
        assert_eq!(comparison.as_bool_typed().true_count().unwrap(), 1);
    }

    #[rstest]
    #[case(Validity::NonNullable, Validity::NonNullable)]
    #[case(Validity::NonNullable, Validity::AllValid)]
    #[case(Validity::AllValid, Validity::NonNullable)]
    #[case(Validity::AllValid, Validity::AllValid)]
    fn compare_date_time_parts_gt(#[case] lhs_validity: Validity, #[case] rhs_validity: Validity) {
        let lhs = dtp_array_from_timestamp(86400i64, lhs_validity); // January 2, 1970, 02:00:00 UTC
        let rhs = dtp_array_from_timestamp(0i64, rhs_validity); // January 1, 1970, 01:00:00 UTC

        let comparison = lhs.to_array().binary(rhs.to_array(), Operator::Gt).unwrap();
        assert_eq!(comparison.as_bool_typed().true_count().unwrap(), 1);
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
        let temporal_array = TemporalArray::new_timestamp(
            PrimitiveArray::new(buffer![0i64], lhs_validity.clone()).into_array(),
            TimeUnit::Seconds,
            Some("UTC".into()),
        );

        let lhs = DateTimePartsArray::try_new(
            DType::Extension(temporal_array.ext_dtype()),
            PrimitiveArray::new(buffer![0i32], lhs_validity).into_array(),
            PrimitiveArray::new(buffer![0u32], Validity::NonNullable).into_array(),
            PrimitiveArray::new(buffer![0i64], Validity::NonNullable).into_array(),
        )
        .unwrap();

        // Timestamp with a value larger than i32::MAX.
        let rhs = dtp_array_from_timestamp(i64::MAX, rhs_validity);

        let comparison = lhs.to_array().binary(rhs.to_array(), Operator::Eq).unwrap();
        assert_eq!(comparison.as_bool_typed().true_count().unwrap(), 0);

        let comparison = lhs
            .to_array()
            .binary(rhs.to_array(), Operator::NotEq)
            .unwrap();
        assert_eq!(comparison.as_bool_typed().true_count().unwrap(), 1);

        let comparison = lhs.to_array().binary(rhs.to_array(), Operator::Lt).unwrap();
        assert_eq!(comparison.as_bool_typed().true_count().unwrap(), 1);

        let comparison = lhs
            .to_array()
            .binary(rhs.to_array(), Operator::Lte)
            .unwrap();
        assert_eq!(comparison.as_bool_typed().true_count().unwrap(), 1);

        // `CompareOperator::Gt` and `CompareOperator::Gte` only cover the case of all lhs values
        // being larger. Therefore, these cases are not covered by unit tests.
    }
}
