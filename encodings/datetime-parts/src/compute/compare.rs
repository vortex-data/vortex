use vortex_array::array::ConstantArray;
use vortex_array::compute::{and, compare, or, try_cast, CompareFn, Operator};
use vortex_array::{Array, IntoArray};
use vortex_datetime_dtype::TemporalMetadata;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::array::{DateTimePartsArray, DateTimePartsEncoding};
use crate::timestamp;

fn compare_dtp(lhs: &Array, rhs: i64, operator: Operator) -> VortexResult<Array> {
    match try_cast(ConstantArray::new(rhs, lhs.len()), lhs.dtype()) {
        Ok(casted) => compare(lhs, casted, operator),
        // The narrowing cast failed. Therefore, attempt to derive the result from the operator.
        _ => {
            let constant_value = match operator {
                Operator::Eq | Operator::Lte => false,
                Operator::NotEq | Operator::Gte => true,
                _ => unreachable!("operator {} not supported", operator),
            };
            Ok(ConstantArray::new(constant_value, lhs.len()).into_array())
        }
    }
}

fn compare_eq(
    lhs: &DateTimePartsArray,
    ts_parts: &timestamp::TimestampParts,
) -> VortexResult<Option<Array>> {
    let mut comparison = compare_dtp(&lhs.days(), ts_parts.days, Operator::Eq)?;
    if comparison.statistics().compute_max::<bool>() == Some(false) {
        // All values are different.
        return Ok(Some(comparison));
    }

    comparison = and(
        compare_dtp(&lhs.seconds(), ts_parts.seconds, Operator::Eq)?,
        comparison,
    )?;

    if comparison.statistics().compute_max::<bool>() == Some(false) {
        // All values are different.
        return Ok(Some(comparison));
    }

    comparison = and(
        compare_dtp(&lhs.subseconds(), ts_parts.subseconds, Operator::Eq)?,
        comparison,
    )?;

    Ok(Some(comparison))
}

fn compare_ne(
    lhs: &DateTimePartsArray,
    ts_parts: &timestamp::TimestampParts,
) -> VortexResult<Option<Array>> {
    let mut comparison = compare_dtp(&lhs.days(), ts_parts.days, Operator::NotEq)?;
    if comparison.statistics().compute_min::<bool>() == Some(true) {
        // All values are different.
        return Ok(Some(comparison));
    }

    comparison = or(
        compare_dtp(&lhs.seconds(), ts_parts.seconds, Operator::NotEq)?,
        comparison,
    )?;

    if comparison.statistics().compute_min::<bool>() == Some(true) {
        // All values are different.
        return Ok(Some(comparison));
    }

    comparison = or(
        compare_dtp(&lhs.subseconds(), ts_parts.subseconds, Operator::NotEq)?,
        comparison,
    )?;

    Ok(Some(comparison))
}

fn compare_lte(
    lhs: &DateTimePartsArray,
    ts_parts: &timestamp::TimestampParts,
) -> VortexResult<Option<Array>> {
    let days_lt = compare_dtp(&lhs.days(), ts_parts.days, Operator::Lt)?;
    if days_lt.statistics().compute_min::<bool>() == Some(true) {
        // All values on the lhs are smaller.
        return Ok(Some(days_lt));
    }

    let days_eq = compare_dtp(&lhs.days(), ts_parts.days, Operator::Eq)?;
    if days_lt.statistics().compute_max::<bool>() == Some(false)
        && days_eq.statistics().compute_max::<bool>() == Some(false)
    {
        // All values on the lhs are larger.
        return Ok(Some(days_lt));
    }

    let sec_lt = compare_dtp(&lhs.seconds(), ts_parts.seconds, Operator::Lt)?;
    let sec_eq = compare_dtp(&lhs.seconds(), ts_parts.seconds, Operator::Eq)?;
    let sub_lte = compare_dtp(&lhs.subseconds(), ts_parts.subseconds, Operator::Lte)?;

    // (days_lhs, seconds_lhs, sub_lhs) <= (days_rhs, seconds_rhs, sub_rhs)
    let result = or(days_lt, and(days_eq, or(sec_lt, and(sec_eq, sub_lte)?)?)?)?;

    Ok(Some(result))
}

fn compare_gte(
    lhs: &DateTimePartsArray,
    ts_parts: &timestamp::TimestampParts,
) -> VortexResult<Option<Array>> {
    let days_gt = compare_dtp(&lhs.days(), ts_parts.days, Operator::Gt)?;
    if days_gt.statistics().compute_min::<bool>() == Some(true) {
        // All values on the lhs are larger.
        return Ok(Some(days_gt));
    }

    let days_eq = compare_dtp(&lhs.days(), ts_parts.days, Operator::Eq)?;
    if days_gt.statistics().compute_max::<bool>() == Some(false)
        && days_eq.statistics().compute_max::<bool>() == Some(false)
    {
        // All values on the lhs are smaller.
        return Ok(Some(days_gt));
    }

    let sec_gt = compare_dtp(&lhs.seconds(), ts_parts.seconds, Operator::Gt)?;
    let sec_eq = compare_dtp(&lhs.seconds(), ts_parts.seconds, Operator::Eq)?;
    let sub_gte = compare_dtp(&lhs.subseconds(), ts_parts.subseconds, Operator::Gte)?;

    // (days_lhs, seconds_lhs, sub_lhs) >= (days_rhs, seconds_rhs, sub_rhs)
    let result = or(days_gt, and(days_eq, or(sec_gt, and(sec_eq, sub_gte)?)?)?)?;

    Ok(Some(result))
}

impl CompareFn<DateTimePartsArray> for DateTimePartsEncoding {
    /// Compares two arrays and returns a new boolean array with the result of the comparison.
    /// Or, returns None if comparison is not supported.
    ///
    /// # NOTE: `Operator::Lt` and `Operator::Gt` are currently not supported.
    fn compare(
        &self,
        lhs: &DateTimePartsArray,
        rhs: &Array,
        operator: Operator,
    ) -> VortexResult<Option<Array>> {
        if !matches!(
            operator,
            Operator::Eq | Operator::NotEq | Operator::Lte | Operator::Gte
        ) {
            return Ok(None);
        }

        let Some(rhs_const) = rhs.as_constant() else {
            return Ok(None);
        };
        let Ok(Some(timestamp)) = rhs_const
            .as_extension()
            .storage()
            .as_primitive()
            .as_::<i64>()
        else {
            return Ok(None);
        };

        let DType::Extension(ext_dtype) = rhs_const.dtype() else {
            return Ok(None);
        };

        let temporal_metadata = TemporalMetadata::try_from(ext_dtype.as_ref())?;
        let ts_parts = timestamp::split(timestamp, temporal_metadata.time_unit())?;

        match operator {
            Operator::Eq => compare_eq(lhs, &ts_parts),
            Operator::NotEq => compare_ne(lhs, &ts_parts),
            Operator::Lte => compare_lte(lhs, &ts_parts),
            Operator::Gte => compare_gte(lhs, &ts_parts),
            _ => unreachable!("operator {} not supported", operator),
        }
    }
}

#[cfg(test)]
mod test {
    use vortex_array::array::{PrimitiveArray, TemporalArray};
    use vortex_array::compute::Operator;
    use vortex_array::validity::Validity;
    use vortex_array::IntoArray;
    use vortex_buffer::buffer;
    use vortex_datetime_dtype::TimeUnit;
    use vortex_dtype::NativePType;

    use super::*;

    fn dtp_array_from_timestamp<T: NativePType>(value: T) -> DateTimePartsArray {
        DateTimePartsArray::try_from(TemporalArray::new_timestamp(
            PrimitiveArray::new(buffer![value], Validity::NonNullable).into_array(),
            TimeUnit::S,
            Some("UTC".to_string()),
        ))
        .expect("Failed to construct DateTimePartsArray from TemporalArray")
    }

    #[test]
    fn compare_date_time_parts_eq() {
        let lhs = dtp_array_from_timestamp(86400i64); // January 2, 1970, 00:00:00 UTC
        let rhs = dtp_array_from_timestamp(86400i64); // January 2, 1970, 00:00:00 UTC
        let comparison = DateTimePartsEncoding
            .compare(&lhs, &rhs, Operator::Eq)
            .unwrap()
            .unwrap();
        assert_eq!(comparison.statistics().compute_true_count().unwrap(), 1);

        let rhs = dtp_array_from_timestamp(0i64); // January 1, 1970, 00:00:00 UTC
        let comparison = DateTimePartsEncoding
            .compare(&lhs, &rhs, Operator::Eq)
            .unwrap()
            .unwrap();
        assert_eq!(comparison.statistics().compute_true_count().unwrap(), 0);
    }

    #[test]
    fn compare_date_time_parts_ne() {
        let lhs = dtp_array_from_timestamp(86400i64); // January 2, 1970, 00:00:00 UTC
        let rhs = dtp_array_from_timestamp(86401i64); // January 2, 1970, 00:00:01 UTC
        let comparison = DateTimePartsEncoding
            .compare(&lhs, &rhs, Operator::NotEq)
            .unwrap()
            .unwrap();
        assert_eq!(comparison.statistics().compute_true_count().unwrap(), 1);

        let rhs = dtp_array_from_timestamp(86400i64); // January 2, 1970, 00:00:00 UTC
        let comparison = DateTimePartsEncoding
            .compare(&lhs, &rhs, Operator::NotEq)
            .unwrap()
            .unwrap();
        assert_eq!(comparison.statistics().compute_true_count().unwrap(), 0);
    }

    #[test]
    fn compare_date_time_parts_lte() {
        let lhs = dtp_array_from_timestamp(86400i64); // January 2, 1970, 00:00:00 UTC
        let rhs = dtp_array_from_timestamp(86400i64); // January 2, 1970, 00:00:00 UTC

        let comparison = DateTimePartsEncoding
            .compare(&lhs, &rhs, Operator::Lte)
            .unwrap()
            .unwrap();
        assert_eq!(comparison.statistics().compute_true_count().unwrap(), 1);

        let lhs = dtp_array_from_timestamp(86400i64 + 3600i64); // January 2, 1970, 01:00:01 UTC
        let rhs = dtp_array_from_timestamp(86400i64 + 7200i64); // January 2, 1970, 02:00:00 UTC

        let comparison = DateTimePartsEncoding
            .compare(&lhs, &rhs, Operator::Lte)
            .unwrap()
            .unwrap();
        assert_eq!(comparison.statistics().compute_true_count().unwrap(), 1);

        // Swap the operands to test the reverse.
        let comparison = DateTimePartsEncoding
            .compare(&rhs, &lhs, Operator::Lte)
            .unwrap()
            .unwrap();
        assert_eq!(comparison.statistics().compute_true_count().unwrap(), 0);
    }

    #[test]
    fn compare_date_time_parts_gte() {
        let lhs = dtp_array_from_timestamp(86400i64); // January 2, 1970, 00:00:00 UTC
        let rhs = dtp_array_from_timestamp(86400i64); // January 2, 1970, 00:00:00 UTC

        let comparison = DateTimePartsEncoding
            .compare(&lhs, &rhs, Operator::Gte)
            .unwrap()
            .unwrap();
        assert_eq!(comparison.statistics().compute_true_count().unwrap(), 1);

        let lhs = dtp_array_from_timestamp(86400i64 + 7200i64); // January 2, 1970, 02:00:00 UTC
        let rhs = dtp_array_from_timestamp(86400i64 + 3600i64); // January 2, 1970, 01:00:00 UTC

        let comparison = DateTimePartsEncoding
            .compare(&lhs, &rhs, Operator::Gte)
            .unwrap()
            .unwrap();
        assert_eq!(comparison.statistics().compute_true_count().unwrap(), 1);

        // Swap the operands to test the reverse.
        let comparison = DateTimePartsEncoding
            .compare(&rhs, &lhs, Operator::Gte)
            .unwrap()
            .unwrap();
        assert_eq!(comparison.statistics().compute_true_count().unwrap(), 0);
    }
}
