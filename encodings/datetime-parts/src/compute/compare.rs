use vortex_array::arrays::ConstantArray;
use vortex_array::compute::{and, compare, or, try_cast, CompareFn, Operator};
use vortex_array::{Array, ArrayRef};
use vortex_datetime_dtype::TemporalMetadata;
use vortex_dtype::DType;
use vortex_error::{VortexExpect as _, VortexResult};

use crate::array::{DateTimePartsArray, DateTimePartsEncoding};
use crate::timestamp;

impl CompareFn<&DateTimePartsArray> for DateTimePartsEncoding {
    /// Compares two arrays and returns a new boolean array with the result of the comparison.
    /// Or, returns None if comparison is not supported.
    fn compare(
        &self,
        lhs: &DateTimePartsArray,
        rhs: &dyn Array,
        operator: Operator,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(rhs_const) = rhs.as_constant() else {
            return Ok(None);
        };
        let Ok(timestamp) = rhs_const
            .as_extension()
            .storage()
            .as_primitive()
            .as_::<i64>()
            .map(|maybe_value| maybe_value.vortex_expect("null scalar handled in top-level"))
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
            // lt and lte have identical behavior, as we optimize
            // for the case that all days on the lhs are smaller.
            //
            // If that special case is not hit, we return `Ok(None)` to
            // signal that the comparison wasn't handled within dtp.
            Operator::Lt => compare_lt(lhs, &ts_parts),
            Operator::Lte => compare_lt(lhs, &ts_parts),
            // (Like for lt, lte)
            Operator::Gt => compare_gt(lhs, &ts_parts),
            Operator::Gte => compare_gt(lhs, &ts_parts),
        }
    }
}

fn compare_eq(
    lhs: &DateTimePartsArray,
    ts_parts: &timestamp::TimestampParts,
) -> VortexResult<Option<ArrayRef>> {
    let mut comparison = compare_dtp(lhs.days(), ts_parts.days, Operator::Eq)?;
    if comparison.statistics().compute_max::<bool>() == Some(false) {
        // All values are different.
        return Ok(Some(comparison));
    }

    comparison = and(
        &compare_dtp(lhs.seconds(), ts_parts.seconds, Operator::Eq)?,
        &comparison,
    )?;

    if comparison.statistics().compute_max::<bool>() == Some(false) {
        // All values are different.
        return Ok(Some(comparison));
    }

    comparison = and(
        &compare_dtp(lhs.subseconds(), ts_parts.subseconds, Operator::Eq)?,
        &comparison,
    )?;

    Ok(Some(comparison))
}

fn compare_ne(
    lhs: &DateTimePartsArray,
    ts_parts: &timestamp::TimestampParts,
) -> VortexResult<Option<ArrayRef>> {
    let mut comparison = compare_dtp(lhs.days(), ts_parts.days, Operator::NotEq)?;
    if comparison.statistics().compute_min::<bool>() == Some(true) {
        // All values are different.
        return Ok(Some(comparison));
    }

    comparison = or(
        &compare_dtp(lhs.seconds(), ts_parts.seconds, Operator::NotEq)?,
        &comparison,
    )?;

    if comparison.statistics().compute_min::<bool>() == Some(true) {
        // All values are different.
        return Ok(Some(comparison));
    }

    comparison = or(
        &compare_dtp(lhs.subseconds(), ts_parts.subseconds, Operator::NotEq)?,
        &comparison,
    )?;

    Ok(Some(comparison))
}

fn compare_lt(
    lhs: &DateTimePartsArray,
    ts_parts: &timestamp::TimestampParts,
) -> VortexResult<Option<ArrayRef>> {
    let days_lt = compare_dtp(lhs.days(), ts_parts.days, Operator::Lt)?;
    if days_lt.statistics().compute_min::<bool>() == Some(true) {
        // All values on the lhs are smaller.
        return Ok(Some(days_lt));
    }

    Ok(None)
}

fn compare_gt(
    lhs: &DateTimePartsArray,
    ts_parts: &timestamp::TimestampParts,
) -> VortexResult<Option<ArrayRef>> {
    let days_gt = compare_dtp(lhs.days(), ts_parts.days, Operator::Gt)?;
    if days_gt.statistics().compute_min::<bool>() == Some(true) {
        // All values on the lhs are larger.
        return Ok(Some(days_gt));
    }

    Ok(None)
}

fn compare_dtp(lhs: &dyn Array, rhs: i64, operator: Operator) -> VortexResult<ArrayRef> {
    match try_cast(&ConstantArray::new(rhs, lhs.len()), lhs.dtype()) {
        Ok(casted) => compare(lhs, &casted, operator),
        // The narrowing cast failed. Therefore, we know lhs < rhs.
        _ => {
            let constant_value = match operator {
                Operator::Eq | Operator::Gte | Operator::Gt => false,
                Operator::NotEq | Operator::Lte | Operator::Lt => true,
            };
            Ok(ConstantArray::new(constant_value, lhs.len()).into_array())
        }
    }
}

#[cfg(test)]
mod test {
    use vortex_array::arrays::{PrimitiveArray, TemporalArray};
    use vortex_array::compute::Operator;
    use vortex_array::validity::Validity;
    use vortex_array::ArrayVariants;
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
        assert_eq!(comparison.as_bool_typed().unwrap().true_count().unwrap(), 1);

        let rhs = dtp_array_from_timestamp(0i64); // January 1, 1970, 00:00:00 UTC
        let comparison = DateTimePartsEncoding
            .compare(&lhs, &rhs, Operator::Eq)
            .unwrap()
            .unwrap();
        assert_eq!(comparison.as_bool_typed().unwrap().true_count().unwrap(), 0);
    }

    #[test]
    fn compare_date_time_parts_ne() {
        let lhs = dtp_array_from_timestamp(86400i64); // January 2, 1970, 00:00:00 UTC
        let rhs = dtp_array_from_timestamp(86401i64); // January 2, 1970, 00:00:01 UTC
        let comparison = DateTimePartsEncoding
            .compare(&lhs, &rhs, Operator::NotEq)
            .unwrap()
            .unwrap();
        assert_eq!(comparison.as_bool_typed().unwrap().true_count().unwrap(), 1);

        let rhs = dtp_array_from_timestamp(86400i64); // January 2, 1970, 00:00:00 UTC
        let comparison = DateTimePartsEncoding
            .compare(&lhs, &rhs, Operator::NotEq)
            .unwrap()
            .unwrap();
        assert_eq!(comparison.as_bool_typed().unwrap().true_count().unwrap(), 0);
    }

    #[test]
    fn compare_date_time_parts_lt() {
        let lhs = dtp_array_from_timestamp(0i64); // January 1, 1970, 01:00:00 UTC
        let rhs = dtp_array_from_timestamp(86400i64); // January 2, 1970, 00:00:00 UTC

        let comparison = DateTimePartsEncoding
            .compare(&lhs, &rhs, Operator::Lt)
            .unwrap()
            .unwrap();
        assert_eq!(comparison.as_bool_typed().unwrap().true_count().unwrap(), 1);
    }

    #[test]
    fn compare_date_time_parts_gt() {
        let lhs = dtp_array_from_timestamp(86400i64); // January 2, 1970, 02:00:00 UTC
        let rhs = dtp_array_from_timestamp(0i64); // January 1, 1970, 01:00:00 UTC

        let comparison = DateTimePartsEncoding
            .compare(&lhs, &rhs, Operator::Gt)
            .unwrap()
            .unwrap();
        assert_eq!(comparison.as_bool_typed().unwrap().true_count().unwrap(), 1);
    }

    #[test]
    fn compare_date_time_parts_narrowing() {
        let temporal_array = TemporalArray::new_timestamp(
            PrimitiveArray::new(buffer![0i64], Validity::NonNullable).into_array(),
            TimeUnit::S,
            Some("UTC".to_string()),
        );

        let lhs = DateTimePartsArray::try_new(
            DType::Extension(temporal_array.ext_dtype()),
            PrimitiveArray::new(buffer![0i32], Validity::NonNullable).into_array(),
            PrimitiveArray::new(buffer![0u32], Validity::NonNullable).into_array(),
            PrimitiveArray::new(buffer![0i64], Validity::NonNullable).into_array(),
        )
        .unwrap();

        // Timestamp with a value larger than i32::MAX.
        let rhs = dtp_array_from_timestamp(i64::MAX);

        let comparison = DateTimePartsEncoding
            .compare(&lhs, &rhs, Operator::Eq)
            .unwrap()
            .unwrap();
        assert_eq!(comparison.as_bool_typed().unwrap().true_count().unwrap(), 0);

        let comparison = DateTimePartsEncoding
            .compare(&lhs, &rhs, Operator::NotEq)
            .unwrap()
            .unwrap();
        assert_eq!(comparison.as_bool_typed().unwrap().true_count().unwrap(), 1);

        let comparison = DateTimePartsEncoding
            .compare(&lhs, &rhs, Operator::Lt)
            .unwrap()
            .unwrap();
        assert_eq!(comparison.as_bool_typed().unwrap().true_count().unwrap(), 1);

        let comparison = DateTimePartsEncoding
            .compare(&lhs, &rhs, Operator::Lte)
            .unwrap()
            .unwrap();
        assert_eq!(comparison.as_bool_typed().unwrap().true_count().unwrap(), 1);

        // `Operator::Gt` and `Operator::Gte` only cover the case of all lhs values
        // being larger. Therefore, these cases are not covered by unit tests.
    }
}
