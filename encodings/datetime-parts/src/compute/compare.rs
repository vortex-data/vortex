use vortex_array::array::ConstantArray;
use vortex_array::compute::{and, compare, try_cast, CompareFn, Operator};
use vortex_array::{Array, IntoArray};
use vortex_datetime_dtype::TemporalMetadata;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::array::{DateTimePartsArray, DateTimePartsEncoding};
use crate::timestamp;

impl CompareFn<DateTimePartsArray> for DateTimePartsEncoding {
    /// Compares two arrays and returns a new boolean array with the result of the comparison.
    /// Or, returns None if comparison is not supported.
    ///
    /// # NOTE: Only `Operator::Eq` is currently supported.
    fn compare(
        &self,
        lhs: &DateTimePartsArray,
        rhs: &Array,
        operator: Operator,
    ) -> VortexResult<Option<Array>> {
        if operator != Operator::Eq {
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

        let compare_dtp = |lhs: &Array, rhs: i64, operator: Operator| -> VortexResult<Array> {
            match try_cast(ConstantArray::new(rhs, lhs.len()), lhs.dtype()) {
                Ok(casted) => compare(lhs, casted, operator),
                // The narrowing cast failed. Therefore, attempt to derive the result from the operator.
                Err(_) => {
                    let constant_value = match operator {
                        Operator::Eq => false,
                        _ => unreachable!("operator {} not supported", operator),
                    };
                    Ok(ConstantArray::new(constant_value, lhs.len()).into_array())
                }
            }
        };

        let mut comparison = compare_dtp(&lhs.days(), ts_parts.days, operator)?;
        // Prefer `compute_max::<bool>` over `compute_true_count` as it ignores `Null`.
        if comparison.statistics().compute_max::<bool>() == Some(false) {
            return Ok(Some(comparison));
        }

        comparison = and(
            compare_dtp(&lhs.seconds(), ts_parts.seconds, operator)?,
            comparison,
        )?;

        if comparison.statistics().compute_max::<bool>() == Some(false) {
            return Ok(Some(comparison));
        }

        comparison = and(
            compare_dtp(&lhs.subseconds(), ts_parts.subseconds, operator)?,
            comparison,
        )?;

        Ok(Some(comparison))
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use vortex_array::array::{PrimitiveArray, TemporalArray};
    use vortex_array::compute::Operator;
    use vortex_array::validity::Validity;
    use vortex_array::IntoArray;
    use vortex_buffer::buffer;
    use vortex_datetime_dtype::{TimeUnit, TIME_ID};
    use vortex_dtype::{ExtDType, NativePType, Nullability, PType};

    use super::*;

    fn dtp_array() -> DateTimePartsArray {
        DateTimePartsArray::try_new(
            DType::Extension(Arc::new(ExtDType::new(
                TIME_ID.clone(),
                Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
                Some(TemporalMetadata::Time(TimeUnit::S).into()),
            ))),
            buffer![1i64].into_array(),
            buffer![0i64].into_array(),
            buffer![0i64].into_array(),
        )
        .unwrap()
    }

    fn dtp_array_from_timestamp<T: NativePType>(value: T) -> DateTimePartsArray {
        DateTimePartsArray::try_from(TemporalArray::new_timestamp(
            PrimitiveArray::new(buffer![value], Validity::NonNullable).into_array(),
            TimeUnit::S,
            Some("UTC".to_string()),
        ))
        .expect("Failed to construct DateTimePartsArray from TemporalArray")
    }

    #[test]
    fn compare_date_time_parts_equal() {
        let lhs = dtp_array_from_timestamp(86400i64); // January 2, 1970, 00:00:00 UTC
        let rhs = dtp_array_from_timestamp(86400i64); // January 2, 1970, 00:00:00 UTC
        let comparison = DateTimePartsEncoding
            .compare(&lhs, &rhs, Operator::Eq)
            .unwrap()
            .unwrap();

        assert_eq!(comparison.statistics().compute_true_count().unwrap(), 1);

        let rhs = dtp_array(); // January 2, 1970, 00:00:00 UTC
        let comparison = DateTimePartsEncoding
            .compare(&lhs, &rhs, Operator::Eq)
            .unwrap()
            .unwrap();

        assert_eq!(comparison.statistics().compute_true_count().unwrap(), 1);
    }

    #[test]
    fn compare_date_time_parts_not_equal() {
        let lhs = dtp_array_from_timestamp(0i64); // January 1, 1970, 00:00:00 UTC
        let rhs = dtp_array_from_timestamp(86400i64); // January 2, 1970, 00:00:00 UTC
        let comparison = DateTimePartsEncoding
            .compare(&lhs, &rhs, Operator::Eq)
            .unwrap()
            .unwrap();

        assert_eq!(comparison.statistics().compute_true_count().unwrap(), 0);

        let rhs = dtp_array(); // January 2, 1970, 00:00:00 UTC
        let comparison = DateTimePartsEncoding
            .compare(&lhs, &rhs, Operator::Eq)
            .unwrap()
            .unwrap();

        assert_eq!(comparison.statistics().compute_true_count().unwrap(), 0);
    }
}
