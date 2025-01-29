use vortex_array::compute::{try_cast, CastFn};
use vortex_array::{ArrayData, IntoArrayData};
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexResult};

use crate::{DateTimePartsArray, DateTimePartsEncoding};

impl CastFn<DateTimePartsArray> for DateTimePartsEncoding {
    fn cast(&self, array: &DateTimePartsArray, dtype: &DType) -> VortexResult<ArrayData> {
        if !array.dtype().eq_ignore_nullability(dtype) {
            vortex_bail!("cannot cast from {} to {}", array.dtype(), dtype);
        };

        Ok(DateTimePartsArray::try_new(
            dtype.clone(),
            try_cast(
                array.days().as_ref(),
                &array.days().dtype().with_nullability(dtype.nullability()),
            )?,
            array.seconds(),
            array.subsecond(),
        )?
        .into_array())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::array::{PrimitiveArray, TemporalArray};
    use vortex_array::compute::try_cast;
    use vortex_array::validity::Validity;
    use vortex_array::{ArrayData, IntoArrayData as _};
    use vortex_buffer::buffer;
    use vortex_datetime_dtype::TimeUnit;
    use vortex_dtype::{DType, Nullability};

    use crate::DateTimePartsArray;

    fn date_time_array(validity: Validity) -> ArrayData {
        DateTimePartsArray::try_from(TemporalArray::new_timestamp(
            PrimitiveArray::new(
                buffer![
                    86_400i64,            // element with only day component
                    86_400i64 + 1000,     // element with day + second components
                    86_400i64 + 1000 + 1, // element with day + second + sub-second components
                ],
                validity,
            )
            .into_array(),
            TimeUnit::Ms,
            Some("UTC".to_string()),
        ))
        .unwrap()
        .into_array()
    }

    #[rstest]
    #[case(Validity::NonNullable, Nullability::Nullable)]
    #[case(Validity::AllValid, Nullability::Nullable)]
    #[case(Validity::AllInvalid, Nullability::Nullable)]
    #[case(Validity::from_iter([true, false, true]), Nullability::Nullable)]
    #[case(Validity::NonNullable, Nullability::NonNullable)]
    #[case(Validity::AllValid, Nullability::NonNullable)]
    #[case(Validity::from_iter([true, true, true]), Nullability::Nullable)]
    fn test_cast_to_compatibile_nullability(
        #[case] validity: Validity,
        #[case] cast_to_nullability: Nullability,
    ) {
        let array = date_time_array(validity);
        let new_dtype = array.dtype().with_nullability(cast_to_nullability);
        let result = try_cast(&array, &new_dtype);
        assert!(result.is_ok(), "{:?}", result);
        assert_eq!(result.unwrap().dtype(), &new_dtype);
    }

    #[rstest]
    #[case(Validity::AllInvalid)]
    #[case(Validity::from_iter([true, false, true]))]
    fn test_bad_cast_fails(#[case] validity: Validity) {
        let array = date_time_array(validity);
        let result = try_cast(&array, &DType::Bool(Nullability::NonNullable));
        assert!(
            result
                .as_ref()
                .is_err_and(|err| err.to_string().contains("cannot cast from")),
            "{:?}",
            result
        );

        let result = try_cast(
            &array,
            &array.dtype().with_nullability(Nullability::NonNullable),
        );
        assert!(
            result.as_ref().is_err_and(|err| err
                .to_string()
                .contains("invalid cast from nullable to non-nullable")),
            "{:?}",
            result
        );
    }
}
