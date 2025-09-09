// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{CastKernel, CastKernelAdapter, cast};
use vortex_array::{Array, ArrayRef, IntoArray, register_kernel};
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::{DateTimePartsArray, DateTimePartsVTable};

impl CastKernel for DateTimePartsVTable {
    fn cast(&self, array: &DateTimePartsArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        if !array.dtype().eq_ignore_nullability(dtype) {
            return Ok(None);
        };

        Ok(Some(
            DateTimePartsArray::try_new(
                dtype.clone(),
                cast(
                    array.days().as_ref(),
                    &array.days().dtype().with_nullability(dtype.nullability()),
                )?,
                array.seconds().clone(),
                array.subseconds().clone(),
            )?
            .into_array(),
        ))
    }
}

register_kernel!(CastKernelAdapter(DateTimePartsVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::arrays::{PrimitiveArray, TemporalArray};
    use vortex_array::compute::cast;
    use vortex_array::validity::Validity;
    use vortex_array::{Array, ArrayRef, IntoArray};
    use vortex_buffer::buffer;
    use vortex_dtype::datetime::TimeUnit;
    use vortex_dtype::{DType, Nullability};

    use crate::DateTimePartsArray;

    fn date_time_array(validity: Validity) -> ArrayRef {
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
            TimeUnit::Milliseconds,
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
    fn test_cast_to_compatible_nullability(
        #[case] validity: Validity,
        #[case] cast_to_nullability: Nullability,
    ) {
        let array = date_time_array(validity);
        let new_dtype = array.dtype().with_nullability(cast_to_nullability);
        let result = cast(&array, &new_dtype);
        assert!(result.is_ok(), "{result:?}");
        assert_eq!(result.unwrap().dtype(), &new_dtype);
    }

    #[rstest]
    #[case(Validity::AllInvalid)]
    #[case(Validity::from_iter([true, false, true]))]
    fn test_bad_cast_fails(#[case] validity: Validity) {
        let array = date_time_array(validity);
        let result = cast(&array, &DType::Bool(Nullability::NonNullable));
        assert!(
            result.as_ref().is_err_and(|err| err.to_string().contains(
                "No compute kernel to cast array vortex.ext with dtype ext(vortex.timestamp, i64, ExtMetadata([2, 3, 0, 85, 84, 67]))? to bool"
            )),
            "Got error: {result:?}"
        );

        let result = cast(
            &array,
            &array.dtype().with_nullability(Nullability::NonNullable),
        );
        assert!(
            result.as_ref().is_err_and(|err| err
                .to_string()
                .contains("invalid cast from nullable to non-nullable")),
            "Got error: {result:?}"
        );
    }

    #[rstest]
    #[case(DateTimePartsArray::try_from(TemporalArray::new_timestamp(
        PrimitiveArray::from_iter([
            0i64,
            86_400_000,  // 1 day in ms
            172_800_000, // 2 days in ms
            259_200_000, // 3 days in ms
            345_600_000, // 4 days in ms
        ]).into_array(),
        TimeUnit::Milliseconds,
        Some("UTC".to_string())
    )).unwrap())]
    #[case(DateTimePartsArray::try_from(TemporalArray::new_timestamp(
        PrimitiveArray::from_option_iter([
            Some(0i64),
            None,
            Some(172_800_000), // 2 days in ms
            Some(259_200_000), // 3 days in ms
            None,
        ]).into_array(),
        TimeUnit::Milliseconds,
        Some("UTC".to_string())
    )).unwrap())]
    #[case(DateTimePartsArray::try_from(TemporalArray::new_timestamp(
        PrimitiveArray::from_iter([86_400_000_000_000i64]).into_array(), // 1 day in ns
        TimeUnit::Nanoseconds,
        Some("UTC".to_string())
    )).unwrap())]
    fn test_cast_datetime_parts_conformance(#[case] array: DateTimePartsArray) {
        use vortex_array::compute::conformance::cast::test_cast_conformance;
        test_cast_conformance(array.as_ref());
    }
}
