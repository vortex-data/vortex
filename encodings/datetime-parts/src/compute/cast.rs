// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_error::VortexResult;

use crate::DateTimeParts;
use crate::array::DateTimePartsArrayExt;
impl CastReduce for DateTimeParts {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        if !array.dtype().eq_ignore_nullability(dtype) {
            return Ok(None);
        };

        Ok(Some(
            DateTimeParts::try_new(
                dtype.clone(),
                array
                    .days()
                    .cast(array.days().dtype().with_nullability(dtype.nullability()))?,
                array.seconds().clone(),
                array.subseconds().clone(),
            )?
            .into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::TemporalArray;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    use crate::DateTimeParts;
    use crate::DateTimePartsArray;

    fn date_time_array(validity: Validity) -> ArrayRef {
        DateTimeParts::try_from_temporal(TemporalArray::new_timestamp(
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
            Some("UTC".into()),
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
        let result = array.cast(new_dtype.clone());
        assert!(result.is_ok(), "{result:?}");
        assert_eq!(result.unwrap().dtype(), &new_dtype);
    }

    #[rstest]
    #[case(Validity::AllInvalid)]
    #[case(Validity::from_iter([true, false, true]))]
    fn test_bad_cast_fails(#[case] validity: Validity) {
        let array = date_time_array(validity);
        // Cast to incompatible type - force evaluation via to_canonical
        #[expect(deprecated)]
        let result = array
            .cast(DType::Bool(Nullability::NonNullable))
            .and_then(|a| a.to_canonical().map(|c| c.into_array()));
        assert!(result.is_err(), "Expected error, got: {result:?}");

        // Cast nullable with nulls to non-nullable - force evaluation via to_canonical
        #[expect(deprecated)]
        let result = array
            .cast(array.dtype().with_nullability(Nullability::NonNullable))
            .and_then(|a| a.to_canonical().map(|c| c.into_array()));
        assert!(result.is_err(), "Expected error, got: {result:?}");
    }

    #[rstest]
    #[case(DateTimeParts::try_from_temporal(TemporalArray::new_timestamp(
        buffer![
            0i64,
            86_400_000,  // 1 day in ms
            172_800_000, // 2 days in ms
            259_200_000, // 3 days in ms
            345_600_000, // 4 days in ms
        ].into_array(),
        TimeUnit::Milliseconds,
        Some("UTC".into())
    )).unwrap())]
    #[case(DateTimeParts::try_from_temporal(TemporalArray::new_timestamp(
        PrimitiveArray::from_option_iter([
            Some(0i64),
            None,
            Some(172_800_000), // 2 days in ms
            Some(259_200_000), // 3 days in ms
            None,
        ]).into_array(),
        TimeUnit::Milliseconds,
        Some("UTC".into())
    )).unwrap())]
    #[case(DateTimeParts::try_from_temporal(TemporalArray::new_timestamp(
        buffer![86_400_000_000_000i64].into_array(), // 1 day in ns
        TimeUnit::Nanoseconds,
        Some("UTC".into())
    )).unwrap())]
    fn test_cast_datetime_parts_conformance(#[case] array: DateTimePartsArray) {
        use vortex_array::compute::conformance::cast::test_cast_conformance;
        test_cast_conformance(&array.into_array());
    }
}
