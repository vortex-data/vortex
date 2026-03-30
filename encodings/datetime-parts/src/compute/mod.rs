// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
mod compare;
mod filter;
pub(crate) mod is_constant;
pub(crate) mod kernel;
mod mask;
pub(super) mod rules;
mod slice;
mod take;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::TemporalArray;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_buffer::buffer;

    use crate::DateTimePartsArray;
    use crate::DateTimePartsData;

    #[rstest]
    // Basic datetime arrays
    #[case::datetime_seconds(DateTimePartsArray::try_from_data(DateTimePartsData::try_from(TemporalArray::new_timestamp(
        buffer![0i64, 86400, 172800, 259200, 345600].into_array(),
        TimeUnit::Seconds,
        Some("UTC".into()),
    )).unwrap()).unwrap())]
    #[case::datetime_millis(DateTimePartsArray::try_from_data(DateTimePartsData::try_from(TemporalArray::new_timestamp(
        buffer![0i64, 86400000, 172800000].into_array(),
        TimeUnit::Milliseconds,
        Some("UTC".into()),
    )).unwrap()).unwrap())]
    #[case::datetime_micros(DateTimePartsArray::try_from_data(DateTimePartsData::try_from(TemporalArray::new_timestamp(
        buffer![0i64, 86400000000, 172800000000].into_array(),
        TimeUnit::Microseconds,
        Some("UTC".into()),
    )).unwrap()).unwrap())]
    #[case::datetime_nanos(DateTimePartsArray::try_from_data(DateTimePartsData::try_from(TemporalArray::new_timestamp(
        buffer![0i64, 86400000000000].into_array(),
        TimeUnit::Nanoseconds,
        Some("UTC".into()),
    )).unwrap()).unwrap())]
    // Nullable arrays
    #[case::datetime_nullable_seconds(DateTimePartsArray::try_from_data(DateTimePartsData::try_from(TemporalArray::new_timestamp(
        PrimitiveArray::from_option_iter([Some(0i64), None, Some(86400), Some(172800), None]).into_array(),
        TimeUnit::Seconds,
        Some("UTC".into()),
    )).unwrap()).unwrap())]
    // Edge cases
    #[case::datetime_single(DateTimePartsArray::try_from_data(DateTimePartsData::try_from(TemporalArray::new_timestamp(
        buffer![1234567890i64].into_array(),
        TimeUnit::Seconds,
        Some("UTC".into()),
    )).unwrap()).unwrap())]
    // Large arrays (> 1024 elements)
    #[case::datetime_large(DateTimePartsArray::try_from_data(DateTimePartsData::try_from(TemporalArray::new_timestamp(
        PrimitiveArray::from_iter((0..1500).map(|i| i as i64 * 86400)).into_array(),
        TimeUnit::Seconds,
        Some("UTC".into()),
    )).unwrap()).unwrap())]
    // Different time patterns
    #[case::datetime_with_subseconds(DateTimePartsArray::try_from_data(DateTimePartsData::try_from(TemporalArray::new_timestamp(
        buffer![123456789i64, 234567890, 345678901, 456789012, 567890123].into_array(),
        TimeUnit::Milliseconds,
        Some("UTC".into()),
    )).unwrap()).unwrap())]

    fn test_datetime_parts_consistency(#[case] array: DateTimePartsArray) {
        test_array_consistency(&array.into_array());
    }
}
