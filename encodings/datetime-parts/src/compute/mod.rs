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
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::TemporalArray;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_buffer::buffer;

    use crate::DateTimeParts;
    use crate::DateTimePartsArray;

    fn dtp_from_temporal(temporal: TemporalArray) -> DateTimePartsArray {
        DateTimeParts::try_from_temporal(temporal, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    }

    #[rstest]
    // Basic datetime arrays
    #[case::datetime_seconds(dtp_from_temporal(TemporalArray::new_timestamp(
        buffer![0i64, 86400, 172800, 259200, 345600].into_array(),
        TimeUnit::Seconds,
        Some("UTC".into()),
    )))]
    #[case::datetime_millis(dtp_from_temporal(TemporalArray::new_timestamp(
        buffer![0i64, 86400000, 172800000].into_array(),
        TimeUnit::Milliseconds,
        Some("UTC".into()),
    )))]
    #[case::datetime_micros(dtp_from_temporal(TemporalArray::new_timestamp(
        buffer![0i64, 86400000000, 172800000000].into_array(),
        TimeUnit::Microseconds,
        Some("UTC".into()),
    )))]
    #[case::datetime_nanos(dtp_from_temporal(TemporalArray::new_timestamp(
        buffer![0i64, 86400000000000].into_array(),
        TimeUnit::Nanoseconds,
        Some("UTC".into()),
    )))]
    // Nullable arrays
    #[case::datetime_nullable_seconds(dtp_from_temporal(TemporalArray::new_timestamp(
        PrimitiveArray::from_option_iter([Some(0i64), None, Some(86400), Some(172800), None]).into_array(),
        TimeUnit::Seconds,
        Some("UTC".into()),
    )))]
    // Edge cases
    #[case::datetime_single(dtp_from_temporal(TemporalArray::new_timestamp(
        buffer![1234567890i64].into_array(),
        TimeUnit::Seconds,
        Some("UTC".into()),
    )))]
    // Large arrays (> 1024 elements)
    #[case::datetime_large(dtp_from_temporal(TemporalArray::new_timestamp(
        PrimitiveArray::from_iter((0..1500).map(|i| i as i64 * 86400)).into_array(),
        TimeUnit::Seconds,
        Some("UTC".into()),
    )))]
    // Different time patterns
    #[case::datetime_with_subseconds(dtp_from_temporal(TemporalArray::new_timestamp(
        buffer![123456789i64, 234567890, 345678901, 456789012, 567890123].into_array(),
        TimeUnit::Milliseconds,
        Some("UTC".into()),
    )))]

    fn test_datetime_parts_consistency(#[case] array: DateTimePartsArray) {
        test_array_consistency(&array.into_array());
    }
}
