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
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::TemporalArray;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use crate::DateTimeParts;
    use crate::DateTimePartsArray;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    fn dtp_from_temporal(temporal: TemporalArray) -> DateTimePartsArray {
        DateTimeParts::try_from_temporal(temporal, &mut SESSION.create_execution_ctx()).unwrap()
    }

    fn dtp_v2_from_temporal(temporal: TemporalArray) -> DateTimePartsArray {
        DateTimeParts::try_from_temporal_v2(temporal, &mut SESSION.create_execution_ctx()).unwrap()
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
    // Scaled subseconds (v2)
    #[case::datetime_micros_scaled_1000(dtp_v2_from_temporal(TemporalArray::new_timestamp(
        buffer![0i64, 1_000, 86_400_500_000, -86_400_250_000, 172_800_999_000].into_array(),
        TimeUnit::Microseconds,
        Some("UTC".into()),
    )))]
    #[case::datetime_nanos_scaled_1000(dtp_v2_from_temporal(TemporalArray::new_timestamp(
        buffer![0i64, 1_000, 86_400_000_500_000, -86_400_000_250_000].into_array(),
        TimeUnit::Nanoseconds,
        None,
    )))]
    #[case::datetime_nanos_scaled_1m(dtp_v2_from_temporal(TemporalArray::new_timestamp(
        buffer![0i64, 1_000_000, 86_400_500_000_000, -86_400_250_000_000].into_array(),
        TimeUnit::Nanoseconds,
        Some("UTC".into()),
    )))]
    #[case::datetime_nullable_scaled(dtp_v2_from_temporal(TemporalArray::new_timestamp(
        PrimitiveArray::from_option_iter([
            Some(0i64),
            None,
            Some(86_400_500_000),
            Some(-1_250_000),
            None,
        ]).into_array(),
        TimeUnit::Microseconds,
        Some("UTC".into()),
    )))]
    // Time arrays (v2)
    #[case::time_seconds(dtp_v2_from_temporal(TemporalArray::new_time(
        buffer![0i32, 3_600, 43_200, 86_399].into_array(),
        TimeUnit::Seconds,
    )))]
    #[case::time_millis(dtp_v2_from_temporal(TemporalArray::new_time(
        buffer![0i32, 3_600_500, 43_200_123, 86_399_999].into_array(),
        TimeUnit::Milliseconds,
    )))]
    #[case::time_micros_whole_seconds(dtp_v2_from_temporal(TemporalArray::new_time(
        buffer![0i64, 3_600_000_000, 43_200_000_000, 86_399_000_000].into_array(),
        TimeUnit::Microseconds,
    )))]
    #[case::time_micros_scaled(dtp_v2_from_temporal(TemporalArray::new_time(
        buffer![0i64, 3_600_500_000, 43_200_123_000, 86_399_999_000].into_array(),
        TimeUnit::Microseconds,
    )))]
    #[case::time_nanos_scaled_1m(dtp_v2_from_temporal(TemporalArray::new_time(
        buffer![0i64, 3_600_500_000_000, 43_200_123_000_000, 86_399_999_000_000].into_array(),
        TimeUnit::Nanoseconds,
    )))]
    #[case::time_nullable_micros(dtp_v2_from_temporal(TemporalArray::new_time(
        PrimitiveArray::from_option_iter([
            Some(0i64),
            None,
            Some(3_600_500_000),
            Some(86_399_000_000),
            None,
        ]).into_array(),
        TimeUnit::Microseconds,
    )))]
    #[case::time_nullable_millis(dtp_v2_from_temporal(TemporalArray::new_time(
        PrimitiveArray::from_option_iter([
            Some(0i32),
            None,
            Some(3_600_500),
            Some(86_399_999),
            None,
        ]).into_array(),
        TimeUnit::Milliseconds,
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
        let ctx = &mut SESSION.create_execution_ctx();
        test_array_consistency(&array.into_array(), ctx);
    }
}
