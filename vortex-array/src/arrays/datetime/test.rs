// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributorsuse crate::dtype::Nullability;

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::IntoArray;
use crate::Precision;
#[expect(deprecated)]
use crate::ToCanonical as _;
use crate::arrays::PrimitiveArray;
use crate::arrays::datetime::TemporalData;
use crate::assert_arrays_eq;
use crate::expr::gt;
use crate::expr::lit;
use crate::expr::root;
use crate::extension::datetime::TemporalMetadata;
use crate::extension::datetime::TimeUnit;
use crate::extension::datetime::Timestamp;
use crate::extension::datetime::TimestampOptions;
use crate::hash::ArrayEq;
use crate::scalar::Scalar;
use crate::validity::Validity;

macro_rules! test_temporal_roundtrip {
    ($prim:ty, $constructor:expr, $unit:expr) => {{
        let array = buffer![100 as $prim].into_array();
        let temporal: TemporalData = $constructor(array, $unit);

        assert_arrays_eq!(
            temporal.temporal_values(),
            PrimitiveArray::from_iter([100 as $prim])
        );
        assert_eq!(temporal.temporal_metadata().time_unit(), $unit);
    }};
}

macro_rules! test_success_case {
    ($name:ident, $prim:ty, $constructor:expr, $unit:expr) => {
        #[test]
        fn $name() {
            test_temporal_roundtrip!($prim, $constructor, $unit);
        }
    };
}

macro_rules! test_fail_case {
    ($name:ident, $prim:ty, $constructor:expr, $unit:expr) => {
        #[test]
        #[should_panic]
        fn $name() {
            test_temporal_roundtrip!($prim, $constructor, $unit)
        }
    };
}

// Time32 conformance tests
test_success_case!(
    test_roundtrip_time32_second,
    i32,
    TemporalData::new_time,
    TimeUnit::Seconds
);
test_success_case!(
    test_roundtrip_time32_millisecond,
    i32,
    TemporalData::new_time,
    TimeUnit::Milliseconds
);
test_fail_case!(
    test_fail_time32_micro,
    i32,
    TemporalData::new_time,
    TimeUnit::Microseconds
);
test_fail_case!(
    test_fail_time32_nano,
    i32,
    TemporalData::new_time,
    TimeUnit::Nanoseconds
);

// Time64 conformance tests
test_success_case!(
    test_roundtrip_time64_us,
    i64,
    TemporalData::new_time,
    TimeUnit::Microseconds
);
test_success_case!(
    test_roundtrip_time64_ns,
    i64,
    TemporalData::new_time,
    TimeUnit::Nanoseconds
);
test_fail_case!(
    test_fail_time64_ms,
    i64,
    TemporalData::new_time,
    TimeUnit::Milliseconds
);
test_fail_case!(
    test_fail_time64_s,
    i64,
    TemporalData::new_time,
    TimeUnit::Seconds
);
test_fail_case!(
    test_fail_time64_i32,
    i32,
    TemporalData::new_time,
    TimeUnit::Nanoseconds
);

// Date32 conformance tests
test_success_case!(
    test_roundtrip_date32,
    i32,
    TemporalData::new_date,
    TimeUnit::Days
);
test_fail_case!(
    test_fail_date32,
    i64,
    TemporalData::new_date,
    TimeUnit::Days
);

// Date64 conformance tests
test_success_case!(
    test_roundtrip_date64,
    i64,
    TemporalData::new_date,
    TimeUnit::Milliseconds
);
test_fail_case!(
    test_fail_date64,
    i32,
    TemporalData::new_date,
    TimeUnit::Milliseconds
);

// We test Timestamp explicitly to avoid the macro getting too complex.
#[test]
fn test_timestamp() {
    let ts = buffer![100i64].into_array();
    let ts_array = ts.into_array();

    for unit in [
        TimeUnit::Seconds,
        TimeUnit::Milliseconds,
        TimeUnit::Microseconds,
        TimeUnit::Nanoseconds,
    ] {
        for tz in [Some("UTC".into()), None] {
            let temporal_array = TemporalData::new_timestamp(ts_array.clone(), unit, tz.clone());

            assert_arrays_eq!(
                temporal_array.temporal_values(),
                PrimitiveArray::from_iter([100i64])
            );
            assert_eq!(
                temporal_array.temporal_metadata(),
                TemporalMetadata::Timestamp(&unit, &tz)
            );
        }
    }
}

#[test]
#[should_panic]
fn test_timestamp_fails_i32() {
    let ts = buffer![100i32].into_array();
    let ts_array = ts.into_array();

    TemporalData::new_timestamp(ts_array, TimeUnit::Seconds, None);
}

#[rstest]
#[case(Validity::NonNullable)]
#[case(Validity::AllValid)]
#[case(Validity::AllInvalid)]
#[case(Validity::from_iter([true, false, true]))]
fn test_validity_preservation(#[case] validity: Validity) {
    let milliseconds = PrimitiveArray::new(
        buffer![
            86_400i64,            // element with only day component
            86_400i64 + 1000,     // element with day + second components
            86_400i64 + 1000 + 1, // element with day + second + sub-second components
        ],
        validity.clone(),
    )
    .into_array();
    let temporal_array =
        TemporalData::new_timestamp(milliseconds, TimeUnit::Milliseconds, Some("UTC".into()));

    #[expect(deprecated)]
    let prim = temporal_array.temporal_values().to_primitive();
    assert!(
        prim.validity()
            .vortex_expect("temporal validity should be derivable")
            .array_eq(&validity, Precision::Ptr)
    );
}

#[test]
fn test222() -> VortexResult<()> {
    // Write file with MILLISECONDS timestamps
    let ts_array = PrimitiveArray::from_iter(vec![1704067200000i64, 1704153600000, 1704240000000])
        .into_array();
    let temporal = TemporalData::new_timestamp(ts_array, TimeUnit::Milliseconds, None);

    // Read with SECONDS filter scalar
    let filter_expr = gt(
        root(),
        lit(Scalar::extension::<Timestamp>(
            TimestampOptions {
                unit: TimeUnit::Seconds,
                tz: None,
            },
            Scalar::from(1704153600i64),
        )),
    );

    let _result = temporal.into_array().apply(&filter_expr);

    // let err = result.is_err().unwrap();
    // println!("Expected error: {}", err);

    Ok(())
}
