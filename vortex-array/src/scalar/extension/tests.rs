// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexResult;

use crate::dtype::Nullability;
use crate::extension::datetime::Date;
use crate::extension::datetime::Time;
use crate::extension::datetime::TimeUnit;
use crate::extension::datetime::Timestamp;
use crate::scalar::PValue;
use crate::scalar::ScalarValue;
use crate::scalar::extension::ExtScalarValue;
use crate::scalar::extension::ExtScalarValueRef;

#[test]
fn try_new_date_valid() -> VortexResult<()> {
    let ext_dtype = Date::new(TimeUnit::Days, Nullability::NonNullable);
    let storage = ScalarValue::Primitive(PValue::I32(100));

    let sv = ExtScalarValue::<Date>::try_new(&ext_dtype, storage.clone())?;

    assert_eq!(sv.id().as_ref(), "vortex.date");
    assert_eq!(sv.storage_value(), &storage);
    assert_eq!(sv.vtable(), &Date);
    Ok(())
}

#[test]
fn try_new_time_rejects_out_of_range() -> VortexResult<()> {
    let ext_dtype = Time::new(TimeUnit::Seconds, Nullability::NonNullable);

    // 25 hours in seconds exceeds valid time-of-day range.
    let too_large = ScalarValue::Primitive(PValue::I32(90_000));
    assert!(ExtScalarValue::<Time>::try_new(&ext_dtype, too_large).is_err());

    // Negative time is invalid.
    let negative = ScalarValue::Primitive(PValue::I32(-1));
    assert!(ExtScalarValue::<Time>::try_new(&ext_dtype, negative).is_err());

    // Just under 24h should succeed.
    let just_valid = ScalarValue::Primitive(PValue::I32(86_399));
    assert!(ExtScalarValue::<Time>::try_new(&ext_dtype, just_valid).is_ok());

    Ok(())
}

#[cfg_attr(miri, ignore)]
#[test]
fn try_new_timestamp_rejects_invalid_tz() -> VortexResult<()> {
    let ext_dtype = Timestamp::new_with_tz(
        TimeUnit::Seconds,
        Some(Arc::from("Not/A/Timezone")),
        Nullability::NonNullable,
    );

    let storage = ScalarValue::Primitive(PValue::I64(0));
    assert!(ExtScalarValue::<Timestamp>::try_new(&ext_dtype, storage).is_err());

    Ok(())
}

#[test]
fn typed_erased_downcast_roundtrip() -> VortexResult<()> {
    let ext_dtype = Timestamp::new(TimeUnit::Microseconds, Nullability::NonNullable);
    let storage = ScalarValue::Primitive(PValue::I64(1_000_000));

    let typed = ExtScalarValue::<Timestamp>::try_new(&ext_dtype, storage)?;
    let erased: ExtScalarValueRef = typed.clone().erased();

    assert_eq!(erased.id().as_ref(), "vortex.timestamp");
    assert_eq!(erased.storage_value(), typed.storage_value());

    let roundtripped = erased.downcast::<Timestamp>();
    assert_eq!(typed, roundtripped);
    Ok(())
}

#[test]
fn downcast_wrong_type_fails() -> VortexResult<()> {
    let ext_dtype = Date::new(TimeUnit::Days, Nullability::NonNullable);
    let storage = ScalarValue::Primitive(PValue::I32(42));

    let erased = ExtScalarValue::<Date>::try_new(&ext_dtype, storage)?.erased();

    // Wrong types return Err, preserving the value for retry.
    let erased = erased.try_downcast::<Time>().unwrap_err();
    let erased = erased.try_downcast::<Timestamp>().unwrap_err();

    // Correct type succeeds after failed attempts.
    let typed = erased.downcast::<Date>();
    assert_eq!(typed.vtable(), &Date);
    assert_eq!(
        typed.storage_value(),
        &ScalarValue::Primitive(PValue::I32(42))
    );

    Ok(())
}

#[test]
fn try_new_rejects_unsupported_unit_and_overflow() -> VortexResult<()> {
    // Timestamp rejects Days time unit.
    let ts_days = Timestamp::new(TimeUnit::Days, Nullability::NonNullable);
    assert!(
        ExtScalarValue::<Timestamp>::try_new(&ts_days, ScalarValue::Primitive(PValue::I64(0)))
            .is_err()
    );

    // Time in milliseconds rejects values exceeding 24h (86_400_000ms).
    let time_ms = Time::new(TimeUnit::Milliseconds, Nullability::NonNullable);
    assert!(
        ExtScalarValue::<Time>::try_new(&time_ms, ScalarValue::Primitive(PValue::I32(86_400_001)))
            .is_err()
    );

    // Time in nanoseconds rejects values exceeding 24h.
    let time_ns = Time::new(TimeUnit::Nanoseconds, Nullability::NonNullable);
    let nanos_25h = 25 * 3600 * 1_000_000_000i64;
    assert!(
        ExtScalarValue::<Time>::try_new(&time_ns, ScalarValue::Primitive(PValue::I64(nanos_25h)))
            .is_err()
    );

    Ok(())
}

#[test]
fn erased_display() -> VortexResult<()> {
    let ext_dtype = Date::new(TimeUnit::Days, Nullability::NonNullable);
    let storage = ScalarValue::Primitive(PValue::I32(365));

    let erased = ExtScalarValue::<Date>::try_new(&ext_dtype, storage)?.erased();
    let display = format!("{erased}");

    assert!(display.contains("vortex.date"));
    assert!(display.contains("365"));

    Ok(())
}
