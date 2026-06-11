// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arbitrary generation of temporal extension dtypes and in-range storage values.

use std::sync::Arc;

use arbitrary::Result;
use arbitrary::Unstructured;

use crate::dtype::Nullability;
use crate::dtype::extension::ExtDTypeRef;
use crate::extension::datetime::Date;
use crate::extension::datetime::TemporalMetadata;
use crate::extension::datetime::Time;
use crate::extension::datetime::TimeUnit;
use crate::extension::datetime::Timestamp;

/// Timezones used for arbitrary zoned timestamps. Must be valid IANA names so that scalar
/// validation (`jiff::Timestamp::in_tz`) succeeds.
const TIMEZONES: &[&str] = &["UTC", "America/New_York", "Asia/Tokyo"];

/// Conservative bounds on days relative to the Unix epoch that stay within jiff's supported
/// civil date range (years -9999 to 9999), so generated values always unpack to valid
/// temporal scalars.
const MIN_EPOCH_DAYS: i64 = -4_000_000;
const MAX_EPOCH_DAYS: i64 = 2_900_000;

const SECONDS_PER_DAY: i64 = 86_400;
const MS_PER_DAY: i64 = SECONDS_PER_DAY * 1_000;
const US_PER_DAY: i64 = MS_PER_DAY * 1_000;
const NS_PER_DAY: i64 = US_PER_DAY * 1_000;

/// Generates an arbitrary temporal extension dtype: a [`Timestamp`] (optionally zoned),
/// [`Date`], or [`Time`] with a valid time unit for that type.
pub fn random_temporal_ext_dtype(
    u: &mut Unstructured<'_>,
    nullability: Nullability,
) -> Result<ExtDTypeRef> {
    Ok(match u.int_in_range(0..=2)? {
        0 => {
            let unit = random_sub_day_unit(u)?;
            let tz = u
                .arbitrary::<bool>()?
                .then(|| u.choose(TIMEZONES).map(|tz| Arc::<str>::from(*tz)))
                .transpose()?;
            Timestamp::new_with_tz(unit, tz, nullability).erased()
        }
        1 => {
            let unit = if u.arbitrary()? {
                TimeUnit::Days
            } else {
                TimeUnit::Milliseconds
            };
            Date::new(unit, nullability).erased()
        }
        2 => Time::new(random_sub_day_unit(u)?, nullability).erased(),
        _ => unreachable!("Number out of range"),
    })
}

fn random_sub_day_unit(u: &mut Unstructured<'_>) -> Result<TimeUnit> {
    Ok(match u.int_in_range(0..=3)? {
        0 => TimeUnit::Nanoseconds,
        1 => TimeUnit::Microseconds,
        2 => TimeUnit::Milliseconds,
        3 => TimeUnit::Seconds,
        _ => unreachable!("Number out of range"),
    })
}

/// Generates a storage value within the valid range for the given temporal metadata, so the
/// value unpacks to a valid jiff temporal value.
pub fn random_temporal_storage_value(
    u: &mut Unstructured<'_>,
    metadata: &TemporalMetadata<'_>,
) -> Result<i64> {
    let (min, max) = storage_value_range(metadata);
    u.int_in_range(min..=max)
}

/// The inclusive range of valid storage values for the given temporal metadata.
fn storage_value_range(metadata: &TemporalMetadata<'_>) -> (i64, i64) {
    match metadata {
        TemporalMetadata::Time(unit) => match unit {
            TimeUnit::Seconds => (0, SECONDS_PER_DAY - 1),
            TimeUnit::Milliseconds => (0, MS_PER_DAY - 1),
            TimeUnit::Microseconds => (0, US_PER_DAY - 1),
            TimeUnit::Nanoseconds => (0, NS_PER_DAY - 1),
            TimeUnit::Days => unreachable!("Time does not support Days unit"),
        },
        TemporalMetadata::Date(unit) => match unit {
            TimeUnit::Days => (MIN_EPOCH_DAYS, MAX_EPOCH_DAYS),
            TimeUnit::Milliseconds => (MIN_EPOCH_DAYS * MS_PER_DAY, MAX_EPOCH_DAYS * MS_PER_DAY),
            _ => unreachable!("Date only supports Days and Milliseconds units"),
        },
        TemporalMetadata::Timestamp(unit, _) => match unit {
            TimeUnit::Seconds => (
                MIN_EPOCH_DAYS * SECONDS_PER_DAY,
                MAX_EPOCH_DAYS * SECONDS_PER_DAY,
            ),
            TimeUnit::Milliseconds => (MIN_EPOCH_DAYS * MS_PER_DAY, MAX_EPOCH_DAYS * MS_PER_DAY),
            TimeUnit::Microseconds => (MIN_EPOCH_DAYS * US_PER_DAY, MAX_EPOCH_DAYS * US_PER_DAY),
            // Any i64 nanosecond count is within roughly +/-292 years of the epoch, which is
            // always a valid jiff timestamp.
            TimeUnit::Nanoseconds => (i64::MIN, i64::MAX),
            TimeUnit::Days => unreachable!("Timestamp does not support Days unit"),
        },
    }
}

#[cfg(test)]
mod tests {
    use arbitrary::Unstructured;
    use vortex_error::VortexResult;

    use super::random_temporal_ext_dtype;
    use super::random_temporal_storage_value;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::extension::datetime::AnyTemporal;
    use crate::scalar::PValue;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;

    #[test]
    fn random_temporal_values_unpack_to_valid_scalars() -> VortexResult<()> {
        // A fixed pseudo-random byte stream is enough to cover all dtype kinds and units.
        let bytes = (0u32..4096)
            .map(|i| (i * 31 % 251) as u8)
            .collect::<Vec<_>>();
        let mut u = Unstructured::new(&bytes);

        while u.len() > 64 {
            let Ok(ext_dtype) = random_temporal_ext_dtype(&mut u, Nullability::NonNullable) else {
                break;
            };
            let metadata = ext_dtype.metadata::<AnyTemporal>();
            let Ok(value) = random_temporal_storage_value(&mut u, &metadata) else {
                break;
            };
            // Values must unpack to a valid temporal value.
            metadata.to_jiff(value)?;

            let pvalue = match ext_dtype.storage_dtype() {
                DType::Primitive(PType::I32, _) => PValue::I32(i32::try_from(value)?),
                DType::Primitive(PType::I64, _) => PValue::I64(value),
                d => unreachable!("unexpected storage dtype {d}"),
            };
            // Scalar construction validates the value against the extension dtype.
            Scalar::try_new(
                DType::Extension(ext_dtype),
                Some(ScalarValue::Primitive(pvalue)),
            )?;
        }
        Ok(())
    }
}
