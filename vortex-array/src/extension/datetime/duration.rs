// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::registry::CachedId;

use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::extension::datetime::TimeUnit;
use crate::scalar::ScalarValue;

/// A signed duration stored as an `i64` count of seconds, milliseconds, microseconds, or
/// nanoseconds.
///
/// Every storage value is valid. Formatting falls back to the raw count when jiff cannot format it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Duration;

impl Duration {
    /// Creates a duration dtype with the given time unit and nullability.
    ///
    /// # Errors
    ///
    /// Returns an error for [`TimeUnit::Days`].
    pub fn try_new(time_unit: TimeUnit, nullability: Nullability) -> VortexResult<ExtDType<Self>> {
        ExtDType::try_new(time_unit, nullability)
    }

    /// Creates a duration dtype with the given time unit and nullability.
    ///
    /// # Panics
    ///
    /// Panics if `time_unit` is [`TimeUnit::Days`].
    pub fn new(time_unit: TimeUnit, nullability: Nullability) -> ExtDType<Self> {
        Self::try_new(time_unit, nullability).vortex_expect("duration time unit must be valid")
    }
}

/// Unpacked value of a [`Duration`] extension scalar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DurationValue {
    /// Seconds.
    Seconds(i64),
    /// Milliseconds.
    Milliseconds(i64),
    /// Microseconds.
    Microseconds(i64),
    /// Nanoseconds.
    Nanoseconds(i64),
}

impl fmt::Display for DurationValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (unit, value) = match *self {
            DurationValue::Seconds(v) => (TimeUnit::Seconds, v),
            DurationValue::Milliseconds(v) => (TimeUnit::Milliseconds, v),
            DurationValue::Microseconds(v) => (TimeUnit::Microseconds, v),
            DurationValue::Nanoseconds(v) => (TimeUnit::Nanoseconds, v),
        };

        // jiff cannot hold every i64 storage value.
        let Ok(span) = unit.to_jiff_span(value) else {
            return write!(f, "{value}{unit}");
        };

        write!(f, "{span}")
    }
}

impl ExtVTable for Duration {
    type Metadata = TimeUnit;
    type StoredMetadata = Self::Metadata;
    type NativeValue<'a> = DurationValue;

    fn id(&self) -> ExtId {
        static ID: CachedId = CachedId::new("vortex.duration");
        *ID
    }

    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(vec![u8::from(*metadata)])
    }

    fn deserialize_metadata(&self, data: &[u8]) -> VortexResult<Self::StoredMetadata> {
        vortex_ensure!(!data.is_empty(), "Duration metadata must not be empty");
        let tag = data[0];
        TimeUnit::try_from(tag)
    }

    fn metadata_from_storage(
        &self,
        stored_metadata: Self::StoredMetadata,
        _storage_dtype: &DType,
    ) -> VortexResult<Self::Metadata> {
        Ok(stored_metadata)
    }

    fn storage_dtype(_metadata: &Self::Metadata) -> DType {
        DType::Primitive(PType::I64, Nullability::NonNullable)
    }

    fn validate_metadata(metadata: &Self::Metadata) -> VortexResult<()> {
        vortex_ensure!(
            *metadata != TimeUnit::Days,
            "Duration time unit must be finer than days, got {metadata}"
        );

        Ok(())
    }

    fn unpack_native<'a>(
        ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        let value = storage_value.as_primitive().cast::<i64>()?;

        Ok(match *ext_dtype.metadata() {
            TimeUnit::Nanoseconds => DurationValue::Nanoseconds(value),
            TimeUnit::Microseconds => DurationValue::Microseconds(value),
            TimeUnit::Milliseconds => DurationValue::Milliseconds(value),
            TimeUnit::Seconds => DurationValue::Seconds(value),
            TimeUnit::Days => vortex_bail!("Duration does not support Days time unit"),
        })
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_error::VortexResult;

    use crate::dtype::DType;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::extension::ExtDTypePlugin;
    use crate::dtype::extension::ExtVTable;
    use crate::extension::datetime::Duration;
    use crate::extension::datetime::TimeUnit;
    use crate::scalar::PValue;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;

    #[test]
    fn validate_duration_scalar() -> VortexResult<()> {
        let dtype = DType::Extension(Duration::new(TimeUnit::Seconds, Nullable).erased());
        Scalar::try_new(dtype, Some(ScalarValue::Primitive(PValue::I64(0))))?;

        Ok(())
    }

    #[test]
    fn test_reject_duration_days() {
        assert!(Duration::try_new(TimeUnit::Days, Nullable).is_err());
    }

    #[test]
    fn test_read_rejects_non_i64_storage() {
        let result = Duration.deserialize(&[u8::from(TimeUnit::Seconds)], DType::Utf8(Nullable));
        assert!(result.is_err_and(|error| {
            error
                .to_string()
                .contains("vortex.duration storage must be i64?, got utf8?")
        }));
    }

    #[rstest]
    #[case::seconds_max(TimeUnit::Seconds, i64::MAX)]
    #[case::milliseconds_max(TimeUnit::Milliseconds, i64::MAX)]
    #[case::microseconds_max(TimeUnit::Microseconds, i64::MAX)]
    #[case::nanoseconds_min(TimeUnit::Nanoseconds, i64::MIN)]
    fn test_accept_duration_storage_extremes(
        #[case] unit: TimeUnit,
        #[case] value: i64,
    ) -> VortexResult<()> {
        let dtype = DType::Extension(Duration::new(unit, Nullable).erased());
        Scalar::try_new(dtype, Some(ScalarValue::from(value)))?;

        Ok(())
    }

    #[rstest]
    #[case::seconds(TimeUnit::Seconds, 10, "PT10S")]
    #[case::negative_ms(TimeUnit::Milliseconds, -250, "-PT0.25S")]
    #[case::zero_us(TimeUnit::Microseconds, 0, "PT0S")]
    #[case::overflow_seconds(TimeUnit::Seconds, i64::MAX, "9223372036854775807s")]
    #[case::overflow_nanos(TimeUnit::Nanoseconds, i64::MIN, "-9223372036854775808ns")]
    fn display_duration_scalar(
        #[case] unit: TimeUnit,
        #[case] value: i64,
        #[case] expected: &str,
    ) -> VortexResult<()> {
        let dtype = DType::Extension(Duration::new(unit, Nullable).erased());
        let scalar = Scalar::try_new(dtype, Some(ScalarValue::from(value)))?;
        assert_eq!(scalar.as_extension().to_string(), expected);

        Ok(())
    }

    #[test]
    fn deserialize_empty_metadata_returns_error() {
        assert!(Duration.deserialize_metadata(&[]).is_err());
    }

    #[test]
    fn deserialize_invalid_tag_returns_error() {
        assert!(Duration.deserialize_metadata(&[0xFF]).is_err());
    }
}
