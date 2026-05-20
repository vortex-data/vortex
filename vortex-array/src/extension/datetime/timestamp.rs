// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Temporal extension data types.

use std::fmt;
use std::sync::Arc;

use jiff::Span;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::extension::datetime::TimeUnit;
use crate::scalar::ScalarValue;

/// Timestamp DType.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Timestamp;

impl Timestamp {
    /// Creates a new Timestamp extension =dtype with the given time unit and nullability.
    pub fn new(time_unit: TimeUnit, nullability: Nullability) -> ExtDType<Self> {
        Self::new_with_tz(time_unit, None, nullability)
    }

    /// Creates a new Timestamp extension dtype with the given time unit, timezone, and nullability.
    pub fn new_with_tz(
        time_unit: TimeUnit,
        timezone: Option<Arc<str>>,
        nullability: Nullability,
    ) -> ExtDType<Self> {
        ExtDType::try_new(
            TimestampOptions {
                unit: time_unit,
                tz: timezone,
            },
            DType::Primitive(PType::I64, nullability),
        )
        .vortex_expect("failed to create timestamp dtype")
    }

    /// Creates a new `Timestamp` extension dtype with the given options and nullability.
    pub fn new_with_options(options: TimestampOptions, nullability: Nullability) -> ExtDType<Self> {
        ExtDType::try_new(options, DType::Primitive(PType::I64, nullability))
            .vortex_expect("failed to create timestamp dtype")
    }
}

/// Options for the Timestamp DType.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TimestampOptions {
    /// The time unit of the timestamp.
    pub unit: TimeUnit,
    /// The timezone of the timestamp, if any.
    pub tz: Option<Arc<str>>,
}

impl fmt::Display for TimestampOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.tz {
            Some(tz) => write!(f, "{}, tz={}", self.unit, tz),
            None => write!(f, "{}", self.unit),
        }
    }
}

/// Unpacked value of a [`Timestamp`] extension scalar.
///
/// Each variant carries the raw storage value and an optional timezone.
pub enum TimestampValue<'a> {
    /// Seconds since the Unix epoch.
    Seconds(i64, Option<&'a Arc<str>>),
    /// Milliseconds since the Unix epoch.
    Milliseconds(i64, Option<&'a Arc<str>>),
    /// Microseconds since the Unix epoch.
    Microseconds(i64, Option<&'a Arc<str>>),
    /// Nanoseconds since the Unix epoch.
    Nanoseconds(i64, Option<&'a Arc<str>>),
}

impl fmt::Display for TimestampValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (span, tz) = match self {
            TimestampValue::Seconds(v, tz) => (Span::new().seconds(*v), *tz),
            TimestampValue::Milliseconds(v, tz) => (Span::new().milliseconds(*v), *tz),
            TimestampValue::Microseconds(v, tz) => (Span::new().microseconds(*v), *tz),
            TimestampValue::Nanoseconds(v, tz) => (Span::new().nanoseconds(*v), *tz),
        };
        let ts = jiff::Timestamp::UNIX_EPOCH + span;

        match tz {
            None => write!(f, "{ts}"),
            Some(tz) => {
                let adjusted_ts = ts.in_tz(tz.as_ref()).vortex_expect("unknown timezone");
                write!(f, "{adjusted_ts}",)
            }
        }
    }
}

impl ExtVTable for Timestamp {
    type Metadata = TimestampOptions;

    type NativeValue<'a> = TimestampValue<'a>;

    fn id(&self) -> ExtId {
        ExtId::new("vortex.timestamp")
    }

    // NOTE(ngates): unfortunately we're stuck with this hand-rolled serialization format for
    //  backwards compatibility.
    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        let mut bytes = Vec::with_capacity(4);
        let unit_tag: u8 = metadata.unit.into();

        bytes.push(unit_tag);

        // Encode time_zone as u16 length followed by utf8 bytes.
        match &metadata.tz {
            None => bytes.extend_from_slice(0u16.to_le_bytes().as_slice()),
            Some(tz) => {
                let tz_bytes = tz.as_bytes();
                let tz_len = u16::try_from(tz_bytes.len())
                    .unwrap_or_else(|err| vortex_panic!("tz did not fit in u16: {}", err));
                bytes.extend_from_slice(tz_len.to_le_bytes().as_slice());
                bytes.extend_from_slice(tz_bytes);
            }
        }

        Ok(bytes)
    }

    fn deserialize_metadata(&self, data: &[u8]) -> VortexResult<Self::Metadata> {
        vortex_ensure!(
            data.len() >= 3,
            "Timestamp metadata must have at least 3 bytes, got {}",
            data.len()
        );

        let tag = data[0];
        let time_unit = TimeUnit::try_from(tag)?;
        let tz_len_bytes: [u8; 2] = data[1..3]
            .try_into()
            .ok()
            .vortex_expect("Verified to have two bytes");
        let tz_len = u16::from_le_bytes(tz_len_bytes) as usize;
        if tz_len == 0 {
            return Ok(TimestampOptions {
                unit: time_unit,
                tz: None,
            });
        }

        // Attempt to load from len-prefixed bytes
        vortex_ensure!(
            data.len() >= 3 + tz_len,
            "Timestamp metadata is truncated: declared timezone length {} but only {} bytes available",
            tz_len,
            data.len() - 3
        );
        let tz_bytes = &data[3..3 + tz_len];
        let tz: Arc<str> = str::from_utf8(tz_bytes)
            .map_err(|e| vortex_err!("timezone is not valid utf8 string: {e}"))?
            .to_string()
            .into();
        Ok(TimestampOptions {
            unit: time_unit,
            tz: Some(tz),
        })
    }

    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        vortex_ensure!(
            matches!(ext_dtype.storage_dtype(), DType::Primitive(PType::I64, _)),
            "Timestamp storage dtype must be i64"
        );
        Ok(())
    }

    fn unpack_native<'a>(
        ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        let metadata = ext_dtype.metadata();
        let ts_value = storage_value.as_primitive().cast::<i64>()?;
        let tz = metadata.tz.as_ref();

        let (span, value) = match metadata.unit {
            TimeUnit::Nanoseconds => (
                Span::new().nanoseconds(ts_value),
                TimestampValue::Nanoseconds(ts_value, tz),
            ),
            TimeUnit::Microseconds => (
                Span::new().microseconds(ts_value),
                TimestampValue::Microseconds(ts_value, tz),
            ),
            TimeUnit::Milliseconds => (
                Span::new().milliseconds(ts_value),
                TimestampValue::Milliseconds(ts_value, tz),
            ),
            TimeUnit::Seconds => (
                Span::new().seconds(ts_value),
                TimestampValue::Seconds(ts_value, tz),
            ),
            TimeUnit::Days => vortex_bail!("Timestamp does not support Days time unit"),
        };

        // Validate the storage value is within the valid range for Timestamp.
        let ts = jiff::Timestamp::UNIX_EPOCH
            .checked_add(span)
            .map_err(|e| vortex_err!("Invalid timestamp scalar: {}", e))?;

        if let Some(tz) = tz {
            ts.in_tz(tz.as_ref())
                .map_err(|e| vortex_err!("Invalid timezone for timestamp scalar: {}", e))?;
        }

        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_error::VortexResult;

    use crate::dtype::DType;
    use crate::dtype::Nullability::Nullable;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;
    use crate::scalar::PValue;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;

    #[test]
    fn validate_timestamp_scalar() -> VortexResult<()> {
        let dtype = DType::Extension(Timestamp::new(TimeUnit::Seconds, Nullable).erased());
        Scalar::try_new(dtype, Some(ScalarValue::Primitive(PValue::I64(0))))?;

        Ok(())
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn reject_timestamp_with_invalid_timezone() {
        let dtype = DType::Extension(
            Timestamp::new_with_tz(
                TimeUnit::Seconds,
                Some(Arc::from("Not/A/Timezone")),
                Nullable,
            )
            .erased(),
        );
        let result = Scalar::try_new(dtype, Some(ScalarValue::Primitive(PValue::I64(0))));
        assert!(result.is_err());
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn display_timestamp_scalar() {
        // Local (no timezone) timestamp.
        let local_dtype = DType::Extension(Timestamp::new(TimeUnit::Seconds, Nullable).erased());
        let scalar = Scalar::new(local_dtype, Some(ScalarValue::Primitive(PValue::I64(0))));
        assert_eq!(format!("{}", scalar.as_extension()), "1970-01-01T00:00:00Z");

        // Zoned timestamp.
        let zoned_dtype = DType::Extension(
            Timestamp::new_with_tz(
                TimeUnit::Seconds,
                Some(Arc::from("America/New_York")),
                Nullable,
            )
            .erased(),
        );
        let scalar = Scalar::new(zoned_dtype, Some(ScalarValue::Primitive(PValue::I64(0))));
        assert_eq!(
            format!("{}", scalar.as_extension()),
            "1969-12-31T19:00:00-05:00[America/New_York]"
        );
    }

    #[test]
    fn deserialize_empty_metadata_returns_error() {
        use crate::dtype::extension::ExtVTable;

        let vtable = Timestamp;
        assert!(vtable.deserialize_metadata(&[]).is_err());
    }

    #[test]
    fn deserialize_too_short_metadata_returns_error() {
        use crate::dtype::extension::ExtVTable;

        let vtable = Timestamp;
        // Only 2 bytes - too short for the required 3-byte header.
        assert!(vtable.deserialize_metadata(&[0x00, 0x01]).is_err());
    }

    #[test]
    fn deserialize_truncated_timezone_returns_error() {
        use crate::dtype::extension::ExtVTable;

        let vtable = Timestamp;
        // Valid tag (0x00 = Nanoseconds), tz_len = 10 (little-endian: [0x0A, 0x00]),
        // but only 3 bytes of timezone data instead of the declared 10.
        let data = [0x00u8, 0x0A, 0x00, b'U', b'T', b'C'];
        assert!(vtable.deserialize_metadata(&data).is_err());
    }
}
