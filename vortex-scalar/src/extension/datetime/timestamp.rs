// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ExtScalarVTable`] implementation for [`Timestamp`] extension scalars.

use std::fmt::Formatter;
use std::sync::Arc;

use jiff::Span;
use vortex_dtype::DType;
use vortex_dtype::datetime::TimeUnit;
use vortex_dtype::datetime::Timestamp;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ScalarValue;
use crate::extension::ExtScalarVTable;
use crate::extension::datetime::SpanExt;

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

impl std::fmt::Display for TimestampValue<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let (span, tz) = match self {
            TimestampValue::Seconds(v, tz) => (Span::new().seconds(*v), *tz),
            TimestampValue::Milliseconds(v, tz) => (Span::new().milliseconds(*v), *tz),
            TimestampValue::Microseconds(v, tz) => (Span::new().microseconds(*v), *tz),
            TimestampValue::Nanoseconds(v, tz) => (Span::new().nanoseconds(*v), *tz),
        };
        let ts = jiff::Timestamp::UNIX_EPOCH + span;

        match tz {
            None => {
                write!(f, "{}", ts)
            }
            Some(tz) => {
                write!(f, "{}", ts.in_tz(tz.as_ref()).map_err(|_| std::fmt::Error)?)
            }
        }
    }
}

impl ExtScalarVTable for Timestamp {
    type Value<'a> = TimestampValue<'a>;

    fn unpack<'a>(
        &self,
        metadata: &'a Self::Metadata,
        _storage_dtype: &'a DType,
        storage_value: &'a ScalarValue,
    ) -> Self::Value<'a> {
        let ts_value = storage_value.as_primitive().cast::<i64>();
        let tz = metadata.tz.as_ref();

        match metadata.unit {
            TimeUnit::Nanoseconds => TimestampValue::Nanoseconds(ts_value, tz),
            TimeUnit::Microseconds => TimestampValue::Microseconds(ts_value, tz),
            TimeUnit::Milliseconds => TimestampValue::Milliseconds(ts_value, tz),
            TimeUnit::Seconds => TimestampValue::Seconds(ts_value, tz),
            TimeUnit::Days => unreachable!(),
        }
    }

    fn validate_scalar_value(
        &self,
        metadata: &Self::Metadata,
        _storage_dtype: &DType,
        storage_value: &ScalarValue,
    ) -> VortexResult<()> {
        // Validate the storage value is within the valid range for Timestamp
        let span =
            Span::from_unit_length(storage_value.as_primitive().cast::<i64>(), metadata.unit);

        let ts = jiff::Timestamp::UNIX_EPOCH
            .checked_add(span)
            .map_err(|e| vortex_err!("Invalid timestamp scalar: {}", e))?;

        if let Some(tz) = &metadata.tz {
            ts.in_tz(tz.as_ref())
                .map_err(|e| vortex_err!("Invalid timezone for timestamp scalar: {}", e))?;
        }

        Ok(())
    }
}
