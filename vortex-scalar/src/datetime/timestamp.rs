// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;
use std::sync::Arc;

use jiff::Span;
use vortex_dtype::DType;
use vortex_dtype::datetime::TimeUnit;
use vortex_dtype::datetime::Timestamp;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ScalarValue;
use crate::datetime::SpanExt;
use crate::extension::ExtScalarVTable;

/// Value representation for Timestamp extension scalars.
pub enum TimestampValue<'a> {
    Seconds(Option<i64>, Option<&'a Arc<str>>),
    Milliseconds(Option<i64>, Option<&'a Arc<str>>),
    Microseconds(Option<i64>, Option<&'a Arc<str>>),
    Nanoseconds(Option<i64>, Option<&'a Arc<str>>),
}

impl ExtScalarVTable for Timestamp {
    type Value<'a> = TimestampValue<'a>;

    fn unpack(
        &self,
        metadata: &Self::Metadata,
        _storage_dtype: &DType,
        storage_value: Option<&ScalarValue>,
    ) -> Self::Value<'_> {
        let ts_value = storage_value.map(|s| s.as_primitive().cast::<i64>());
        let tz = metadata.tz.as_ref();
        match metadata.unit {
            TimeUnit::Nanoseconds => TimestampValue::Nanoseconds(ts_value, tz),
            TimeUnit::Microseconds => TimestampValue::Microseconds(ts_value, tz),
            TimeUnit::Milliseconds => TimestampValue::Milliseconds(ts_value, tz),
            TimeUnit::Seconds => TimestampValue::Seconds(ts_value, tz),
            TimeUnit::Days => unreachable!(),
        }
    }

    fn fmt_scalar(
        &self,
        metadata: &Self::Metadata,
        _storage_dtype: &DType,
        storage_value: &ScalarValue,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        let span =
            Span::from_unit_length(storage_value.as_primitive().cast::<i64>(), metadata.unit);
        let ts = jiff::Timestamp::UNIX_EPOCH + span;
        match &metadata.tz {
            None => {
                write!(f, "{}", ts)
            }
            Some(tz) => {
                write!(f, "{}", ts.in_tz(tz.as_ref()).map_err(|_| std::fmt::Error)?)
            }
        }
    }

    fn validate_scalar(
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
