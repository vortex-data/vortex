// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::ops::Sub;

use jiff::Span;
use vortex_dtype::ExtDType;
use vortex_dtype::datetime::Timestamp;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::PValue;
use crate::ScalarValue;
use crate::datetime::SpanExt;
use crate::extension::ExtScalarVTable;

/// Value for Timestamp extension scalar.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum TimestampValue {
    /// Timestamp with time zone.
    Zoned(jiff::Zoned),
    /// Timestamp without time zone.
    Unzoned(jiff::Timestamp),
}

impl Display for TimestampValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TimestampValue::Zoned(z) => write!(f, "{}", z),
            TimestampValue::Unzoned(ts) => write!(f, "{}", ts),
        }
    }
}

impl ExtScalarVTable for Timestamp {
    type Value = TimestampValue;

    fn unpack(&self, dtype: &ExtDType<Self>, storage: &ScalarValue) -> VortexResult<Self::Value> {
        let ScalarValue::Primitive(pvalue) = storage else {
            vortex_bail!("expected primitive scalar value for Timestamp dtype");
        };
        let v = pvalue.cast::<i64>();

        Ok(match &dtype.metadata().tz {
            None => {
                let epoch = jiff::Timestamp::UNIX_EPOCH;
                let span = Span::from_unit_length(v, dtype.metadata().unit);
                TimestampValue::Unzoned(epoch.checked_add(span)?)
            }
            Some(tz) => {
                let epoch = jiff::Timestamp::UNIX_EPOCH;
                let span = Span::from_unit_length(v, dtype.metadata().unit);
                TimestampValue::Zoned(epoch.checked_add(span)?.in_tz(tz.as_ref())?)
            }
        })
    }

    fn pack(&self, dtype: &ExtDType<Self>, value: &Self::Value) -> VortexResult<ScalarValue> {
        let span = match value {
            TimestampValue::Zoned(zoned) => zoned.timestamp().sub(jiff::Timestamp::UNIX_EPOCH),
            TimestampValue::Unzoned(datetime) => datetime.sub(jiff::Timestamp::UNIX_EPOCH),
        };
        let length = span.get_unit_length(dtype.metadata().unit);
        Ok(ScalarValue::Primitive(PValue::I64(length)))
    }

    fn validate(&self, value: &Self::Value, ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        // NOTE(ngates): this really is indicative that timestamp and timestamp_tz should be two
        //  different extension types, but we'll keep this for back compatibility for now.
        match (value, &ext_dtype.metadata().tz) {
            (TimestampValue::Zoned(_), None) => {
                vortex_bail!("expected unzoned timestamp value for unzoned timestamp dtype")
            }
            (TimestampValue::Unzoned(_), Some(_)) => {
                vortex_bail!("expected zoned timestamp value for zoned timestamp dtype")
            }
            _ => Ok(()),
        }
    }
}
