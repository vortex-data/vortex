// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use jiff::Span;
use vortex_dtype::DType;
use vortex_dtype::ExtDType;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_dtype::datetime::Timestamp;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::Scalar;
use crate::ScalarValue;
use crate::datetime::SpanExt;
use crate::extension::ExtScalarVTable;

#[derive(Clone, Debug, Hash)]
pub enum TimestampValue {
    Zoned(jiff::Zoned),
    Unzoned(jiff::civil::DateTime),
}

impl PartialEq for TimestampValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (TimestampValue::Zoned(a), TimestampValue::Zoned(b)) => a == b,
            (TimestampValue::Unzoned(a), TimestampValue::Unzoned(b)) => a == b,
            _ => false,
        }
    }
}

impl Display for TimestampValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TimestampValue::Zoned(z) => write!(f, "{}", z),
            TimestampValue::Unzoned(dt) => write!(f, "{}", dt),
        }
    }
}

impl ExtScalarVTable for Timestamp {
    type Value = TimestampValue;

    fn zero(&self, metadata: &Self::Metadata) -> Self::Value {
        match &metadata.tz {
            None => {
                let epoch = jiff::civil::DateTime::new(1970, 1, 1, 0, 0, 0, 0)
                    .vortex_expect("failed to create epoch datetime");
                TimestampValue::Unzoned(epoch)
            }
            Some(tz) => {
                let epoch = jiff::Timestamp::UNIX_EPOCH;
                TimestampValue::Zoned(
                    epoch
                        .in_tz(tz.as_ref())
                        .vortex_expect("failed to create zoned epoch"),
                )
            }
        }
    }

    fn unpack(&self, dtype: &ExtDType<Self>, storage: &ScalarValue) -> VortexResult<Self::Value> {
        let v = storage
            .as_pvalue()?
            .vortex_expect("storage is non-null")
            .cast::<i64>();

        Ok(match &dtype.metadata().tz {
            None => {
                let epoch = jiff::civil::DateTime::new(1970, 1, 1, 0, 0, 0, 0)?;
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

    fn pack(
        &self,
        metadata: &Self::Metadata,
        value: Self::Value,
        nullability: Nullability,
    ) -> VortexResult<Scalar> {
        match value {
            TimestampValue::Zoned(zoned) => {
                let epoch = jiff::Timestamp::UNIX_EPOCH;
                let span = zoned.timestamp() - epoch;
                let length = span.get_unit_length(metadata.unit);
                Ok(Scalar::primitive(length, nullability))
            }
            TimestampValue::Unzoned(datetime) => {
                let epoch = jiff::civil::DateTime::new(1970, 1, 1, 0, 0, 0, 0)?;
                let span = datetime - epoch;
                let length = span.get_unit_length(metadata.unit);
                Ok(Scalar::primitive(length, nullability))
            }
        }
    }

    fn pack_null(&self, _metadata: &Self::Metadata) -> VortexResult<Scalar> {
        Ok(Scalar::null(DType::Primitive(
            PType::I64,
            Nullability::Nullable,
        )))
    }
}
