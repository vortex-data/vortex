// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::ops::Sub;

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

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum TimestampValue {
    Zoned(jiff::Zoned),
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

    fn zero(&self, metadata: &Self::Metadata) -> Self::Value {
        match &metadata.tz {
            None => TimestampValue::Unzoned(jiff::Timestamp::UNIX_EPOCH),
            Some(tz) => TimestampValue::Zoned(
                jiff::Timestamp::UNIX_EPOCH
                    .in_tz(tz.as_ref())
                    .vortex_expect("failed to create zoned epoch"),
            ),
        }
    }

    fn unpack(&self, dtype: &ExtDType<Self>, storage: &ScalarValue) -> VortexResult<Self::Value> {
        let v = storage
            .as_pvalue()?
            .vortex_expect("storage is non-null")
            .cast::<i64>();

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

    fn pack(
        &self,
        metadata: &Self::Metadata,
        value: Option<&Self::Value>,
        nullability: Nullability,
    ) -> VortexResult<Scalar> {
        let Some(value) = value else {
            return Ok(Scalar::null(DType::Primitive(
                PType::I64,
                Nullability::Nullable,
            )));
        };

        match value {
            TimestampValue::Zoned(zoned) => {
                let span = zoned.timestamp() - jiff::Timestamp::UNIX_EPOCH;
                let length = span.get_unit_length(metadata.unit);
                Ok(Scalar::primitive(length, nullability))
            }
            TimestampValue::Unzoned(datetime) => {
                let span = datetime.sub(jiff::Timestamp::UNIX_EPOCH);
                let length = span.get_unit_length(metadata.unit);
                Ok(Scalar::primitive(length, nullability))
            }
        }
    }
}
