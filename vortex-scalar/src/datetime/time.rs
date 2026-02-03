// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use jiff::Span;
use vortex_dtype::DType;
use vortex_dtype::datetime::Time;
use vortex_dtype::datetime::TimeUnit;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ScalarValue;
use crate::datetime::SpanExt;
use crate::extension::ExtScalarVTable;

pub enum TimeValue {
    Seconds(Option<i32>),
    Milliseconds(Option<i32>),
    Microseconds(Option<i64>),
    Nanoseconds(Option<i64>),
}

impl ExtScalarVTable for Time {
    type Value<'a> = TimeValue;

    fn unpack(
        &self,
        metadata: &Self::Metadata,
        _storage_dtype: &DType,
        storage_value: Option<&ScalarValue>,
    ) -> Self::Value<'_> {
        match metadata {
            TimeUnit::Seconds => {
                TimeValue::Seconds(storage_value.map(|s| s.as_primitive().cast::<i32>()))
            }
            TimeUnit::Milliseconds => {
                TimeValue::Milliseconds(storage_value.map(|s| s.as_primitive().cast::<i32>()))
            }
            TimeUnit::Microseconds => {
                TimeValue::Microseconds(storage_value.map(|s| s.as_primitive().cast::<i64>()))
            }
            TimeUnit::Nanoseconds => {
                TimeValue::Nanoseconds(storage_value.map(|s| s.as_primitive().cast::<i64>()))
            }
            _ => unreachable!(),
        }
    }

    fn fmt_scalar(
        &self,
        metadata: &Self::Metadata,
        _storage_dtype: &DType,
        storage_value: &ScalarValue,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        let span = Span::from_unit_length(storage_value.as_primitive().cast::<i64>(), *metadata);
        write!(f, "{}", jiff::civil::Time::MIN + span)
    }

    fn validate_scalar(
        &self,
        metadata: &Self::Metadata,
        _storage_dtype: &DType,
        storage_value: &ScalarValue,
    ) -> VortexResult<()> {
        // Validate the storage value is within the valid range for Time
        let span = Span::from_unit_length(storage_value.as_primitive().cast::<i64>(), *metadata);
        jiff::civil::Time::MIN
            .checked_add(span)
            .map_err(|e| vortex_err!("Invalid time scalar: {}", e))?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;
    use vortex_dtype::datetime::TimeUnit;

    use super::*;

    #[test]
    fn test_validate() {
        assert!(
            Time.validate_scalar(
                &TimeUnit::Seconds,
                &DType::Primitive(PType::I32, Nullability::NonNullable),
                &ScalarValue::from(86_400i32),
            )
            .is_ok()
        );

        assert!(
            Time.validate_scalar(
                &TimeUnit::Seconds,
                &DType::Primitive(PType::I32, Nullability::NonNullable),
                &ScalarValue::from(86_400i32 + 1),
            )
            .is_err()
        );
    }
}
