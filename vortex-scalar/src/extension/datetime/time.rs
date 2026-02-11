// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ExtScalarVTable`] implementation for [`Time`] extension scalars.

use std::fmt::Formatter;

use jiff::Span;
use vortex_dtype::DType;
use vortex_dtype::datetime::Time;
use vortex_dtype::datetime::TimeUnit;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ScalarValue;
use crate::extension::ExtScalarVTable;
use crate::extension::datetime::SpanExt;

/// Unpacked value of a [`Time`] extension scalar.
pub enum TimeValue {
    /// Seconds since midnight.
    Seconds(i32),
    /// Milliseconds since midnight.
    Milliseconds(i32),
    /// Microseconds since midnight.
    Microseconds(i64),
    /// Nanoseconds since midnight.
    Nanoseconds(i64),
}

impl std::fmt::Display for TimeValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let min = jiff::civil::Time::MIN;

        let time = match self {
            TimeValue::Seconds(s) => min + Span::new().seconds(*s),
            TimeValue::Milliseconds(ms) => min + Span::new().milliseconds(*ms),
            TimeValue::Microseconds(us) => min + Span::new().microseconds(*us),
            TimeValue::Nanoseconds(ns) => min + Span::new().nanoseconds(*ns),
        };

        write!(f, "{}", time)
    }
}

impl ExtScalarVTable for Time {
    type Value<'a> = TimeValue;

    fn unpack(
        &self,
        metadata: &Self::Metadata,
        _storage_dtype: &DType,
        storage_value: &ScalarValue,
    ) -> Self::Value<'_> {
        match metadata {
            TimeUnit::Seconds => TimeValue::Seconds(storage_value.as_primitive().cast::<i32>()),
            TimeUnit::Milliseconds => {
                TimeValue::Milliseconds(storage_value.as_primitive().cast::<i32>())
            }
            TimeUnit::Microseconds => {
                TimeValue::Microseconds(storage_value.as_primitive().cast::<i64>())
            }
            TimeUnit::Nanoseconds => {
                TimeValue::Nanoseconds(storage_value.as_primitive().cast::<i64>())
            }
            _ => unreachable!(),
        }
    }

    fn validate_scalar_value(
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
        // 86_399 seconds = 23:59:59, valid civil time
        assert!(
            Time.validate_scalar_value(
                &TimeUnit::Seconds,
                &DType::Primitive(PType::I32, Nullability::NonNullable),
                &ScalarValue::from(86_399i32),
            )
            .is_ok()
        );

        // 86_400 seconds = 24:00:00, invalid civil time (wraps to next day)
        assert!(
            Time.validate_scalar_value(
                &TimeUnit::Seconds,
                &DType::Primitive(PType::I32, Nullability::NonNullable),
                &ScalarValue::from(86_400i32),
            )
            .is_err()
        );
    }
}
