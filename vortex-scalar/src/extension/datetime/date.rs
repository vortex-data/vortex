// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ExtScalarVTable`] implementation for [`Date`] extension scalars.

use std::fmt;

use jiff::Span;
use vortex_dtype::DType;
use vortex_dtype::datetime::Date;
use vortex_dtype::datetime::TimeUnit;
use vortex_error::VortexResult;

use crate::ScalarValue;
use crate::extension::ExtScalarVTable;

/// The Unix epoch date (1970-01-01).
const EPOCH: jiff::civil::Date = jiff::civil::Date::constant(1970, 1, 1);

/// Unpacked value of a [`Date`] extension scalar.
pub enum DateValue {
    /// Days since the Unix epoch.
    Days(i32),
    /// Milliseconds since the Unix epoch.
    Milliseconds(i64),
}

impl fmt::Display for DateValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let date = match self {
            DateValue::Days(days) => EPOCH + Span::new().days(*days),
            DateValue::Milliseconds(ms) => EPOCH + Span::new().milliseconds(*ms),
        };
        write!(f, "{}", date)
    }
}

impl ExtScalarVTable for Date {
    type Value<'a> = DateValue;

    fn unpack(
        &self,
        metadata: &Self::Metadata,
        _storage_dtype: &DType,
        storage_value: &ScalarValue,
    ) -> Self::Value<'_> {
        match metadata {
            TimeUnit::Milliseconds => {
                DateValue::Milliseconds(storage_value.as_primitive().cast::<i64>())
            }
            TimeUnit::Days => DateValue::Days(storage_value.as_primitive().cast::<i32>()),
            _ => unreachable!(),
        }
    }

    fn validate_scalar_value(
        &self,
        _metadata: &<Self as vortex_dtype::extension::ExtDTypeVTable>::Metadata,
        _storage_dtype: &DType,
        _storage_value: &ScalarValue,
    ) -> VortexResult<()> {
        // We know that the dtype is correct for this extension type (primitive) by the
        // precondition, and we know that the `Scalar` we came from has verified that the storage
        // value is a primitive. We also say that any i32 or i64 is a valid date value.
        Ok(())
    }
}
