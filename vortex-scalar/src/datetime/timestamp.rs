// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use jiff::Span;
use vortex_dtype::DType;
use vortex_dtype::datetime::Timestamp;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ScalarValue;
use crate::datetime::SpanExt;
use crate::extension::ExtScalarVTable;

impl ExtScalarVTable for Timestamp {
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
        match metadata.tz {
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
            .map_err(|e| vortex_err!("Invalid timestamp scalar: {}", span))?;

        if let Some(tz) = &metadata.tz {
            ts.in_tz(tz.as_ref())
                .map_err(|e| vortex_err!("Invalid timezone for timestamp scalar: {}", e))?;
        }

        Ok(())
    }
}
