// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Formatter;

use jiff::Span;
use vortex_dtype::DType;
use vortex_dtype::datetime::Date;
use vortex_error::VortexResult;

use crate::ScalarValue;
use crate::datetime::SpanExt;
use crate::extension::ExtScalarVTable;

const EPOCH: jiff::civil::Date = jiff::civil::Date::constant(1970, 1, 1);

impl ExtScalarVTable for Date {
    fn fmt_scalar(
        &self,
        metadata: &Self::Metadata,
        _storage_dtype: &DType,
        storage_value: &ScalarValue,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        let span = Span::from_unit_length(storage_value.as_primitive().cast::<i64>(), *metadata);
        write!(f, "{}", EPOCH + span)
    }

    fn validate_scalar(
        &self,
        _metadata: &Self::Metadata,
        _storage_dtype: &DType,
        _storage_value: &ScalarValue,
    ) -> VortexResult<()> {
        // Date DType has already validated the storage dtype
        Ok(())
    }
}
