// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Sub;

use jiff::Span;
use vortex_dtype::ExtDType;
use vortex_dtype::datetime::Date;
use vortex_dtype::datetime::TimeUnit;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::PValue;
use crate::ScalarValue;
use crate::datetime::SpanExt;
use crate::extension::ExtScalarVTable;

impl ExtScalarVTable for Date {
    type Value = jiff::civil::Date;

    fn unpack(&self, dtype: &ExtDType<Self>, storage: &ScalarValue) -> VortexResult<Self::Value> {
        let ScalarValue::Primitive(pvalue) = storage else {
            vortex_bail!("expected primitive scalar value for Date dtype");
        };
        let v = pvalue.cast::<i64>();
        let span = Span::from_unit_length(v, *dtype.metadata());
        let epoch = jiff::civil::Date::new(1970, 1, 1)?;
        Ok(epoch.checked_add(span)?)
    }

    fn pack(&self, dtype: &ExtDType<Self>, value: &Self::Value) -> VortexResult<ScalarValue> {
        let epoch = jiff::civil::Date::new(1970, 1, 1)?;
        let span = value.sub(epoch);
        let length = span.get_unit_length(*dtype.metadata());

        let pvalue = match dtype.metadata() {
            TimeUnit::Milliseconds => PValue::I64(length),
            TimeUnit::Days => {
                let length =
                    i32::try_from(length).map_err(|_| vortex_err!("date does not fit in i32"))?;
                PValue::I32(length)
            }
            _ => unreachable!("Date only supports Milliseconds and Days time units"),
        };

        Ok(ScalarValue::Primitive(pvalue))
    }

    fn validate(&self, _value: &Self::Value, _ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        // All jiff::civil::Date values are valid, so no additional validation is needed.
        Ok(())
    }
}
