// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Sub;

use jiff::Span;
use vortex_dtype::ExtDType;
use vortex_dtype::datetime::Time;
use vortex_dtype::datetime::TimeUnit;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::PValue;
use crate::ScalarValue;
use crate::datetime::SpanExt;
use crate::extension::ExtScalarVTable;

impl ExtScalarVTable for Time {
    type Value = jiff::civil::Time;

    fn unpack(&self, dtype: &ExtDType<Self>, storage: &ScalarValue) -> VortexResult<Self::Value> {
        let ScalarValue::Primitive(pvalue) = storage else {
            vortex_bail!("expected primitive scalar value for Time dtype");
        };
        let v = pvalue.cast::<i64>();
        let span = Span::from_unit_length(v, *dtype.metadata());
        let epoch = jiff::civil::Time::MIN;
        Ok(epoch.checked_add(span)?)
    }

    fn pack(&self, dtype: &ExtDType<Self>, value: &Self::Value) -> VortexResult<ScalarValue> {
        let epoch = jiff::civil::Time::MIN;
        let span = value.sub(epoch);
        let length = span.get_unit_length(*dtype.metadata());

        let pvalue = match dtype.metadata() {
            TimeUnit::Nanoseconds | TimeUnit::Microseconds => PValue::I64(length),
            TimeUnit::Milliseconds | TimeUnit::Seconds => {
                let length =
                    i32::try_from(length).map_err(|_| vortex_err!("time does not fit in i32"))?;
                PValue::I32(length)
            }
            TimeUnit::Days => unreachable!("TimeUnit::Days is not supported for Time types"),
        };

        Ok(ScalarValue::Primitive(pvalue))
    }

    fn validate(&self, _value: &Self::Value, _ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        // All jiff::civil::Time values are valid, so no additional validation is needed.
        Ok(())
    }
}
