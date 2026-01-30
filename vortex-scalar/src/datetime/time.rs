// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Sub;

use jiff::Span;
use vortex_dtype::DType;
use vortex_dtype::ExtDType;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_dtype::datetime::Time;
use vortex_dtype::datetime::TimeUnit;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::Scalar;
use crate::ScalarValue;
use crate::datetime::SpanExt;
use crate::extension::ExtScalarVTable;

impl ExtScalarVTable for Time {
    type Value = jiff::civil::Time;

    fn zero(&self, _metadata: &Self::Metadata) -> Self::Value {
        jiff::civil::Time::MIN
    }

    fn unpack(&self, dtype: &ExtDType<Self>, storage: &ScalarValue) -> VortexResult<Self::Value> {
        let v = storage
            .as_pvalue()?
            .vortex_expect("storage is non-null")
            .cast::<i64>();
        let span = Span::from_unit_length(v, *dtype.metadata());
        let epoch = jiff::civil::Time::MIN;
        Ok(epoch.checked_add(span)?)
    }

    fn pack(
        &self,
        metadata: &Self::Metadata,
        value: Option<&Self::Value>,
        nullability: Nullability,
    ) -> VortexResult<Scalar> {
        let Some(value) = value else {
            let ptype = match metadata {
                TimeUnit::Nanoseconds | TimeUnit::Microseconds => PType::I64,
                TimeUnit::Milliseconds | TimeUnit::Seconds => PType::I32,
                TimeUnit::Days => unreachable!("TimeUnit::Days is not supported for Time types"),
            };
            return Ok(Scalar::null(DType::Primitive(ptype, Nullability::Nullable)));
        };

        let epoch = jiff::civil::Time::MIN;
        let span = value.sub(epoch);
        let length = span.get_unit_length(*metadata);

        Ok(match metadata {
            TimeUnit::Nanoseconds | TimeUnit::Microseconds => {
                Scalar::primitive(length, nullability)
            }
            TimeUnit::Milliseconds | TimeUnit::Seconds => {
                let length =
                    i32::try_from(length).map_err(|_| vortex_err!("time does not fit in i32"))?;
                Scalar::primitive(length, nullability)
            }
            TimeUnit::Days => unreachable!("TimeUnit::Days is not supported for Time types"),
        })
    }
}
