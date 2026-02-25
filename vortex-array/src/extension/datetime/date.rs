// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;

use jiff::Span;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::extension::datetime::TimeUnit;
use crate::scalar::ScalarValue;

/// The Unix epoch date (1970-01-01).
const EPOCH: jiff::civil::Date = jiff::civil::Date::constant(1970, 1, 1);

/// Date DType.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Date;

impl Date {
    /// Creates a new Date extension dtype with the given time unit and nullability.
    ///
    /// Note that only Milliseconds and Days time units are supported for Date.
    pub fn try_new(time_unit: TimeUnit, nullability: Nullability) -> VortexResult<ExtDType<Self>> {
        let ptype = date_ptype(&time_unit)
            .ok_or_else(|| vortex_err!("Date type does not support time unit {}", time_unit))?;
        ExtDType::try_new(time_unit, DType::Primitive(ptype, nullability))
    }

    /// Creates a new Date extension dtype with the given time unit and nullability.
    ///
    /// # Panics
    ///
    /// Panics if the `time_unit` is not supported by date types.
    pub fn new(time_unit: TimeUnit, nullability: Nullability) -> ExtDType<Self> {
        Self::try_new(time_unit, nullability).vortex_expect("failed to create date dtype")
    }
}

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

impl ExtVTable for Date {
    type Metadata = TimeUnit;
    type Value<'a> = DateValue;

    fn id(&self) -> ExtId {
        ExtId::new_ref("vortex.date")
    }

    fn serialize(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(vec![u8::from(*metadata)])
    }

    fn deserialize(&self, metadata: &[u8]) -> VortexResult<Self::Metadata> {
        let tag = metadata[0];
        TimeUnit::try_from(tag)
    }

    fn validate_dtype(&self, metadata: &Self::Metadata, storage_dtype: &DType) -> VortexResult<()> {
        let ptype = date_ptype(metadata)
            .ok_or_else(|| vortex_err!("Date type does not support time unit {}", metadata))?;

        vortex_ensure!(
            storage_dtype.as_ptype() == ptype,
            "Date storage dtype for {} must be {}",
            metadata,
            ptype
        );

        Ok(())
    }

    fn validate_scalar_value(
        &self,
        _metadata: &Self::Metadata,
        _storage_dtype: &DType,
        _storage_value: &ScalarValue,
    ) -> VortexResult<()> {
        // We know that the dtype is correct for this extension type (primitive) by the
        // precondition that `validate_dtype` has already been called successfully, and we know that
        // the `Scalar` we came from has verified that the storage value is a primitive.
        // We also say that any i32 or i64 is a valid date value, so we do not need to verify the
        // values at all.
        Ok(())
    }

    fn unpack(
        &self,
        metadata: &Self::Metadata,
        _storage_dtype: &DType,
        storage_value: &ScalarValue,
    ) -> Self::Value<'_> {
        match metadata {
            TimeUnit::Milliseconds => {
                DateValue::Milliseconds(storage_value.as_primitive().cast::<i64>().vortex_expect(
                    "The Scalar validation already checked that the value must be an i64",
                ))
            }
            TimeUnit::Days => {
                DateValue::Days(storage_value.as_primitive().cast::<i32>().vortex_expect(
                    "The Scalar validation already checked that the value must be an i32",
                ))
            }
            _ => unreachable!(),
        }
    }
}

fn date_ptype(time_unit: &TimeUnit) -> Option<PType> {
    match time_unit {
        TimeUnit::Nanoseconds => None,
        TimeUnit::Microseconds => None,
        TimeUnit::Milliseconds => Some(PType::I64),
        TimeUnit::Seconds => None,
        TimeUnit::Days => Some(PType::I32),
    }
}
