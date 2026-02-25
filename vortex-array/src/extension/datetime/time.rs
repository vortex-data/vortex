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

/// Time DType.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Time;

impl Time {
    /// Creates a new Time extension dtype with the given time unit and nullability.
    ///
    /// Note that only Milliseconds and Days time units are supported for Time.
    pub fn try_new(time_unit: TimeUnit, nullability: Nullability) -> VortexResult<ExtDType<Self>> {
        let ptype = time_ptype(&time_unit)
            .ok_or_else(|| vortex_err!("Time type does not support time unit {}", time_unit))?;
        ExtDType::try_new(time_unit, DType::Primitive(ptype, nullability))
    }

    /// Creates a new Time extension dtype with the given time unit and nullability.
    pub fn new(time_unit: TimeUnit, nullability: Nullability) -> ExtDType<Self> {
        Self::try_new(time_unit, nullability).vortex_expect("failed to create time dtype")
    }
}

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

impl fmt::Display for TimeValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

impl ExtVTable for Time {
    type Metadata = TimeUnit;

    type Value<'a> = TimeValue;

    fn id(&self) -> ExtId {
        ExtId::new_ref("vortex.time")
    }

    fn serialize(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(vec![u8::from(*metadata)])
    }

    fn deserialize(&self, data: &[u8]) -> VortexResult<Self::Metadata> {
        let tag = data[0];
        TimeUnit::try_from(tag)
    }

    fn validate_dtype(&self, metadata: &Self::Metadata, storage_dtype: &DType) -> VortexResult<()> {
        let ptype = time_ptype(metadata)
            .ok_or_else(|| vortex_err!("Time type does not support time unit {}", metadata))?;

        vortex_ensure!(
            storage_dtype.as_ptype() == ptype,
            "Time storage dtype for {} must be {}",
            metadata,
            ptype
        );

        Ok(())
    }

    fn validate_scalar_value(
        &self,
        metadata: &Self::Metadata,
        _storage_dtype: &DType,
        storage_value: &ScalarValue,
    ) -> VortexResult<()> {
        let length_of_time = storage_value.as_primitive().cast::<i64>()?;

        // Validate the storage value is within the valid range for Time.
        let span = match *metadata {
            TimeUnit::Nanoseconds => Span::new().nanoseconds(length_of_time),
            TimeUnit::Microseconds => Span::new().microseconds(length_of_time),
            TimeUnit::Milliseconds => Span::new().milliseconds(length_of_time),
            TimeUnit::Seconds => Span::new().seconds(length_of_time),
            TimeUnit::Days => Span::new().days(length_of_time),
        };

        jiff::civil::Time::MIN
            .checked_add(span)
            .map_err(|e| vortex_err!("Invalid time scalar: {}", e))?;

        Ok(())
    }

    fn unpack(
        &self,
        metadata: &Self::Metadata,
        _storage_dtype: &DType,
        storage_value: &ScalarValue,
    ) -> Self::Value<'_> {
        match metadata {
            TimeUnit::Seconds => {
                TimeValue::Seconds(storage_value.as_primitive().cast::<i32>().vortex_expect(
                    "The Scalar validation already checked that the value must be an i32",
                ))
            }
            TimeUnit::Milliseconds => {
                TimeValue::Milliseconds(storage_value.as_primitive().cast::<i32>().vortex_expect(
                    "The Scalar validation already checked that the value must be an i32",
                ))
            }
            TimeUnit::Microseconds => {
                TimeValue::Microseconds(storage_value.as_primitive().cast::<i64>().vortex_expect(
                    "The Scalar validation already checked that the value must be an i64",
                ))
            }
            TimeUnit::Nanoseconds => {
                TimeValue::Nanoseconds(storage_value.as_primitive().cast::<i64>().vortex_expect(
                    "The Scalar validation already checked that the value must be an i64",
                ))
            }
            _ => unreachable!(),
        }
    }
}

fn time_ptype(time_unit: &TimeUnit) -> Option<PType> {
    Some(match time_unit {
        TimeUnit::Nanoseconds | TimeUnit::Microseconds => PType::I64,
        TimeUnit::Milliseconds | TimeUnit::Seconds => PType::I32,
        TimeUnit::Days => return None,
    })
}
