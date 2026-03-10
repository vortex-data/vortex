// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;

use jiff::Span;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::DynArray;
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

fn time_ptype(time_unit: &TimeUnit) -> Option<PType> {
    Some(match time_unit {
        TimeUnit::Nanoseconds | TimeUnit::Microseconds => PType::I64,
        TimeUnit::Milliseconds | TimeUnit::Seconds => PType::I32,
        TimeUnit::Days => return None,
    })
}

impl Time {
    /// Creates a new Time extension dtype with the given time unit and nullability.
    ///
    /// Note that Days units are not supported for Time.
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

    type NativeValue<'a> = TimeValue;

    fn id(&self) -> ExtId {
        ExtId::new_ref("vortex.time")
    }

    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(vec![u8::from(*metadata)])
    }

    fn deserialize_metadata(&self, data: &[u8]) -> VortexResult<Self::Metadata> {
        let tag = data[0];
        TimeUnit::try_from(tag)
    }

    fn validate_dtype(&self, ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        let metadata = ext_dtype.metadata();
        let ptype = time_ptype(metadata)
            .ok_or_else(|| vortex_err!("Time type does not support time unit {}", metadata))?;

        vortex_ensure!(
            ext_dtype.storage_dtype().as_ptype() == ptype,
            "Time storage dtype for {} must be {}",
            metadata,
            ptype
        );

        Ok(())
    }

    fn unpack_native(
        &self,
        ext_dtype: &ExtDType<Self>,
        storage_value: &ScalarValue,
    ) -> VortexResult<Self::NativeValue<'_>> {
        let length_of_time = storage_value.as_primitive().cast::<i64>()?;

        let (span, value) = match *ext_dtype.metadata() {
            TimeUnit::Seconds => {
                let v = i32::try_from(length_of_time)
                    .map_err(|e| vortex_err!("Time seconds value out of i32 range: {e}"))?;
                (Span::new().seconds(v), TimeValue::Seconds(v))
            }
            TimeUnit::Milliseconds => {
                let v = i32::try_from(length_of_time)
                    .map_err(|e| vortex_err!("Time milliseconds value out of i32 range: {e}"))?;
                (Span::new().milliseconds(v), TimeValue::Milliseconds(v))
            }
            TimeUnit::Microseconds => (
                Span::new().microseconds(length_of_time),
                TimeValue::Microseconds(length_of_time),
            ),
            TimeUnit::Nanoseconds => (
                Span::new().nanoseconds(length_of_time),
                TimeValue::Nanoseconds(length_of_time),
            ),
            d @ TimeUnit::Days => vortex_bail!("Time type does not support time unit {d}"),
        };

        // Validate the storage value is within the valid range for Time.
        jiff::civil::Time::MIN
            .checked_add(span)
            .map_err(|e| vortex_err!("Invalid time scalar: {}", e))?;

        Ok(value)
    }

    fn validate_array<'a>(
        &self,
        ext_dtype: &'a ExtDType<Self>,
        storage_array: &'a dyn DynArray,
    ) -> VortexResult<()> {
        // We check both min and max because the stored integer can be negative, which is invalid
        // for a time of day.
        let stats = storage_array.statistics();

        let metadata = ext_dtype.metadata();
        match metadata {
            TimeUnit::Seconds | TimeUnit::Milliseconds => {
                let build_span = |v: i32| match metadata {
                    TimeUnit::Seconds => Span::new().seconds(v),
                    TimeUnit::Milliseconds => Span::new().milliseconds(v),
                    _ => unreachable!(),
                };

                if let Some(min) = stats.compute_min::<i32>() {
                    jiff::civil::Time::MIN
                        .checked_add(build_span(min))
                        .map_err(|e| {
                            vortex_err!("Time array min value {min} is out of range: {e}")
                        })?;
                }
                if let Some(max) = stats.compute_max::<i32>() {
                    jiff::civil::Time::MIN
                        .checked_add(build_span(max))
                        .map_err(|e| {
                            vortex_err!("Time array max value {max} is out of range: {e}")
                        })?;
                }
            }
            TimeUnit::Microseconds | TimeUnit::Nanoseconds => {
                let build_span = |v: i64| match metadata {
                    TimeUnit::Microseconds => Span::new().microseconds(v),
                    TimeUnit::Nanoseconds => Span::new().nanoseconds(v),
                    _ => unreachable!(),
                };

                if let Some(min) = stats.compute_min::<i64>() {
                    jiff::civil::Time::MIN
                        .checked_add(build_span(min))
                        .map_err(|e| {
                            vortex_err!("Time array min value {min} is out of range: {e}")
                        })?;
                }
                if let Some(max) = stats.compute_max::<i64>() {
                    jiff::civil::Time::MIN
                        .checked_add(build_span(max))
                        .map_err(|e| {
                            vortex_err!("Time array max value {max} is out of range: {e}")
                        })?;
                }
            }
            d @ TimeUnit::Days => vortex_bail!("Time type does not support time unit {d}"),
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use crate::array::IntoArray;
    use crate::arrays::ExtensionArray;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability::Nullable;
    use crate::extension::datetime::Time;
    use crate::extension::datetime::TimeUnit;
    use crate::scalar::PValue;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;

    #[test]
    fn validate_time_scalar() -> VortexResult<()> {
        // 3661 seconds = 1 hour, 1 minute, 1 second.
        let dtype = DType::Extension(Time::new(TimeUnit::Seconds, Nullable).erased());
        Scalar::try_new(dtype, Some(ScalarValue::Primitive(PValue::I32(3661))))?;

        Ok(())
    }

    #[test]
    fn reject_time_out_of_range() {
        // 86400 seconds = exactly 24 hours, which exceeds the valid `jiff::civil::Time` range.
        let dtype = DType::Extension(Time::new(TimeUnit::Seconds, Nullable).erased());
        let result = Scalar::try_new(dtype, Some(ScalarValue::Primitive(PValue::I32(86400))));
        assert!(result.is_err());
    }

    #[test]
    fn display_time_scalar() {
        let dtype = DType::Extension(Time::new(TimeUnit::Seconds, Nullable).erased());

        let scalar = Scalar::new(
            dtype.clone(),
            Some(ScalarValue::Primitive(PValue::I32(3661))),
        );
        assert_eq!(format!("{}", scalar.as_extension()), "01:01:01");

        let scalar = Scalar::new(dtype, Some(ScalarValue::Primitive(PValue::I32(0))));
        assert_eq!(format!("{}", scalar.as_extension()), "00:00:00");
    }

    #[test]
    fn validate_time_array() -> VortexResult<()> {
        let ext_dtype = Time::new(TimeUnit::Seconds, Nullable).erased();
        let storage = PrimitiveArray::from_option_iter([Some(0i32), Some(3661), Some(86399)]);
        ExtensionArray::try_new(ext_dtype, storage.into_array())?;
        Ok(())
    }

    #[test]
    fn reject_time_array_out_of_range() {
        // 86400 seconds = exactly 24 hours, which exceeds the valid time-of-day range.
        let ext_dtype = Time::new(TimeUnit::Seconds, Nullable).erased();
        let storage = PrimitiveArray::from_option_iter([Some(0i32), Some(86400)]);
        assert!(ExtensionArray::try_new(ext_dtype, storage.into_array()).is_err());
    }

    #[test]
    fn reject_time_array_negative() {
        let ext_dtype = Time::new(TimeUnit::Seconds, Nullable).erased();
        let storage = PrimitiveArray::from_option_iter([Some(-1i32)]);
        assert!(ExtensionArray::try_new(ext_dtype, storage.into_array()).is_err());
    }
}
