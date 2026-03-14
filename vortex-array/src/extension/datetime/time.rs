// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;

use jiff::Span;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ExtensionArray;
use crate::arrays::extension::ExtArray;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::extension::datetime::TimeUnit;
use crate::matcher::Matcher;
use crate::scalar::ScalarValue;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::operators::Operator;

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

    fn can_coerce_from(&self, ext_dtype: &ExtDType<Self>, other: &DType) -> bool {
        let DType::Extension(other_ext) = other else {
            return false;
        };
        let Some(other_unit) = other_ext.metadata_opt::<Time>() else {
            return false;
        };
        let our_unit = ext_dtype.metadata();
        our_unit <= other_unit && (ext_dtype.storage_dtype().is_nullable() || !other.is_nullable())
    }

    fn least_supertype(&self, ext_dtype: &ExtDType<Self>, other: &DType) -> Option<DType> {
        let DType::Extension(other_ext) = other else {
            return None;
        };
        let other_unit = other_ext.metadata_opt::<Time>()?;
        let our_unit = ext_dtype.metadata();
        let finest = (*our_unit).min(*other_unit);
        let union_null = ext_dtype.storage_dtype().nullability() | other.nullability();
        Some(DType::Extension(Time::new(finest, union_null).erased()))
    }

    fn reduce_parent_array(
        &self,
        array: &ExtArray<Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let _ = child_idx;
        let Some(cast_view) = ExactScalarFn::<Cast>::try_match(parent.as_ref()) else {
            return Ok(None);
        };
        let target = cast_view.options;

        let DType::Extension(target_ext) = target else {
            return Ok(None);
        };
        let Some(target_unit) = target_ext.metadata_opt::<Time>() else {
            return Ok(None);
        };
        let source_unit = array.ext_dtype().metadata();

        let source_nanos = source_unit.nanos_per_unit();
        let target_nanos = target_unit.nanos_per_unit();

        // Cast storage to target ptype first (e.g. i32 → i64 for Seconds → Nanoseconds).
        let storage = array
            .storage_array()
            .cast(target_ext.storage_dtype().clone())?;

        let storage = if source_nanos == target_nanos {
            storage
        } else if source_nanos > target_nanos {
            let factor = source_nanos / target_nanos;
            storage.binary(
                ConstantArray::new(factor, storage.len()).into_array(),
                Operator::Mul,
            )?
        } else {
            let factor = target_nanos / source_nanos;
            storage.binary(
                ConstantArray::new(factor, storage.len()).into_array(),
                Operator::Div,
            )?
        };

        Ok(Some(
            ExtensionArray::new(target_ext.clone(), storage).into_array(),
        ))
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
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

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
    fn least_supertype_time_units() {
        use crate::dtype::Nullability::NonNullable;

        let secs = DType::Extension(Time::new(TimeUnit::Seconds, NonNullable).erased());
        let ns = DType::Extension(Time::new(TimeUnit::Nanoseconds, NonNullable).erased());
        let expected = DType::Extension(Time::new(TimeUnit::Nanoseconds, NonNullable).erased());
        assert_eq!(secs.least_supertype(&ns).unwrap(), expected);
        assert_eq!(ns.least_supertype(&secs).unwrap(), expected);
    }

    #[test]
    fn cast_time_seconds_to_nanoseconds() {
        use vortex_buffer::buffer;

        use crate::IntoArray;
        use crate::arrays::ExtensionArray;
        use crate::builtins::ArrayBuiltins;
        use crate::dtype::Nullability::NonNullable;

        let source_dtype = Time::new(TimeUnit::Seconds, NonNullable).erased();
        let target_dtype = Time::new(TimeUnit::Nanoseconds, NonNullable).erased();

        let storage = buffer![1i32, 2].into_array();
        let arr = ExtensionArray::new(source_dtype, storage).into_array();

        let result = arr.cast(DType::Extension(target_dtype)).unwrap();
        let ext = result.to_canonical().unwrap().as_extension().clone();
        let prim = ext
            .storage_array()
            .to_canonical()
            .unwrap()
            .as_primitive()
            .clone();
        // Seconds → Nanoseconds: ×1_000_000_000
        assert_eq!(prim.as_slice::<i64>(), &[1_000_000_000i64, 2_000_000_000]);
    }
}
