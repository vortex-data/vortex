// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;

use jiff::Span;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
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

fn date_ptype(time_unit: &TimeUnit) -> Option<PType> {
    match time_unit {
        TimeUnit::Nanoseconds => None,
        TimeUnit::Microseconds => None,
        TimeUnit::Milliseconds => Some(PType::I64),
        TimeUnit::Seconds => None,
        TimeUnit::Days => Some(PType::I32),
    }
}

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
    type NativeValue<'a> = DateValue;

    fn id(&self) -> ExtId {
        ExtId::new("vortex.date")
    }

    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(vec![u8::from(*metadata)])
    }

    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::Metadata> {
        let tag = metadata[0];
        TimeUnit::try_from(tag)
    }

    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        let metadata = ext_dtype.metadata();
        let ptype = date_ptype(metadata)
            .ok_or_else(|| vortex_err!("Date type does not support time unit {}", metadata))?;

        vortex_ensure!(
            ext_dtype.storage_dtype().as_ptype() == ptype,
            "Date storage dtype for {} must be {}",
            metadata,
            ptype
        );

        Ok(())
    }

    fn can_coerce_from(ext_dtype: &ExtDType<Self>, other: &DType) -> bool {
        let DType::Extension(other_ext) = other else {
            return false;
        };
        let Some(other_unit) = other_ext.metadata_opt::<Date>() else {
            return false;
        };
        let our_unit = ext_dtype.metadata();
        // We can coerce from other if our unit is finer (<=) and nullability is compatible.
        our_unit <= other_unit && (ext_dtype.storage_dtype().is_nullable() || !other.is_nullable())
    }

    fn least_supertype(ext_dtype: &ExtDType<Self>, other: &DType) -> Option<DType> {
        let DType::Extension(other_ext) = other else {
            return None;
        };
        let other_unit = other_ext.metadata_opt::<Date>()?;
        let our_unit = ext_dtype.metadata();
        let finest = (*our_unit).min(*other_unit);
        let union_null = ext_dtype.storage_dtype().nullability() | other.nullability();
        Some(DType::Extension(Date::new(finest, union_null).erased()))
    }

    fn unpack_native<'a>(
        ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        let metadata = ext_dtype.metadata();
        match metadata {
            TimeUnit::Milliseconds => Ok(DateValue::Milliseconds(
                storage_value.as_primitive().cast::<i64>()?,
            )),
            TimeUnit::Days => Ok(DateValue::Days(storage_value.as_primitive().cast::<i32>()?)),
            _ => vortex_bail!("Date type does not support time unit {}", metadata),
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use crate::dtype::DType;
    use crate::dtype::Nullability::Nullable;
    use crate::extension::datetime::Date;
    use crate::extension::datetime::TimeUnit;
    use crate::scalar::PValue;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;

    #[test]
    fn validate_date_scalar() -> VortexResult<()> {
        let days_dtype = DType::Extension(Date::new(TimeUnit::Days, Nullable).erased());
        Scalar::try_new(days_dtype, Some(ScalarValue::Primitive(PValue::I32(0))))?;

        let ms_dtype = DType::Extension(Date::new(TimeUnit::Milliseconds, Nullable).erased());
        Scalar::try_new(
            ms_dtype,
            Some(ScalarValue::Primitive(PValue::I64(86_400_000))),
        )?;

        Ok(())
    }

    #[test]
    fn reject_date_with_overflowing_value() {
        // Days storage is `I32`, so an `I64` value that overflows `i32` should fail the cast.
        let dtype = DType::Extension(Date::new(TimeUnit::Days, Nullable).erased());
        let result = Scalar::try_new(dtype, Some(ScalarValue::Primitive(PValue::I64(i64::MAX))));
        assert!(result.is_err());
    }

    #[test]
    fn display_date_scalar() {
        let dtype = DType::Extension(Date::new(TimeUnit::Days, Nullable).erased());

        let scalar = Scalar::new(dtype.clone(), Some(ScalarValue::Primitive(PValue::I32(0))));
        assert_eq!(format!("{}", scalar.as_extension()), "1970-01-01");

        let scalar = Scalar::new(dtype, Some(ScalarValue::Primitive(PValue::I32(365))));
        assert_eq!(format!("{}", scalar.as_extension()), "1971-01-01");
    }

    #[test]
    fn least_supertype_date_units() {
        use crate::dtype::Nullability::NonNullable;

        let days = DType::Extension(Date::new(TimeUnit::Days, NonNullable).erased());
        let ms = DType::Extension(Date::new(TimeUnit::Milliseconds, NonNullable).erased());
        let expected = DType::Extension(Date::new(TimeUnit::Milliseconds, NonNullable).erased());
        assert_eq!(days.least_supertype(&ms).unwrap(), expected);
        assert_eq!(ms.least_supertype(&days).unwrap(), expected);
    }

    #[test]
    fn can_coerce_from_date() {
        use crate::dtype::Nullability::NonNullable;

        let days = DType::Extension(Date::new(TimeUnit::Days, NonNullable).erased());
        let ms = DType::Extension(Date::new(TimeUnit::Milliseconds, NonNullable).erased());
        assert!(ms.can_coerce_from(&days));
        assert!(!days.can_coerce_from(&ms));
    }
}
