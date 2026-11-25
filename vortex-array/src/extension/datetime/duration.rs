// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use jiff::SignedDuration;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_session::registry::CachedId;

use crate::dtype::Claim;
use crate::dtype::Claims;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::extension::datetime::TimeUnit;
use crate::scalar::ScalarValue;

/// A signed elapsed-time value in seconds, milliseconds, microseconds, or nanoseconds.
///
/// Stored as an `i64` value, with the time unit in the dtype metadata.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Duration;

impl Duration {
    /// Creates a duration dtype with the given unit and nullability.
    ///
    /// # Errors
    ///
    /// Returns an error for [`TimeUnit::Days`].
    pub fn try_new(unit: TimeUnit, nullability: Nullability) -> VortexResult<ExtDType<Self>> {
        ExtDType::try_new(unit, nullability)
    }

    /// Creates a duration dtype with the given unit and nullability.
    ///
    /// # Panics
    ///
    /// Panics for [`TimeUnit::Days`].
    pub fn new(unit: TimeUnit, nullability: Nullability) -> ExtDType<Self> {
        Self::try_new(unit, nullability).vortex_expect("duration unit must be finer than days")
    }

    pub(super) fn to_jiff(unit: TimeUnit, value: i64) -> VortexResult<SignedDuration> {
        Ok(match unit {
            TimeUnit::Seconds => SignedDuration::from_secs(value),
            TimeUnit::Milliseconds => SignedDuration::from_millis(value),
            TimeUnit::Microseconds => SignedDuration::from_micros(value),
            TimeUnit::Nanoseconds => SignedDuration::from_nanos(value),
            TimeUnit::Days => vortex_bail!("Duration unit must be finer than days"),
        })
    }
}

impl ExtVTable for Duration {
    type Metadata = TimeUnit;
    type StoredMetadata = Self::Metadata;
    type NativeValue<'a> = SignedDuration;

    fn id(&self) -> ExtId {
        static ID: CachedId = CachedId::new("vortex.duration");
        *ID
    }

    fn serialize_metadata(&self, unit: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(vec![u8::from(*unit)])
    }

    fn deserialize_metadata(&self, data: &[u8]) -> VortexResult<Self::StoredMetadata> {
        vortex_ensure_eq!(
            data.len(),
            1,
            "Duration metadata must contain one time unit"
        );
        TimeUnit::try_from(data[0])
    }

    fn metadata_from_storage(
        &self,
        unit: Self::StoredMetadata,
        _storage_dtype: &DType,
    ) -> VortexResult<Self::Metadata> {
        Ok(unit)
    }

    fn storage_dtype(_unit: &Self::Metadata) -> DType {
        DType::Primitive(PType::I64, Nullability::NonNullable)
    }

    fn validate_metadata(unit: &Self::Metadata) -> VortexResult<()> {
        vortex_ensure!(
            *unit != TimeUnit::Days,
            "Duration unit must be finer than days"
        );
        Ok(())
    }

    fn claims(_unit: &Self::Metadata) -> Claims {
        Claims::none().with(Claim::Ord)
    }

    fn unpack_native<'a>(
        ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        let value = storage_value.as_primitive().cast::<i64>()?;
        Self::to_jiff(*ext_dtype.metadata(), value)
    }
}

#[cfg(test)]
mod tests {
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::extension::ExtDType;
    use crate::extension::datetime::Duration;
    use crate::extension::datetime::TimeUnit;

    #[test]
    fn test_duration_rejects_days_and_non_i64_storage() {
        assert!(Duration::try_new(TimeUnit::Days, Nullability::Nullable).is_err());
        assert!(
            ExtDType::<Duration>::try_from_storage(
                TimeUnit::Milliseconds,
                DType::Primitive(PType::I32, Nullability::Nullable),
            )
            .is_err()
        );
    }
}
