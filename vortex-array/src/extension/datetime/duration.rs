// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use jiff::SignedDuration;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_session::registry::CachedId;

use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::extension::datetime::TimeUnit;
use crate::scalar::ScalarValue;

/// A signed count of elapsed seconds, milliseconds, microseconds, or nanoseconds.
///
/// Stored as an `i64` count, with the time unit in the dtype metadata.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Duration;

impl Duration {
    /// Creates a duration dtype. The unit must be seconds or finer.
    pub fn try_new(unit: TimeUnit, nullability: Nullability) -> VortexResult<ExtDType<Self>> {
        ExtDType::try_new(unit, nullability)
    }

    /// Creates a duration dtype.
    ///
    /// # Panics
    ///
    /// Panics if the unit is days.
    pub fn new(unit: TimeUnit, nullability: Nullability) -> ExtDType<Self> {
        Self::try_new(unit, nullability).vortex_expect("failed to create duration dtype")
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
        if *unit == TimeUnit::Days {
            vortex_bail!("Duration time unit must be seconds or finer");
        }
        Ok(())
    }

    fn unpack_native<'a>(
        ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        let value = storage_value.as_primitive().cast::<i64>()?;
        Ok(match ext_dtype.metadata() {
            TimeUnit::Seconds => SignedDuration::from_secs(value),
            TimeUnit::Milliseconds => SignedDuration::from_millis(value),
            TimeUnit::Microseconds => SignedDuration::from_micros(value),
            TimeUnit::Nanoseconds => SignedDuration::from_nanos(value),
            TimeUnit::Days => vortex_bail!("Duration time unit must be seconds or finer"),
        })
    }
}
