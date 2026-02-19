// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::dtype::DType;
use crate::dtype::ExtDType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::datetime::TimeUnit;
use crate::dtype::extension::ExtDTypeVTable;
use crate::dtype::extension::ExtID;

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

impl ExtDTypeVTable for Date {
    type Metadata = TimeUnit;

    fn id(&self) -> ExtID {
        ExtID::new_ref("vortex.date")
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
