// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::DType;
use crate::ExtDType;
use crate::Nullability;
use crate::PType;
use crate::datetime::TimeUnit;
use crate::extension::ExtDTypeVTable;
use crate::extension::ExtID;

/// Date DType.
#[derive(Clone, Debug, Default)]
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
    type Options = TimeUnit;

    fn id(&self) -> ExtID {
        ExtID::new_ref("vortex.date")
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Vec<u8>> {
        Ok(vec![u8::from(*options)])
    }

    fn deserialize(&self, data: &[u8]) -> VortexResult<Self::Options> {
        let tag = data[0];
        TimeUnit::try_from(tag)
    }

    fn validate(&self, options: &Self::Options, storage_dtype: &DType) -> VortexResult<()> {
        let ptype = date_ptype(options)
            .ok_or_else(|| vortex_err!("Date type does not support time unit {}", options))?;

        vortex_ensure!(
            storage_dtype.as_ptype() == ptype,
            "Date storage dtype for {} must be {}",
            options,
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
