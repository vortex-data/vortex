// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::DType;
use crate::ExtDType;
use crate::Nullability;
use crate::PType;
use crate::datetime::TimeUnit;
use crate::extension::ExtID;
use crate::extension::VTable;

/// Date DType.
pub struct Date;

impl Date {
    pub const ID: ExtID = ExtID::new_ref("vortex.date");

    /// Creates a new Date extension dtype with the given time unit and nullability.
    ///
    /// Note that only Milliseconds and Days time units are supported for Date.
    pub fn try_new(time_unit: TimeUnit, nullability: Nullability) -> VortexResult<ExtDType<Self>> {
        let ptype = date_ptype(&time_unit)
            .ok_or_else(|| vortex_err!("Date type does not support time unit {}", time_unit))?;
        ExtDType::try_new(time_unit, DType::Primitive(ptype, nullability))
    }

    pub fn new(time_unit: TimeUnit, nullability: Nullability) -> ExtDType<Self> {
        Self::try_new(time_unit, nullability).vortex_expect("failed to create date dtype")
    }
}

impl VTable for Date {
    type Options = TimeUnit;

    fn id(_options: &Self::Options) -> ExtId {
        Self::ID
    }

    fn serialize(options: &Self::Options) -> VortexResult<Vec<u8>> {
        Ok(vec![u8::from(*options)])
    }

    fn deserialize(data: &[u8], _session: &VortexSession) -> VortexResult<Self::Options> {
        let tag = data[0];
        Ok(TimeUnit::try_from(tag)?)
    }

    fn validate(options: &Self::Options, storage_dtype: &DType) -> VortexResult<()> {
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
