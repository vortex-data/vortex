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
use crate::extension::ExtID;
use crate::extension::VTable;

/// Time DType.
#[derive(Clone, Debug, Default)]
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

impl VTable for Time {
    type Options = TimeUnit;

    fn id(&self) -> ExtID {
        ExtID::new_ref("vortex.time")
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Vec<u8>> {
        Ok(vec![u8::from(*options)])
    }

    fn deserialize(&self, data: &[u8]) -> VortexResult<Self::Options> {
        let tag = data[0];
        Ok(TimeUnit::try_from(tag)?)
    }

    fn validate(&self, options: &Self::Options, storage_dtype: &DType) -> VortexResult<()> {
        let ptype = time_ptype(options)
            .ok_or_else(|| vortex_err!("Time type does not support time unit {}", options))?;

        vortex_ensure!(
            storage_dtype.as_ptype() == ptype,
            "Time storage dtype for {} must be {}",
            options,
            ptype
        );

        Ok(())
    }
}

fn time_ptype(time_unit: &TimeUnit) -> Option<PType> {
    Some(match time_unit {
        TimeUnit::Nanoseconds | TimeUnit::Microseconds => PType::I64,
        TimeUnit::Milliseconds | TimeUnit::Seconds => PType::I32,
        TimeUnit::Days => return None,
    })
}
