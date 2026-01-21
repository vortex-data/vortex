// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::DType;
use crate::ExtId;
use crate::Nullability;
use crate::PType;
use crate::VTable;
use crate::datetime::TimeUnit;
use crate::v2::ExtDType;

pub struct Date;

impl Date {
    pub fn new(time_unit: TimeUnit, nullability: Nullability) -> ExtDType<Self> {
        let ptype = date_ptype(&time_unit)
            .ok_or_else(|| vortex_err!("Date type does not support time unit {}", time_unit))
            .vortex_expect("failed to create date dtype");
        ExtDType::try_new(time_unit, DType::Primitive(ptype, nullability))
            .vortex_expect("failed to create date dtype")
    }
}

impl VTable for Date {
    type Options = TimeUnit;

    fn id(_options: &Self::Options) -> ExtId {
        ExtId::from("vortex.date")
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
