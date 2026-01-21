// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Temporal extension data types.

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::DType;
use crate::Nullability;
use crate::PType;
use crate::VTable;
use crate::datetime::TimeUnit;
use crate::extension::vtable::ExtId;
use crate::v2::ExtDType;

pub struct Timestamp;

impl Timestamp {
    pub fn new(nullability: Nullability) -> ExtDType<Self> {
        ExtDType::try_new(
            TimestampOptions {
                time_unit: TimeUnit::Milliseconds,
                timezone: None,
            },
            DType::Primitive(PType::I64, nullability),
        )
        .vortex_expect("failed to create timestamp dtype")
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TimestampOptions {
    pub time_unit: TimeUnit,
    pub timezone: Option<String>,
}

impl Display for TimestampOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.timezone {
            Some(tz) => write!(f, "unit={}, tz={}", self.time_unit, tz),
            None => write!(f, "unit={}", self.time_unit),
        }
    }
}

impl VTable for Timestamp {
    type Options = TimestampOptions;

    fn id(_options: &Self::Options) -> ExtId {
        ExtId::new_ref("vortex.timestamp")
    }

    fn serialize(options: &Self::Options) -> VortexResult<Vec<u8>> {
        todo!()
    }

    fn deserialize(data: &[u8], session: &VortexSession) -> VortexResult<Self::Options> {
        todo!()
    }

    fn validate(_options: &Self::Options, storage_dtype: &DType) -> VortexResult<()> {
        vortex_ensure!(
            matches!(storage_dtype, DType::Primitive(PType::I64, _)),
            "Timestamp storage dtype must be i64"
        );
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use vortex_error::vortex_bail;
    use vortex_error::vortex_panic;

    use super::*;
    use crate::Nullability::NonNullable;

    #[test]
    fn test_stuff() {
        let dtype = Timestamp::new(NonNullable).erase();
        let Some(opts) = dtype.try_options::<Timestamp>() else {
            vortex_panic!("failed to match as Timestamp");
        };
    }
}
