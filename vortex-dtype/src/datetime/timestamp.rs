// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Temporal extension data types.

use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

use crate::DType;
use crate::ExtDType;
use crate::Nullability;
use crate::PType;
use crate::datetime::TimeUnit;
use crate::extension::ExtDTypeVTable;
use crate::extension::ExtID;

/// Timestamp DType.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Timestamp;

impl Timestamp {
    /// Creates a new Timestamp extension =dtype with the given time unit and nullability.
    pub fn new(time_unit: TimeUnit, nullability: Nullability) -> ExtDType<Self> {
        Self::new_with_tz(time_unit, None, nullability)
    }

    /// Creates a new Timestamp extension dtype with the given time unit, timezone, and nullability.
    pub fn new_with_tz(
        time_unit: TimeUnit,
        timezone: Option<Arc<str>>,
        nullability: Nullability,
    ) -> ExtDType<Self> {
        ExtDType::try_new(
            TimestampOptions {
                unit: time_unit,
                tz: timezone,
            },
            DType::Primitive(PType::I64, nullability),
        )
        .vortex_expect("failed to create timestamp dtype")
    }
}

/// Options for the Timestamp DType.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TimestampOptions {
    /// The time unit of the timestamp.
    pub unit: TimeUnit,
    /// The timezone of the timestamp, if any.
    pub tz: Option<Arc<str>>,
}

impl Display for TimestampOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.tz {
            Some(tz) => write!(f, "{}, tz={}", self.unit, tz),
            None => write!(f, "{}", self.unit),
        }
    }
}

impl ExtDTypeVTable for Timestamp {
    type Metadata = TimestampOptions;

    fn id(&self) -> ExtID {
        ExtID::new_ref("vortex.timestamp")
    }

    // NOTE(ngates): unfortunately we're stuck with this hand-rolled serialization format for
    //  backwards compatibility.
    fn serialize(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        let mut bytes = Vec::with_capacity(4);
        let unit_tag: u8 = metadata.unit.into();

        bytes.push(unit_tag);

        // Encode time_zone as u16 length followed by utf8 bytes.
        match &metadata.tz {
            None => bytes.extend_from_slice(0u16.to_le_bytes().as_slice()),
            Some(tz) => {
                let tz_bytes = tz.as_bytes();
                let tz_len = u16::try_from(tz_bytes.len())
                    .unwrap_or_else(|err| vortex_panic!("tz did not fit in u16: {}", err));
                bytes.extend_from_slice(tz_len.to_le_bytes().as_slice());
                bytes.extend_from_slice(tz_bytes);
            }
        }

        Ok(bytes)
    }

    fn deserialize(&self, data: &[u8]) -> VortexResult<Self::Metadata> {
        let tag = data[0];
        let time_unit = TimeUnit::try_from(tag)?;
        let tz_len_bytes = &data[1..3];
        let tz_len = u16::from_le_bytes(tz_len_bytes.try_into()?) as usize;
        if tz_len == 0 {
            return Ok(TimestampOptions {
                unit: time_unit,
                tz: None,
            });
        }

        // Attempt to load from len-prefixed bytes
        let tz_bytes = &data[3..][..tz_len];
        let tz: Arc<str> = str::from_utf8(tz_bytes)
            .map_err(|e| vortex_err!("timezone is not valid utf8 string: {e}"))?
            .to_string()
            .into();
        Ok(TimestampOptions {
            unit: time_unit,
            tz: Some(tz),
        })
    }

    fn validate_dtype(
        &self,
        _metadata: &Self::Metadata,
        storage_dtype: &DType,
    ) -> VortexResult<()> {
        vortex_ensure!(
            matches!(storage_dtype, DType::Primitive(PType::I64, _)),
            "Timestamp storage dtype must be i64"
        );
        Ok(())
    }
}
