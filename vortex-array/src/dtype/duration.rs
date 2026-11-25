// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;

use num_enum::IntoPrimitive;
use vortex_error::VortexError;
use vortex_error::vortex_bail;

/// The unit of a duration value.
#[derive(Clone, Copy, Debug, Eq, Hash, IntoPrimitive, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u8)]
pub enum DurationUnit {
    /// Nanoseconds.
    Nanoseconds = 0,
    /// Microseconds.
    Microseconds = 1,
    /// Milliseconds.
    Milliseconds = 2,
    /// Seconds.
    Seconds = 3,
}

impl TryFrom<u8> for DurationUnit {
    type Error = VortexError;

    fn try_from(unit: u8) -> Result<Self, Self::Error> {
        Ok(match unit {
            0 => Self::Nanoseconds,
            1 => Self::Microseconds,
            2 => Self::Milliseconds,
            3 => Self::Seconds,
            _ => vortex_bail!("invalid duration unit {unit}"),
        })
    }
}

impl fmt::Display for DurationUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Nanoseconds => "ns",
            Self::Microseconds => "us",
            Self::Milliseconds => "ms",
            Self::Seconds => "s",
        })
    }
}
