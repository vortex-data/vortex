// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Display, Formatter};

use jiff::Span;
use num_enum::IntoPrimitive;
use vortex_error::{VortexError, VortexResult, vortex_bail};

/// Time units for temporal data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, IntoPrimitive)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u8)]
pub enum TimeUnit {
    /// Nanoseconds
    Nanoseconds = 0,
    /// Microseconds
    Microseconds = 1,
    /// Milliseconds
    Milliseconds = 2,
    /// Seconds
    Seconds = 3,
    /// Days
    Days = 4,
}

impl TryFrom<u8> for TimeUnit {
    type Error = VortexError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(TimeUnit::Nanoseconds),
            1 => Ok(TimeUnit::Microseconds),
            2 => Ok(TimeUnit::Milliseconds),
            3 => Ok(TimeUnit::Seconds),
            4 => Ok(TimeUnit::Days),
            _ => vortex_bail!("invalid time unit: {value}u8"),
        }
    }
}

impl TimeUnit {
    /// Convert to a Jiff span.
    pub fn to_jiff_span(&self, v: i64) -> VortexResult<Span> {
        Ok(match self {
            TimeUnit::Nanoseconds => Span::new().try_nanoseconds(v)?,
            TimeUnit::Microseconds => Span::new().try_microseconds(v)?,
            TimeUnit::Milliseconds => Span::new().try_milliseconds(v)?,
            TimeUnit::Seconds => Span::new().try_seconds(v)?,
            TimeUnit::Days => Span::new().try_days(v)?,
        })
    }
}

impl Display for TimeUnit {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nanoseconds => write!(f, "ns"),
            Self::Microseconds => write!(f, "µs"),
            Self::Milliseconds => write!(f, "ms"),
            Self::Seconds => write!(f, "s"),
            Self::Days => write!(f, "days"),
        }
    }
}
