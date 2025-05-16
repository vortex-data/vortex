use std::fmt::{Display, Formatter};

use jiff::Span;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use vortex_error::VortexResult;

/// Time units for temporal data.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, IntoPrimitive, TryFromPrimitive,
)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u8)]
pub enum TimeUnit {
    /// Nanoseconds
    Nano = 0,
    /// Microseconds
    Micro = 1,
    /// Milliseconds
    Milli = 2,
    /// Seconds
    Second = 3,
    /// Days
    Day = 4,
}

impl TimeUnit {
    /// Convert to a Jiff span.
    pub fn to_jiff_span(&self, v: i64) -> VortexResult<Span> {
        Ok(match self {
            TimeUnit::Nano => Span::new().try_nanoseconds(v)?,
            TimeUnit::Micro => Span::new().try_microseconds(v)?,
            TimeUnit::Milli => Span::new().try_milliseconds(v)?,
            TimeUnit::Second => Span::new().try_seconds(v)?,
            TimeUnit::Day => Span::new().try_days(v)?,
        })
    }
}

impl Display for TimeUnit {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nano => write!(f, "ns"),
            Self::Micro => write!(f, "µs"),
            Self::Milli => write!(f, "ms"),
            Self::Second => write!(f, "s"),
            Self::Day => write!(f, "days"),
        }
    }
}
