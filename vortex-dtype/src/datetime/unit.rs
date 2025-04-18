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
    Ns = 0,
    /// Microseconds
    Us = 1,
    /// Milliseconds
    Ms = 2,
    /// Seconds
    S = 3,
    /// Days
    D = 4,
}

impl TimeUnit {
    /// Convert to a Jiff span.
    pub fn to_jiff_span(&self, v: i64) -> VortexResult<Span> {
        Ok(match self {
            TimeUnit::Ns => Span::new().try_nanoseconds(v)?,
            TimeUnit::Us => Span::new().try_microseconds(v)?,
            TimeUnit::Ms => Span::new().try_milliseconds(v)?,
            TimeUnit::S => Span::new().try_seconds(v)?,
            TimeUnit::D => Span::new().try_days(v)?,
        })
    }
}

impl Display for TimeUnit {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ns => write!(f, "ns"),
            Self::Us => write!(f, "µs"),
            Self::Ms => write!(f, "ms"),
            Self::S => write!(f, "s"),
            Self::D => write!(f, "days"),
        }
    }
}
