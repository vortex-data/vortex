mod events;
mod sink;
mod source;
mod test;

use std::fmt::Display;
use std::ops::Deref;

pub use events::*;
pub use sink::*;
pub use source::*;
pub use test::*;
use vortex_error::VortexError;

/// The identifier for a single segment.
// TODO(ngates): should this be a `[u8]` instead? Allowing for arbitrary segment identifiers?
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SegmentId(u32);

impl From<u32> for SegmentId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl TryFrom<usize> for SegmentId {
    type Error = VortexError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Ok(Self::from(u32::try_from(value)?))
    }
}

impl Deref for SegmentId {
    type Target = u32;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Display for SegmentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SegmentId({})", self.0)
    }
}
