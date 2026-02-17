// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cache;
mod shared;
mod sink;
mod source;

#[cfg(any(test, feature = "_test-harness"))]
mod test;

use std::fmt::Display;
use std::ops::Deref;

pub use cache::*;
pub use shared::*;
pub use sink::*;
pub use source::*;
#[cfg(any(test, feature = "_test-harness"))]
pub use test::*;

/// The identifier for a single segment.
// TODO(ngates): should this be a `[u8]` instead? Allowing for arbitrary segment identifiers?
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SegmentId(u32);

impl From<u32> for SegmentId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<usize> for SegmentId {
    fn from(value: usize) -> Self {
        Self::from(u32::try_from(value).expect("SegmentID must fit 32-bits unsigned integer"))
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
