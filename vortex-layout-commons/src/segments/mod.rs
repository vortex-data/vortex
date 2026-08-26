// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Segment access contracts for layouts.
//!
//! Layouts refer to byte ranges by [`SegmentId`]. A [`SegmentSource`] resolves those ids to buffer
//! handles for readers, while a [`SegmentSink`] assigns ids when writers emit buffers. Cache and
//! request-sharing policies belong to the runtime using these contracts.

mod sink;
mod source;

use std::fmt::Display;
use std::ops::Deref;

pub use sink::*;
pub use source::*;
use vortex_error::VortexError;

/// Identifier for a single physical segment referenced by a layout.
///
/// Segment ids are local to a file or segment source. The file footer maps ids to physical offsets;
/// custom storage systems may map them to object-store keys or other random-access locations.
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
