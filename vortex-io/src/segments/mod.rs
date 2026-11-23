// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Segments: the blobs a source names before a scan runs, and the sources that read them.

mod file;
mod read;
mod shared;

use std::fmt;
use std::fmt::Display;
use std::ops::Deref;
use std::sync::Arc;

pub use file::*;
use futures::future::BoxFuture;
pub use shared::*;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_error::VortexError;
use vortex_error::VortexResult;

/// Identifier for a single physical segment referenced by a layout.
///
/// Segment ids are local to a source. A Vortex file's footer maps ids to physical offsets; other
/// formats assign ids to the blobs their metadata names, and custom storage systems may map them
/// to object-store keys or other random-access locations.
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SegmentId({})", self.0)
    }
}

/// Static future resolving to a segment byte buffer.
pub type SegmentFuture = BoxFuture<'static, VortexResult<BufferHandle>>;

/// Reads segments by id.
///
/// Implementations may issue asynchronous file reads, object-store requests, cache lookups, or
/// in-memory buffer slices. Returned futures must be independent and safe to poll concurrently.
pub trait SegmentSource: 'static + Send + Sync {
    /// Request a segment, returning a future that will eventually resolve to the segment data.
    fn request(&self, id: SegmentId) -> SegmentFuture;
}

impl SegmentSource for Arc<dyn SegmentSource> {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        self.as_ref().request(id)
    }
}

/// Where a segment's bytes live in its source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentLocation {
    /// Byte offset of the segment from the start of the source.
    pub offset: u64,
    /// Length of the segment in bytes.
    pub length: u32,
    /// Alignment the segment's bytes must have once delivered.
    pub alignment: Alignment,
}
