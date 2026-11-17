// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Source inputs and byte access for scans.

use std::fmt;
use std::fmt::Display;
use std::ops::Deref;
use std::sync::Arc;

use futures::future::BoxFuture;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_error::VortexError;
use vortex_error::VortexResult;

use crate::execution::SegmentLocation;
use crate::execution::cache::ScanCache;
use crate::plan::LazyPlanRef;

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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SegmentId({})", self.0)
    }
}

/// Static future resolving to a segment byte buffer.
pub type SegmentFuture = BoxFuture<'static, VortexResult<BufferHandle>>;

/// Provides bytes to scan sources.
///
/// Implementations may issue asynchronous file reads, object-store requests, cache lookups, or
/// in-memory buffer slices. Returned futures must be independent and safe to poll concurrently.
pub trait SegmentSource: 'static + Send + Sync {
    /// Request a segment, returning a future that will eventually resolve to the segment data.
    fn request(&self, id: SegmentId) -> SegmentFuture;
}

/// The plan and byte source used to prepare a scan.
///
/// The lazy root defers lowering until preparation runs on a compute task. Segment identifiers
/// are local to this input. Reuse a cache only while the plan, dtype, locations, and source
/// describe the same data.
#[derive(Clone)]
pub struct ScanInput {
    /// The dtype of the root plan.
    pub dtype: DType,
    /// The root plan, initialized without reading payloads.
    pub root: LazyPlanRef,
    /// Byte locations indexed by source-local segment identifiers.
    pub locations: Arc<[SegmentLocation]>,
    /// Fetches bytes by source-local segment identifier.
    pub segments: Arc<dyn SegmentSource>,
    /// State shared by repeated scans of this input.
    pub cache: Option<ScanCache>,
}
