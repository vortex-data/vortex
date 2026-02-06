// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::future::BoxFuture;
use vortex_array::buffer::BufferHandle;
use vortex_error::VortexResult;

use crate::segments::SegmentId;

/// Static future resolving to a segment byte buffer.
pub type SegmentFuture = BoxFuture<'static, VortexResult<BufferHandle>>;

/// Priority levels for segment requests.
///
/// Lower values indicate higher priority. Requests with higher priority
/// will be serviced before lower priority requests when the I/O system
/// has to choose between multiple pending requests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum SegmentPriority {
    /// Zone map segments - highest priority.
    /// These enable pruning, which can eliminate the need to fetch other segments.
    ZoneMap = 0,
    /// Filter column segments - medium priority.
    /// These enable filtering, which reduces downstream work.
    FilterColumn = 1,
    /// Projection column segments - default/lowest priority.
    /// These are the final output data.
    #[default]
    ProjectionColumn = 2,
}

/// A trait for providing segment data to a [`crate::LayoutReader`].
pub trait SegmentSource: 'static + Send + Sync {
    /// Request a segment with default priority, returning a future that will eventually
    /// resolve to the segment data.
    fn request(&self, id: SegmentId) -> SegmentFuture;

    /// Request a segment with specified priority.
    ///
    /// Higher priority segments (lower numeric value) will be fetched before
    /// lower priority segments when the I/O system is choosing between pending requests.
    fn request_with_priority(&self, id: SegmentId, priority: SegmentPriority) -> SegmentFuture {
        // Default implementation ignores priority
        let _ = priority;
        self.request(id)
    }
}
