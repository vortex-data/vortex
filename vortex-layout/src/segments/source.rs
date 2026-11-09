// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use futures::FutureExt;
use futures::TryFutureExt;
use futures::future::BoxFuture;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::segments::SegmentId;
/// Static future resolving to a segment byte buffer.
pub type SegmentFuture = BoxFuture<'static, VortexResult<BufferHandle>>;

/// The bytes of a segment that a read asks for.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SegmentBytes {
    /// The whole segment.
    All,
    /// The bytes in this range, counted from the start of the segment, that lie in the segment.
    /// A reader that guesses where its bytes are reads fewer bytes past the end of a segment,
    /// not an error.
    Range(Range<u64>),
    /// The last bytes of the segment: this many, or all of them when the segment is shorter.
    Last(u64),
}

impl SegmentBytes {
    /// The range of a segment of `len` bytes that these bytes cover.
    pub fn resolve(&self, len: u64) -> VortexResult<Range<u64>> {
        match self {
            Self::All => Ok(0..len),
            Self::Range(range) => {
                if range.start > range.end {
                    vortex_bail!("Bytes {range:?} of a segment end before they start");
                }
                Ok(range.start.min(len)..range.end.min(len))
            }
            Self::Last(count) => Ok(len - (*count).min(len)..len),
        }
    }

    /// Slices these bytes out of `segment`, which holds the whole segment.
    pub fn slice(&self, segment: BufferHandle) -> VortexResult<BufferHandle> {
        if *self == Self::All {
            return Ok(segment);
        }

        // A part of a segment keeps no alignment, so it can start at any byte.
        let range = self.resolve(segment.len() as u64)?;
        let segment = segment.ensure_aligned(Alignment::none())?;
        Ok(segment.slice(usize::try_from(range.start)?..usize::try_from(range.end)?))
    }
}

/// Provides segment data to a [`crate::LayoutReader`].
///
/// Implementations may issue asynchronous file reads, object-store requests, cache lookups, or
/// in-memory buffer slices. Returned futures must be independent and safe to poll concurrently.
pub trait SegmentSource: 'static + Send + Sync {
    /// Request a segment, returning a future that will eventually resolve to the segment data.
    fn request(&self, id: SegmentId) -> SegmentFuture;

    /// Whether [`request_bytes`](Self::request_bytes) reads only the bytes asked for. When it
    /// does not, a reader that needs several parts of a segment reads the whole segment once
    /// instead.
    fn reads_ranges(&self) -> bool {
        false
    }

    /// Request some bytes of a segment.
    ///
    /// By default this reads the whole segment and slices it. A source that can read part of a
    /// segment, such as a file, reads only the bytes, and says so with
    /// [`reads_ranges`](Self::reads_ranges).
    fn request_bytes(&self, id: SegmentId, bytes: SegmentBytes) -> SegmentFuture {
        if bytes == SegmentBytes::All {
            return self.request(id);
        }
        self.request(id)
            .and_then(move |segment| futures::future::ready(bytes.slice(segment)))
            .boxed()
    }
}
