// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Flat segments whose row-proportional buffers are interleaved in stripes of rows.
//!
//! A flat segment normally stores each buffer contiguously, so the bytes for one row are spread
//! over as many ranges as the array has buffers. A striped segment stores stripe 0 of every
//! striped buffer, then stripe 1 of every striped buffer, and so on, followed by the remaining
//! bytes in their usual order. The bytes for a block of rows are then one contiguous range.
//!
//! Readers keep working with logical offsets, i.e. those of the unstriped segment that the array
//! tree describes: [`StripedSegmentSource`] maps them to physical ranges and reassembles whole
//! segments.

use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::try_join_all;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::segments::SegmentFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;

/// A buffer of a striped segment: its logical byte range and the bytes of each stripe.
#[derive(Clone, Copy, PartialEq, Eq, prost::Message)]
pub struct StripedBuffer {
    /// Logical offset of the buffer within the segment.
    #[prost(uint64, tag = "1")]
    pub offset: u64,
    /// Length of the buffer in bytes.
    #[prost(uint64, tag = "2")]
    pub length: u64,
    /// Bytes of the buffer in each full stripe.
    #[prost(uint64, tag = "3")]
    pub bytes_per_stripe: u64,
}

/// A contiguous run of bytes at `logical` in the unstriped segment and `physical` on disk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Extent {
    logical: u64,
    physical: u64,
    len: u64,
}

/// The mapping between logical and physical offsets of a striped segment.
#[derive(Debug)]
pub(super) struct StripeMap {
    /// Extents sorted by logical offset, covering the whole segment.
    extents: Vec<Extent>,
}

impl StripeMap {
    pub(super) fn try_new(segment_len: u64, buffers: &[StripedBuffer]) -> VortexResult<Self> {
        let mut sorted = buffers.to_vec();
        sorted.sort_unstable_by_key(|buffer| buffer.offset);
        let mut previous_end = 0;
        for buffer in &sorted {
            vortex_ensure!(buffer.bytes_per_stripe > 0, "Stripe must not be empty");
            vortex_ensure!(
                buffer.offset >= previous_end,
                "Striped buffers must not overlap"
            );
            previous_end = buffer
                .offset
                .checked_add(buffer.length)
                .filter(|end| *end <= segment_len)
                .ok_or_else(|| vortex_error::vortex_err!("Striped buffer out of bounds"))?;
        }

        let mut extents = Vec::new();
        let mut physical = 0u64;
        let stripes = sorted
            .iter()
            .map(|buffer| buffer.length.div_ceil(buffer.bytes_per_stripe))
            .max()
            .unwrap_or(0);
        for stripe in 0..stripes {
            for buffer in &sorted {
                let start = stripe * buffer.bytes_per_stripe;
                if start >= buffer.length {
                    continue;
                }
                let len = buffer.bytes_per_stripe.min(buffer.length - start);
                extents.push(Extent {
                    logical: buffer.offset + start,
                    physical,
                    len,
                });
                physical += len;
            }
        }
        let mut logical = 0u64;
        for buffer in sorted
            .iter()
            .map(|buffer| (buffer.offset, buffer.offset + buffer.length))
            .chain([(segment_len, segment_len)])
        {
            if buffer.0 > logical {
                extents.push(Extent {
                    logical,
                    physical,
                    len: buffer.0 - logical,
                });
                physical += buffer.0 - logical;
            }
            logical = buffer.1;
        }
        extents.sort_unstable_by_key(|extent| extent.logical);
        Ok(Self { extents })
    }

    /// The physical ranges holding the logical `range`, in logical order, with physically
    /// adjacent pieces merged.
    pub(super) fn physical_ranges(&self, range: Range<u64>) -> Vec<Range<u64>> {
        let first = self
            .extents
            .partition_point(|extent| extent.logical + extent.len <= range.start);
        let mut pieces: Vec<Range<u64>> = Vec::new();
        for extent in &self.extents[first..] {
            if extent.logical >= range.end {
                break;
            }
            let start = range.start.max(extent.logical);
            let end = range.end.min(extent.logical + extent.len);
            let physical = extent.physical + (start - extent.logical)
                ..extent.physical + (end - extent.logical);
            match pieces.last_mut() {
                Some(last) if last.end == physical.start => last.end = physical.end,
                _ => pieces.push(physical),
            }
        }
        pieces
    }

    /// Reorder an unstriped segment into its striped form.
    pub(super) fn stripe(&self, logical: &[u8], into: &mut ByteBufferMut) {
        let start = into.len();
        into.extend_from_slice(logical);
        let physical = &mut into.as_mut_slice()[start..];
        for extent in &self.extents {
            let (from, to) = Self::usize_ranges(extent);
            physical[to].copy_from_slice(&logical[from]);
        }
    }

    /// Reorder a striped segment back into its unstriped form.
    pub(super) fn unstripe(&self, physical: &[u8], into: &mut ByteBufferMut) {
        let start = into.len();
        into.extend_from_slice(physical);
        let logical = &mut into.as_mut_slice()[start..];
        for extent in &self.extents {
            let (to, from) = Self::usize_ranges(extent);
            logical[to].copy_from_slice(&physical[from]);
        }
    }

    /// The logical and physical byte ranges of `extent`.
    #[allow(clippy::cast_possible_truncation)]
    fn usize_ranges(extent: &Extent) -> (Range<usize>, Range<usize>) {
        // Extents lie within an in-memory segment, so they fit in usize.
        let (logical, physical, len) = (
            extent.logical as usize,
            extent.physical as usize,
            extent.len as usize,
        );
        (logical..logical + len, physical..physical + len)
    }
}

/// Serves logical byte ranges of one striped segment from its physical layout.
pub(super) struct StripedSegmentSource {
    inner: Arc<dyn SegmentSource>,
    segment_id: SegmentId,
    map: Arc<StripeMap>,
}

impl StripedSegmentSource {
    pub(super) fn new(
        inner: Arc<dyn SegmentSource>,
        segment_id: SegmentId,
        map: Arc<StripeMap>,
    ) -> Self {
        Self {
            inner,
            segment_id,
            map,
        }
    }
}

impl SegmentSource for StripedSegmentSource {
    fn preferred_read_size(&self) -> Option<u64> {
        self.inner.preferred_read_size()
    }

    fn segment_len(&self, id: SegmentId) -> Option<u64> {
        self.inner.segment_len(id)
    }

    fn request(&self, id: SegmentId) -> SegmentFuture {
        let segment = self.inner.request(id);
        if id != self.segment_id {
            return segment;
        }
        let map = Arc::clone(&self.map);
        async move {
            let segment = segment.await?;
            let alignment = segment.alignment();
            let physical = segment.try_into_host_sync()?;
            let mut logical = ByteBufferMut::with_capacity_aligned(physical.len(), alignment);
            map.unstripe(&physical, &mut logical);
            Ok(BufferHandle::new_host(logical.freeze()))
        }
        .boxed()
    }

    fn request_range(&self, id: SegmentId, range: Range<u64>) -> SegmentFuture {
        self.request_ranges(id, vec![range])
            .pop()
            .unwrap_or_else(|| async { vortex_bail!("No range requested") }.boxed())
    }

    fn request_ranges(&self, id: SegmentId, ranges: Vec<Range<u64>>) -> Vec<SegmentFuture> {
        if id != self.segment_id {
            return self.inner.request_ranges(id, ranges);
        }
        let pieces = ranges
            .iter()
            .map(|range| self.map.physical_ranges(range.clone()))
            .collect::<Vec<_>>();
        let mut reads = self
            .inner
            .request_ranges(id, pieces.iter().flatten().cloned().collect())
            .into_iter();
        pieces
            .iter()
            .map(|pieces| {
                let reads = reads.by_ref().take(pieces.len()).collect::<Vec<_>>();
                async move {
                    let mut buffers = try_join_all(reads).await?;
                    if buffers.len() == 1
                        && let Some(buffer) = buffers.pop()
                    {
                        return Ok(buffer);
                    }
                    let len = buffers.iter().map(BufferHandle::len).sum();
                    let mut joined = ByteBufferMut::with_capacity(len);
                    for buffer in buffers {
                        let buffer: ByteBuffer = buffer.try_into_host_sync()?;
                        joined.extend_from_slice(&buffer);
                    }
                    Ok(BufferHandle::new_host(joined.freeze()))
                }
                .boxed()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn buffers() -> Vec<StripedBuffer> {
        vec![
            StripedBuffer {
                offset: 4,
                length: 10,
                bytes_per_stripe: 4,
            },
            StripedBuffer {
                offset: 20,
                length: 6,
                bytes_per_stripe: 2,
            },
        ]
    }

    #[test]
    fn stripe_round_trips() -> VortexResult<()> {
        let logical = (0u8..32).collect::<Vec<_>>();
        let map = StripeMap::try_new(32, &buffers())?;
        let mut physical = ByteBufferMut::empty();
        map.stripe(&logical, &mut physical);
        assert_eq!(
            physical.as_slice(),
            &[
                4, 5, 6, 7, 20, 21, // stripe 0
                8, 9, 10, 11, 22, 23, // stripe 1
                12, 13, 24, 25, // stripe 2
                0, 1, 2, 3, 14, 15, 16, 17, 18, 19, 26, 27, 28, 29, 30, 31,
            ]
        );
        let mut round_trip = ByteBufferMut::empty();
        map.unstripe(&physical, &mut round_trip);
        assert_eq!(round_trip.as_slice(), logical.as_slice());
        Ok(())
    }

    #[rstest]
    #[case::within_stripe(5..7, vec![1..3])]
    #[case::same_stripe_of_both_buffers(8..12, vec![6..10])]
    #[case::across_stripes(6..10, vec![2..4, 6..8])]
    #[case::unstriped(0..4, vec![16..20])]
    #[case::stripe_then_gap(12..16, vec![12..14, 20..22])]
    fn maps_logical_ranges(
        #[case] logical: Range<u64>,
        #[case] expected: Vec<Range<u64>>,
    ) -> VortexResult<()> {
        let map = StripeMap::try_new(32, &buffers())?;
        assert_eq!(map.physical_ranges(logical), expected);
        Ok(())
    }
}
