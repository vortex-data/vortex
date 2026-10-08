// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Flat segments whose row-proportional buffers are interleaved in stripes of rows.
//!
//! A flat segment normally stores each buffer contiguously, so the bytes for one row are spread
//! over as many ranges as the array has buffers. A striped segment stores stripe 0 of every
//! striped buffer, then stripe 1 of every striped buffer, and so on, followed by the remaining
//! bytes in their usual order. The bytes for a block of rows are then one contiguous range.
//!
//! Most striped buffers hold the same number of bytes in every stripe. Buffers whose elements
//! are not proportional to rows, such as the indices and values of sorted patches, are instead
//! split by a per-stripe element count, so each stripe also carries the patches of its rows.
//!
//! Readers keep working with logical offsets, i.e. those of the unstriped segment that the array
//! tree describes: [`StripedSegmentSource`] maps them to physical ranges and reassembles whole
//! segments.

use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::TryFutureExt;
use futures::future::try_join_all;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::segments::SegmentFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;

/// A buffer of a striped segment: its logical byte range and how it is split into stripes.
#[derive(Clone, Copy, PartialEq, Eq, prost::Message)]
pub struct StripedBuffer {
    /// Logical offset of the buffer within the segment.
    #[prost(uint64, tag = "1")]
    pub offset: u64,
    /// Length of the buffer in bytes.
    #[prost(uint64, tag = "2")]
    pub length: u64,
    /// Bytes of the buffer in each full stripe, or zero when the buffer is split by the
    /// segment's per-stripe element counts.
    #[prost(uint64, tag = "3")]
    pub bytes_per_stripe: u64,
    /// Bytes per element of a buffer split by the per-stripe element counts.
    #[prost(uint64, tag = "4")]
    pub bytes_per_element: u64,
}

/// A stretch of the logical segment, in logical order.
#[derive(Debug, Clone)]
enum Region {
    /// Unstriped bytes, stored after the stripes starting at `physical`.
    Gap { logical: Range<u64>, physical: u64 },
    /// The striped buffer at this index of [`StripeMap::buffers`].
    Buffer(usize),
}

/// The mapping between logical and physical offsets of a striped segment.
///
/// The mapping is computed rather than stored, so a map costs one entry per stripe at most.
#[derive(Debug)]
pub(super) struct StripeMap {
    /// Striped buffers sorted by logical offset, which is also their order within a stripe.
    buffers: Vec<StripedBuffer>,
    regions: Vec<Region>,
    stripes: u64,
    /// The elements before each stripe of the element-split buffers, `stripes + 1` entries, or
    /// empty when there are none.
    element_starts: Vec<u32>,
    /// Bytes in each full stripe of the evenly split buffers.
    even_bytes_per_stripe: u64,
    /// Bytes per element over all element-split buffers.
    bytes_per_element: u64,
}

impl StripeMap {
    pub(super) fn try_new(
        segment_len: u64,
        buffers: &[StripedBuffer],
        element_counts: &[u32],
    ) -> VortexResult<Self> {
        let mut buffers = buffers.to_vec();
        buffers.sort_unstable_by_key(|buffer| buffer.offset);

        let mut even_stripes = None;
        let mut element_split = false;
        let mut previous_end = 0;
        for buffer in &buffers {
            vortex_ensure!(
                buffer.offset >= previous_end,
                "Striped buffers must not overlap"
            );
            previous_end = buffer
                .offset
                .checked_add(buffer.length)
                .filter(|end| *end <= segment_len)
                .ok_or_else(|| vortex_err!("Striped buffer out of bounds"))?;
            match (buffer.bytes_per_stripe, buffer.bytes_per_element) {
                (0, 0) => vortex_bail!("Stripe must not be empty"),
                (bytes_per_stripe, 0) => {
                    let stripes = buffer.length.div_ceil(bytes_per_stripe);
                    vortex_ensure!(
                        *even_stripes.get_or_insert(stripes) == stripes,
                        "Striped buffers must have the same number of stripes"
                    );
                }
                (0, _) => element_split = true,
                _ => vortex_bail!("Striped buffer must be split by bytes or by elements"),
            }
        }

        let element_starts = if element_split {
            let mut starts = Vec::with_capacity(element_counts.len() + 1);
            let mut total = 0u32;
            starts.push(total);
            for count in element_counts {
                total = total
                    .checked_add(*count)
                    .ok_or_else(|| vortex_err!("Stripe element count overflow"))?;
                starts.push(total);
            }
            starts
        } else {
            Vec::new()
        };
        let stripes = match (even_stripes, element_split) {
            (Some(stripes), false) => stripes,
            (stripes, true) => {
                let counted = element_counts.len() as u64;
                vortex_ensure!(
                    stripes.is_none_or(|stripes| stripes == counted),
                    "Stripe element counts must cover every stripe"
                );
                counted
            }
            (None, false) => 0,
        };
        let elements = u64::from(element_starts.last().copied().unwrap_or(0));
        for buffer in buffers.iter().filter(|buffer| buffer.bytes_per_element > 0) {
            vortex_ensure!(
                elements.checked_mul(buffer.bytes_per_element) == Some(buffer.length),
                "Element-split buffer length must match its element counts"
            );
        }

        let even_bytes_per_stripe = buffers.iter().map(|buffer| buffer.bytes_per_stripe).sum();
        let bytes_per_element = buffers.iter().map(|buffer| buffer.bytes_per_element).sum();
        let mut physical: u64 = buffers.iter().map(|buffer| buffer.length).sum();
        let mut regions = Vec::new();
        let mut logical = 0u64;
        for (index, buffer) in buffers.iter().enumerate() {
            if buffer.offset > logical {
                regions.push(Region::Gap {
                    logical: logical..buffer.offset,
                    physical,
                });
                physical += buffer.offset - logical;
            }
            regions.push(Region::Buffer(index));
            logical = buffer.offset + buffer.length;
        }
        if segment_len > logical {
            regions.push(Region::Gap {
                logical: logical..segment_len,
                physical,
            });
        }

        Ok(Self {
            buffers,
            regions,
            stripes,
            element_starts,
            even_bytes_per_stripe,
            bytes_per_element,
        })
    }

    /// The elements before each stripe of the element-split buffers, with a final entry for the
    /// total, or `None` when no buffer is split by elements.
    pub(super) fn element_starts(&self) -> Option<&[u32]> {
        (!self.element_starts.is_empty()).then_some(self.element_starts.as_slice())
    }

    /// The element count of each stripe of the element-split buffers.
    pub(super) fn element_counts(&self) -> Vec<u32> {
        self.element_starts
            .windows(2)
            .map(|window| window[1] - window[0])
            .collect()
    }

    /// The bytes of `buffer` in `stripe`, relative to the start of the buffer.
    fn stripe_range(&self, buffer: &StripedBuffer, stripe: u64) -> Range<u64> {
        if buffer.bytes_per_stripe > 0 {
            let start = (stripe * buffer.bytes_per_stripe).min(buffer.length);
            start..(start + buffer.bytes_per_stripe).min(buffer.length)
        } else {
            let elements = self.elements(stripe);
            elements.start * buffer.bytes_per_element..elements.end * buffer.bytes_per_element
        }
    }

    fn elements(&self, stripe: u64) -> Range<u64> {
        let stripe = usize::try_from(stripe).unwrap_or(usize::MAX);
        match self.element_starts.get(stripe..=stripe.saturating_add(1)) {
            Some([start, end]) => u64::from(*start)..u64::from(*end),
            _ => 0..0,
        }
    }

    /// The stripe holding byte `offset` of `buffer`, relative to the start of the buffer.
    fn stripe_of(&self, buffer: &StripedBuffer, offset: u64) -> u64 {
        offset
            .checked_div(buffer.bytes_per_stripe)
            .unwrap_or_else(|| {
                let element = offset / buffer.bytes_per_element;
                self.element_starts[1..].partition_point(|start| u64::from(*start) <= element)
                    as u64
            })
    }

    /// The physical offset of stripe `stripe` of the buffer at `index`.
    fn piece_offset(&self, index: usize, stripe: u64) -> u64 {
        // Only the last stripe of an evenly split buffer is short, so every earlier stripe holds
        // whole stripes of the evenly split buffers plus its share of the elements.
        let stripe_start = stripe * self.even_bytes_per_stripe
            + u64::from(
                usize::try_from(stripe)
                    .ok()
                    .and_then(|stripe| self.element_starts.get(stripe))
                    .copied()
                    .unwrap_or(0),
            ) * self.bytes_per_element;
        stripe_start
            + self.buffers[..index]
                .iter()
                .map(|buffer| self.stripe_range(buffer, stripe))
                .map(|range| range.end - range.start)
                .sum::<u64>()
    }

    /// The physical ranges holding the logical `range`, in logical order, with physically
    /// adjacent pieces merged.
    pub(super) fn physical_ranges(&self, range: Range<u64>) -> Vec<Range<u64>> {
        let mut pieces: Vec<Range<u64>> = Vec::new();
        let mut push = |piece: Range<u64>| match pieces.last_mut() {
            _ if piece.is_empty() => {}
            Some(last) if last.end == piece.start => last.end = piece.end,
            _ => pieces.push(piece),
        };
        for region in &self.regions {
            match region {
                Region::Gap { logical, physical } => {
                    let start = range.start.max(logical.start);
                    let end = range.end.min(logical.end);
                    if start < end {
                        let offset = physical + (start - logical.start);
                        push(offset..offset + (end - start));
                    }
                }
                Region::Buffer(index) => {
                    let buffer = &self.buffers[*index];
                    let start = range.start.max(buffer.offset);
                    let end = range.end.min(buffer.offset + buffer.length);
                    if start >= end {
                        continue;
                    }
                    let (start, end) = (start - buffer.offset, end - buffer.offset);
                    let mut stripe = self.stripe_of(buffer, start);
                    while stripe < self.stripes {
                        let bytes = self.stripe_range(buffer, stripe);
                        if bytes.start >= end {
                            break;
                        }
                        let piece_start = start.max(bytes.start);
                        let piece_end = end.min(bytes.end);
                        if piece_start < piece_end {
                            let offset =
                                self.piece_offset(*index, stripe) + (piece_start - bytes.start);
                            push(offset..offset + (piece_end - piece_start));
                        }
                        stripe += 1;
                    }
                }
            }
        }
        pieces
    }

    /// Call `f(logical, physical)` for every contiguous run of the segment.
    fn for_each_extent(&self, mut f: impl FnMut(Range<usize>, Range<usize>)) {
        #[allow(clippy::cast_possible_truncation)]
        let mut emit = |logical: u64, physical: u64, len: u64| {
            // Extents lie within an in-memory segment, so they fit in usize.
            let (logical, physical, len) = (logical as usize, physical as usize, len as usize);
            f(logical..logical + len, physical..physical + len);
        };
        for stripe in 0..self.stripes {
            for (index, buffer) in self.buffers.iter().enumerate() {
                let bytes = self.stripe_range(buffer, stripe);
                if !bytes.is_empty() {
                    emit(
                        buffer.offset + bytes.start,
                        self.piece_offset(index, stripe),
                        bytes.end - bytes.start,
                    );
                }
            }
        }
        for region in &self.regions {
            if let Region::Gap { logical, physical } = region {
                emit(logical.start, *physical, logical.end - logical.start);
            }
        }
    }

    /// Reorder an unstriped segment into its striped form.
    pub(super) fn stripe(&self, logical: &[u8], physical: &mut [u8]) {
        self.for_each_extent(|from, to| physical[to].copy_from_slice(&logical[from]));
    }

    /// Reorder a striped segment back into its unstriped form.
    pub(super) fn unstripe(&self, physical: &[u8], logical: &mut [u8]) {
        self.for_each_extent(|to, from| logical[to].copy_from_slice(&physical[from]));
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
            let mut logical = ByteBufferMut::zeroed_aligned(physical.len(), alignment);
            map.unstripe(&physical, logical.as_mut_slice());
            Ok(BufferHandle::new_host(logical.freeze()))
        }
        .boxed()
    }

    fn request_range(&self, id: SegmentId, range: Range<u64>) -> SegmentFuture {
        self.request_ranges(id, vec![range])
            .pop()
            .unwrap_or_else(|| async { vortex_bail!("No range requested") }.boxed())
    }

    /// Read every requested range in one batch of physical reads, merging pieces that touch, so
    /// the rows a request selects cost one read per contiguous run of stripes.
    fn request_ranges(&self, id: SegmentId, ranges: Vec<Range<u64>>) -> Vec<SegmentFuture> {
        if id != self.segment_id {
            return self.inner.request_ranges(id, ranges);
        }
        let pieces = ranges
            .into_iter()
            .map(|range| self.map.physical_ranges(range))
            .collect::<Vec<_>>();
        let mut reads = pieces.iter().flatten().cloned().collect::<Vec<_>>();
        reads.sort_unstable_by_key(|read| read.start);
        reads.dedup_by(|next, read| {
            let touches = next.start <= read.end;
            if touches {
                read.end = read.end.max(next.end);
            }
            touches
        });
        let reads = Arc::new(reads);
        let batch = try_join_all(self.inner.request_ranges(id, reads.to_vec()))
            .map_err(Arc::new)
            .boxed()
            .shared();
        pieces
            .into_iter()
            .map(|pieces| {
                let batch = batch.clone();
                let reads = Arc::clone(&reads);
                async move {
                    let buffers: Vec<BufferHandle> = batch.await.map_err(VortexError::from)?;
                    let slice = |piece: &Range<u64>| -> VortexResult<BufferHandle> {
                        let read = reads.partition_point(|read| read.end < piece.end);
                        let base = reads
                            .get(read)
                            .filter(|read| read.start <= piece.start)
                            .ok_or_else(|| vortex_err!("Striped piece outside its reads"))?
                            .start;
                        let start = usize::try_from(piece.start - base)?;
                        let end = usize::try_from(piece.end - base)?;
                        Ok(buffers[read].slice(start..end))
                    };
                    if let [piece] = pieces.as_slice() {
                        return slice(piece);
                    }
                    let len = pieces
                        .iter()
                        .map(|piece| piece.end - piece.start)
                        .sum::<u64>();
                    let mut joined = ByteBufferMut::with_capacity(usize::try_from(len)?);
                    for piece in &pieces {
                        let buffer: ByteBuffer = slice(piece)?.try_into_host_sync()?;
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

    fn even(offset: u64, length: u64, bytes_per_stripe: u64) -> StripedBuffer {
        StripedBuffer {
            offset,
            length,
            bytes_per_stripe,
            bytes_per_element: 0,
        }
    }

    fn buffers() -> Vec<StripedBuffer> {
        vec![even(4, 10, 4), even(20, 6, 2)]
    }

    #[test]
    fn stripe_round_trips() -> VortexResult<()> {
        let logical = (0u8..32).collect::<Vec<_>>();
        let map = StripeMap::try_new(32, &buffers(), &[])?;
        let mut physical = ByteBufferMut::zeroed(32);
        map.stripe(&logical, physical.as_mut_slice());
        assert_eq!(
            physical.as_slice(),
            &[
                4, 5, 6, 7, 20, 21, // stripe 0
                8, 9, 10, 11, 22, 23, // stripe 1
                12, 13, 24, 25, // stripe 2
                0, 1, 2, 3, 14, 15, 16, 17, 18, 19, 26, 27, 28, 29, 30, 31,
            ]
        );
        let mut round_trip = ByteBufferMut::zeroed(32);
        map.unstripe(&physical, round_trip.as_mut_slice());
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
        let map = StripeMap::try_new(32, &buffers(), &[])?;
        assert_eq!(map.physical_ranges(logical), expected);
        Ok(())
    }

    /// Two element-split buffers (2 and 1 bytes per element) follow each stripe's even bytes.
    fn element_split() -> Vec<StripedBuffer> {
        vec![
            even(0, 6, 2),
            StripedBuffer {
                offset: 6,
                length: 8,
                bytes_per_stripe: 0,
                bytes_per_element: 2,
            },
            StripedBuffer {
                offset: 14,
                length: 4,
                bytes_per_stripe: 0,
                bytes_per_element: 1,
            },
        ]
    }

    #[test]
    fn element_split_buffers_follow_their_stripes() -> VortexResult<()> {
        let logical = (0u8..18).collect::<Vec<_>>();
        let map = StripeMap::try_new(18, &element_split(), &[1, 0, 3])?;
        let mut physical = ByteBufferMut::zeroed(18);
        map.stripe(&logical, physical.as_mut_slice());
        assert_eq!(
            physical.as_slice(),
            &[
                0, 1, 6, 7, 14, // stripe 0: one element
                2, 3, // stripe 1: no elements
                4, 5, 8, 9, 10, 11, 12, 13, 15, 16, 17, // stripe 2: three elements
            ]
        );
        let mut round_trip = ByteBufferMut::zeroed(18);
        map.unstripe(&physical, round_trip.as_mut_slice());
        assert_eq!(round_trip.as_slice(), logical.as_slice());
        assert_eq!(map.physical_ranges(10..14), vec![11..15]);
        assert_eq!(map.physical_ranges(6..10), vec![2..4, 9..11]);
        assert_eq!(map.element_counts(), vec![1, 0, 3]);
        Ok(())
    }

    #[test]
    fn element_counts_must_match_stripes() {
        assert!(StripeMap::try_new(18, &element_split(), &[1, 3]).is_err());
        assert!(StripeMap::try_new(18, &element_split(), &[1, 0, 2]).is_err());
    }
}
