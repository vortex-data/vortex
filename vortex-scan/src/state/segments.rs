// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use dashmap::DashMap;
use futures::future::BoxFuture;
use futures::stream::FuturesUnordered;
use futures::{FutureExt, Stream, StreamExt};
use std::any::Any;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, ready};
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexExpect, VortexResult};
use vortex_layout::segments::{SegmentId, SegmentSource, Segments};
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

/// The working set of segments used by the scan.
pub(super) struct SegmentCache {
    source: Arc<dyn SegmentSource>,

    futures: FuturesUnordered<BoxFuture<'static, VortexResult<(SegmentId, ByteBuffer)>>>,
    in_flight: HashSet<SegmentId>,

    ref_counts: HashMap<SegmentId, usize>,
    working_set: Arc<DashMap<SegmentId, ByteBuffer>>,
    working_set_size: u64,
}

impl SegmentCache {
    pub(super) fn new(source: Arc<dyn SegmentSource>) -> Self {
        Self {
            source,
            futures: FuturesUnordered::new(),
            in_flight: HashSet::new(),
            ref_counts: HashMap::new(),
            working_set: Arc::new(DashMap::new()),
            working_set_size: 0,
        }
    }

    pub(super) fn segments(&self) -> Arc<dyn Segments> {
        self.working_set.clone()
    }

    /// Launch requests for the given segments, returning the pending segments.
    pub(super) fn request<'a>(&mut self, segment_ids: impl IntoIterator<Item = &'a SegmentId>) {
        for segment_id in segment_ids.into_iter().copied() {
            if self.working_set.contains_key(&segment_id) {
                // We've already fetched this segment.
                continue;
            }

            if self.in_flight.contains(&segment_id) {
                // We're already in-flight.
                continue;
            }

            // Otherwise, we need to fetch this segment.
            let fut = self.source.request(segment_id);
            self.in_flight.insert(segment_id);
            self.futures.push(
                async move {
                    let buffer = fut.await?;
                    Ok((segment_id, buffer))
                }
                .boxed(),
            );
        }
    }

    pub(super) fn inflight_count(&self) -> usize {
        self.in_flight.len()
    }

    pub(super) fn segment_count(&self) -> usize {
        self.working_set.len() + self.in_flight.len()
    }

    pub(super) fn ref_counts(&self) -> &HashMap<SegmentId, usize> {
        &self.ref_counts
    }

    pub(super) fn acquire<'a, I: IntoIterator<Item = &'a SegmentId>>(&mut self, segment_ids: I) {
        for segment_id in segment_ids.into_iter() {
            *self.ref_counts.entry(*segment_id).or_default() += 1;
        }
    }

    /// Release the reference to the given segments, dropping any fetched buffers if possible.
    pub(super) fn release<'a, I: IntoIterator<Item = &'a SegmentId>>(&mut self, segment_ids: I) {
        for segment_id in segment_ids.into_iter() {
            let ref_count = self
                .ref_counts
                .get(segment_id)
                .vortex_expect("unknown segment");
            if *ref_count == 1 {
                if let Some((_, buffer)) = self.working_set.remove(segment_id) {
                    self.working_set_size -= buffer.len() as u64;
                }
                self.ref_counts.remove(segment_id);
            } else {
                let ref_count = self
                    .ref_counts
                    .get_mut(segment_id)
                    .vortex_expect("unknown segment");
                *ref_count -= 1;
            }
        }
    }
}

impl Stream for SegmentCache {
    type Item = VortexResult<SegmentId>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let Some(next) = ready!(self.futures.poll_next_unpin(cx)) else {
            return Poll::Ready(None);
        };

        match next {
            Ok((segment_id, buffer)) => {
                assert!(
                    self.in_flight.remove(&segment_id),
                    "Unrecognized segment ID {}",
                    segment_id
                );
                self.working_set_size += buffer.len() as u64;
                self.working_set.insert(segment_id, buffer);
                Poll::Ready(Some(Ok(segment_id)))
            }
            Err(e) => Poll::Ready(Some(Err(e))),
        }
    }
}

/// An opaque segment request returned from a `SegmentSource`.
pub trait SegmentRequest {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn size(&mut self) -> usize;
}

pub trait SegmentSource2 {
    /// Return the byte size of the given segment.
    fn size(&self, segment_id: &SegmentId) -> usize;

    /// Return an empty segment request.
    fn empty_request(&self) -> Box<dyn SegmentRequest>;

    /// Request to add the given segment into the request object.
    fn request_segment(&self, segment_id: &SegmentId, request: &mut dyn SegmentRequest) -> bool;

    /// Request the given segments.
    fn submit(&self, request: Box<dyn SegmentRequest>, callback: Arc<dyn SegmentCallback>);
}

pub trait SegmentCallback {
    fn on_segment(&self, segment_id: &SegmentId, buffer: VortexResult<ByteBuffer>);
}
