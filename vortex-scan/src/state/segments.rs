// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use dashmap::DashMap;
use futures::channel::mpsc;
use futures::stream::BoxStream;
use futures::{Stream, StreamExt};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, ready};
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexExpect, VortexResult, vortex_err};
use vortex_layout::segments::{SegmentId, SegmentSource, Segments};
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

/// The working set of segments used by the scan.
pub(super) struct SegmentCache {
    requests_send: mpsc::UnboundedSender<SegmentId>,
    results: BoxStream<'static, (SegmentId, VortexResult<ByteBuffer>)>,
    in_flight: HashSet<SegmentId>,

    ref_counts: HashMap<SegmentId, usize>,

    working_set: Arc<DashMap<SegmentId, ByteBuffer>>,
    working_set_size: u64,
}

impl SegmentCache {
    pub(super) fn new(source: Arc<dyn SegmentSource>) -> Self {
        let (requests_send, requests_recv) = mpsc::unbounded();
        let results = SegmentSourceAdapter(source).drive(requests_recv.boxed());

        Self {
            requests_send,
            results,
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
    pub(super) fn request<'a>(
        &mut self,
        segment_ids: impl IntoIterator<Item = &'a SegmentId>,
    ) -> VortexResult<()> {
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
            log::debug!("Requesting segment {}", segment_id);
            self.in_flight.insert(segment_id);
            self.requests_send
                .unbounded_send(segment_id)
                .map_err(|e| vortex_err!("SegmentSource driver lost {e}"))?;
        }

        Ok(())
    }

    pub(super) fn inflight_count(&self) -> usize {
        self.in_flight.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.working_set.is_empty() && self.ref_counts.is_empty()
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
        let Some((segment_id, result)) = ready!(self.results.poll_next_unpin(cx)) else {
            return Poll::Ready(None);
        };

        match result {
            Ok(buffer) => {
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

/// An I/O source for layout segments.
pub trait SegmentSource2: 'static + Send + Sync {
    /// The driver of a segment source accepts a stream of requested SegmentIDs and is polled
    /// by the scan to make progress. Successfully fetched segments are returned on the result
    /// stream in any order.
    ///
    /// Note that an implementation is permitted to return segments that were not explicitly
    /// requested, as may be the case if a coalesced read covers unrequested segments.
    ///
    /// The scan will batch up as many requests as it is able, therefore [`StreamExt::ready_chunks`]
    /// may be helpful if batched requests are useful to the implementation.
    fn drive(
        &self,
        requests: BoxStream<'static, SegmentId>,
    ) -> BoxStream<'static, (SegmentId, VortexResult<ByteBuffer>)>;
}

pub struct SegmentSourceAdapter(pub Arc<dyn SegmentSource>);

impl SegmentSource2 for SegmentSourceAdapter {
    fn drive(
        &self,
        requests: BoxStream<'static, SegmentId>,
    ) -> BoxStream<'static, (SegmentId, VortexResult<ByteBuffer>)> {
        let source = self.0.clone();
        requests
            .map(move |segment_id| {
                let source = source.clone();
                async move { (segment_id, source.request(segment_id).await) }
            })
            .buffer_unordered(128)
            .boxed()
    }
}
