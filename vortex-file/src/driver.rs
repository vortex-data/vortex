use std::fmt::{Debug, Formatter};
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::stream::BoxStream;
use futures::{Stream, StreamExt};
use itertools::Itertools;
use linked_hash_set::LinkedHashSet;
use vortex_array::aliases::hash_map::HashMap;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_panic};
use vortex_io::{PerformanceHint, VortexReadAt};
use vortex_layout::segments::{SegmentEvent, SegmentId, SegmentRequest};
use vortex_metrics::{Counter, VortexMetrics};

use crate::SegmentSpec;

/// An I/O driver that assembles coalesced requests based on a performance hint and a
/// pre-configured pre-fetching window.
pub struct CoalescedDriver {
    segment_map: Arc<[SegmentSpec]>,
    events: BoxStream<'static, SegmentEvent>,

    requested_counter: Arc<Counter>,
    polled_counter: Arc<Counter>,
    coalesced_counter: Arc<Counter>,
    coalesced_bytes_counter: Arc<Counter>,

    performance_hint: PerformanceHint,
    /// The maximum number of bytes to hold in the prefetch buffer.
    max_prefetch_bytes: i64,

    first_poll: bool,
    state: HashMap<SegmentId, PendingSegment>,
    /// Maintain a set of segments that have been requested, ordered by insertion.
    requested: LinkedHashSet<SegmentId>,
    /// The segments that have been explicitly polled, ordered by insertion.
    polled: LinkedHashSet<SegmentId>,
    /// The number of bytes that have been prefetched but not yet consumed.
    prefetched_bytes: i64,
}

struct PendingSegment {
    /// Whether the segment has been explicitly polled.
    polled: bool,
    /// Whether the segment is counted as part of the prefetch buffer.
    is_prefetched: bool,
    /// The request for the segment, used to resolve the byte buffer.
    request: Option<SegmentRequest>,
}

impl CoalescedDriver {
    pub fn new(
        performance_hint: PerformanceHint,
        segment_map: Arc<[SegmentSpec]>,
        events: BoxStream<'static, SegmentEvent>,
        metrics: VortexMetrics,
    ) -> Self {
        Self {
            segment_map,
            events,

            requested_counter: metrics.counter("vortex.file.segments.requested"),
            polled_counter: metrics.counter("vortex.file.segments.polled"),
            coalesced_counter: metrics.counter("vortex.file.coalesced"),
            coalesced_bytes_counter: metrics.counter("vortex.file.coalesced.bytes"),

            performance_hint,
            max_prefetch_bytes: 32 << 20, // 32 MB

            first_poll: false,
            state: Default::default(),
            requested: Default::default(),
            polled: Default::default(),
            prefetched_bytes: 0,
        }
    }

    fn segment_spec(&self, id: SegmentId) -> &SegmentSpec {
        &self.segment_map[*id as usize]
    }

    fn segment_state(&self, id: SegmentId) -> &PendingSegment {
        self.state.get(&id).vortex_expect("segment does not exist")
    }

    fn segment_state_mut(&mut self, id: SegmentId) -> &mut PendingSegment {
        self.state
            .get_mut(&id)
            .vortex_expect("segment does not exist")
    }

    /// Mark a segment as prefetched (if it hasn't been polled), and update the prefetch
    /// buffer count.
    fn mark_as_prefetched(&mut self, id: SegmentId) {
        let state = self.segment_state(id);

        // If the segment has been pre-fetched, we can remove its bytes from the buffer.
        assert!(!state.is_prefetched, "segment already prefetched");
        if !state.polled {
            self.prefetched_bytes += self.segment_spec(id).length as i64;
            self.segment_state_mut(id).is_prefetched = false;
        }
    }

    /// Unmark a segment as prefetched, and update the prefetch buffer count.
    fn unmark_as_prefetched(&mut self, id: SegmentId) {
        let Some(state) = self.state.get(&id) else {
            return;
        };

        // If the segment has been pre-fetched, we can remove its bytes from the buffer.
        if state.is_prefetched {
            self.prefetched_bytes -= self.segment_spec(id).length as i64;
            self.segment_state_mut(id).is_prefetched = false;
        }
    }

    fn on_requested(&mut self, request: SegmentRequest) {
        self.requested.insert(request.id());
        self.state.insert(
            request.id(),
            PendingSegment {
                polled: false,
                is_prefetched: false,
                request: Some(request),
            },
        );
        self.requested_counter.inc();
    }

    fn on_polled(&mut self, id: SegmentId) {
        // We don't launch _any_ I/O until the first poll.
        self.first_poll = true;

        // If the segment has been pre-fetched, we can remove its bytes from the buffer.
        self.unmark_as_prefetched(id);

        let state = self.segment_state_mut(id);
        state.polled = true;
        self.polled.insert(id);
        self.polled_counter.inc();
    }

    fn on_dropped(&mut self, id: SegmentId) {
        // If the segment has been pre-fetched, we can remove its bytes from the buffer.
        self.unmark_as_prefetched(id);
        self.state.remove(&id);
        self.polled.remove(&id);
        self.requested.remove(&id);
    }

    fn on_resolved(&mut self, _id: SegmentId) {}

    /// Request a segment from the underlying storage.
    fn coalesce_request(&mut self, request: SegmentRequest) -> CoalescedSegmentRequest {
        let spec = self.segment_spec(request.id());

        // We start with the requested segment, and then loop in any additional segments that fall
        // within the coalescing window.
        let mut coalesced = CoalescedSegmentRequest {
            byte_range: spec.byte_range(),
            requests: vec![request],
            segment_map: self.segment_map.clone(),
        };

        // TODO(ngates): dynamically update the coalescing window based on request duration.
        //  We should estimate latency + bandwidth.
        let window = self.performance_hint.coalescing_window();
        let max_read = self.performance_hint.max_read();

        // We keep expanding our coalesced window until we reach max_read or no more segments
        // can be coalesced.
        loop {
            let request_count = coalesced.requests.len();

            // We find the range of segment IDs that intersect the coalescing window. We can do
            // this because segments are ordered by byte offset.
            let lowest_segment = self.segment_map.partition_point(|s| {
                (s.offset + s.length as u64) < coalesced.byte_range.start.saturating_sub(window)
            });
            let highest_segment = self
                .segment_map
                .partition_point(|s| s.offset < coalesced.byte_range.end.saturating_add(window));

            for id in lowest_segment..highest_segment {
                let segment_id = SegmentId::try_from(id).vortex_expect("ID not a u32");

                // Skip the segment if it hasn't been requested at all.
                if !self.state.contains_key(&segment_id) {
                    continue;
                }
                // If the segment has already been requested, it's request will be absent.
                if self.segment_state_mut(segment_id).request.is_none() {
                    continue;
                }

                // Compute the new coalesced range if we were to include this segment.
                let segment_range = self.segment_spec(segment_id).byte_range();
                let new_range = coalesced.byte_range.start.min(segment_range.start)
                    ..coalesced.byte_range.end.max(segment_range.end);

                // If the segment falls within the existing window, we should always include it.
                if segment_range.start >= coalesced.byte_range.start
                    && segment_range.end <= coalesced.byte_range.end
                {
                    coalesced.byte_range = new_range;
                    coalesced.requests.push(
                        self.segment_state_mut(segment_id)
                            .request
                            .take()
                            .vortex_expect("checked above as present"),
                    );
                    continue;
                }

                // If the new range exceeds the max read, we skip the segment.
                if max_read
                    .map(|max_read| new_range.end - new_range.start > max_read)
                    .unwrap_or(false)
                {
                    continue;
                }

                coalesced.byte_range = new_range;
                coalesced.requests.push(
                    self.segment_state_mut(segment_id)
                        .request
                        .take()
                        .vortex_expect("checked above as present"),
                );
            }

            // If we added no new segments, we're done.
            if coalesced.requests.len() == request_count {
                break;
            }
        }

        // Ensure the coalesced requests are sorted
        coalesced.requests.sort_by_key(|r| r.id());

        // Maintain the prefetch buffer count.
        for request in &coalesced.requests {
            self.mark_as_prefetched(request.id());
        }

        log::debug!("Coalesced request: {:?}", coalesced);
        self.coalesced_counter.inc();
        self.coalesced_bytes_counter
            .add(coalesced.size_bytes().try_into().vortex_expect("isize"));
        coalesced
    }
}

impl Stream for CoalescedDriver {
    type Item = CoalescedSegmentRequest;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        // First, we consume any events that have been emitted.
        while let Poll::Ready(event) = this.events.poll_next_unpin(cx) {
            let Some(event) = event else {
                // The event source has shut down, we can end the stream.
                return Poll::Ready(None);
            };

            // Process the event.
            log::debug!("Processing: {:?}", event);
            match event {
                SegmentEvent::Requested(req) => this.on_requested(req),
                SegmentEvent::Polled(id) => this.on_polled(id),
                SegmentEvent::Dropped(id) => this.on_dropped(id),
                SegmentEvent::Resolved(id) => this.on_resolved(id),
            }
        }

        // If no segments have been polled yet, then we hold off on launching I/O.
        if !this.first_poll {
            return Poll::Pending;
        }

        // Now we check to see if we should launch any I/O.
        while let Some(id) = this.polled.pop_front() {
            if let Some(request) = this
                .state
                .get_mut(&id)
                .and_then(|state| state.request.take())
            {
                return Poll::Ready(Some(this.coalesce_request(request)));
            }
        }

        // Only perform pre-fetching if we have spare capacity.
        log::trace!(
            "Used {} / {} prefetched bytes",
            this.prefetched_bytes,
            this.max_prefetch_bytes
        );
        if this.prefetched_bytes < this.max_prefetch_bytes {
            while let Some(id) = this.requested.pop_front() {
                if let Some(request) = this
                    .state
                    .get_mut(&id)
                    .and_then(|state| state.request.take())
                {
                    let coalesced = this.coalesce_request(request);
                    log::debug!("Prefetching: {:?}", coalesced);
                    return Poll::Ready(Some(coalesced));
                }
            }
        }

        // Otherwise, wait for more events.
        Poll::Pending
    }
}

pub struct CoalescedSegmentRequest {
    /// The range of the file to read.
    byte_range: Range<u64>,
    /// The original segment requests, ordered by segment ID.
    requests: Vec<SegmentRequest>,
    /// A copy of the segment map so we can resolve the requests.
    segment_map: Arc<[SegmentSpec]>,
}

impl Debug for CoalescedSegmentRequest {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoalescedSegmentRequest")
            .field("byte_range", &self.byte_range)
            .field("size", &self.size_bytes())
            .field("requests", &self.requests.iter().map(|r| r.id()).join(", "))
            .finish()
    }
}

impl CoalescedSegmentRequest {
    fn size_bytes(&self) -> u64 {
        self.byte_range.end - self.byte_range.start
    }

    /// Resolve the requests with the provided buffer.
    pub fn resolve(self, buffer: VortexResult<ByteBuffer>) {
        let buffer = match buffer {
            Ok(buffer) => {
                // Strip the alignment from the buffer so we can slice it arbitrarily.
                buffer.aligned(Alignment::none())
            }
            Err(e) => {
                // If we fail to read the buffer, we need to resolve all the requests with the error.
                let err = Arc::new(e);
                for request in self.requests {
                    request.resolve(Err(err.clone().into()));
                }
                return;
            }
        };

        if buffer.len() != self.size_bytes() as usize {
            vortex_panic!(
                "Buffer size mismatch: expected {} bytes, got {}",
                self.size_bytes(),
                buffer.len()
            );
        }

        // Split the buffer into segments and resolve the requests.
        for request in self.requests {
            let spec = &self.segment_map[*request.id() as usize];
            let start = usize::try_from(spec.offset - self.byte_range.start)
                .vortex_expect("start too large");
            let stop = start + spec.length as usize;
            request.resolve(Ok(buffer.slice(start..stop).aligned(spec.alignment)))
        }
    }

    /// Launch the request, reading the byte range from the provided reader.
    pub async fn launch<R: VortexReadAt>(self, read: &R) {
        let alignment = self.segment_map[*self.requests[0].id() as usize].alignment;
        let byte_range = self.byte_range.clone();
        let buffer = read
            .read_byte_range(byte_range, alignment)
            .await
            .map_err(VortexError::from);
        self.resolve(buffer)
    }
}
