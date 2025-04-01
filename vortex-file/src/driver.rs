use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::{Stream, StreamExt, stream};
use pin_project_lite::pin_project;
use vortex_array::aliases::hash_map::HashMap;
use vortex_error::{VortexError, VortexExpect, VortexResult};
use vortex_io::{PerformanceHint, VortexReadAt};
use vortex_layout::LayoutWriterExt;
use vortex_layout::segments::{SegmentEvent, SegmentId, SegmentRequest};
use vortex_metrics::VortexMetrics;

use crate::segments::CoalescedSegmentRequest;
use crate::{Footer, SegmentSpec};

pub struct CoalescedDriver {
    performance_hint: PerformanceHint,
    footer: Footer,
    events: BoxStream<'static, SegmentEvent>,
    metrics: VortexMetrics,

    first_poll: bool,
    state: HashMap<SegmentId, PendingSegment>,
    // The segments that have been explicitly polled.
    polled: VecDeque<SegmentId>,
}

struct PendingSegment {
    state: SegmentState,
    request: Option<SegmentRequest>,
}

enum SegmentState {
    Requested,
    Polled,
    Dropped,
    Resolved,
}

impl CoalescedDriver {
    pub fn new(
        performance_hint: PerformanceHint,
        footer: Footer,
        events: BoxStream<'static, SegmentEvent>,
        metrics: VortexMetrics,
    ) -> Self {
        Self {
            performance_hint,
            footer,
            events,
            metrics,

            first_poll: false,
            state: Default::default(),
            polled: Default::default(),
        }
    }

    pub fn into_stream(self) -> impl Stream<Item = CoalescedSegmentRequest> {
        self
    }

    fn segment_spec(&self, id: SegmentId) -> &SegmentSpec {
        &self.footer.segment_map()[*id as usize]
    }

    fn on_requested(&mut self, request: SegmentRequest) {
        self.state.insert(
            request.id(),
            PendingSegment {
                state: SegmentState::Requested,
                request: Some(request),
            },
        );
    }

    fn on_polled(&mut self, id: SegmentId) {
        // We don't launch _any_ I/O until the first poll.
        self.first_poll = true;

        self.state
            .get_mut(&id)
            .vortex_expect("polled segment does not exist")
            .state = SegmentState::Polled;
        self.polled.push_back(id);
    }

    fn on_dropped(&mut self, id: SegmentId) {
        self.state
            .get_mut(&id)
            .vortex_expect("dropped segment does not exist")
            .state = SegmentState::Dropped;
    }

    fn on_resolved(&mut self, id: SegmentId) {
        self.state
            .get_mut(&id)
            .vortex_expect("resolved segment does not exist")
            .state = SegmentState::Resolved;
    }

    /// Request a segment from the underlying storage.
    fn coalesce_request(&mut self, request: SegmentRequest) -> CoalescedSegmentRequest {
        let spec = self.segment_spec(request.id());

        // We start with the requested segment, and then loop in any additional segments that fall
        // within the coalescing window.
        let coalesced = CoalescedSegmentRequest {
            byte_range: spec.byte_range(),
            requests: vec![request],
            segment_map: self.footer.segment_map().clone(),
        };

        // TODO(ngates): coalesce

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
            log::debug!("Processing segment event: {:?}", event);
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

        // Otherwise, wait for more events.
        Poll::Pending
    }
}
