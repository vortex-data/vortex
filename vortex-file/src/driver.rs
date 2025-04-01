use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::{Stream, StreamExt, stream};
use pin_project_lite::pin_project;
use vortex_array::aliases::hash_map::HashMap;
use vortex_error::{VortexExpect, VortexResult};
use vortex_io::VortexReadAt;
use vortex_layout::segments::{SegmentEvent, SegmentId, SegmentRequest};
use vortex_metrics::VortexMetrics;

use crate::segments::CoalescedSegmentRequest;
use crate::{Footer, SegmentSpec};

pub struct CoalescedDriver<R> {
    read: R,
    footer: Footer,
    events: BoxStream<'static, SegmentEvent>,
    metrics: VortexMetrics,

    first_poll: bool,
    state: HashMap<SegmentId, PendingSegment>,
    // The segments that have been explicitly polled.
    polled: VecDeque<SegmentId>,
}

struct PendingSegment {
    request: SegmentRequest,
    state: SegmentState,
}

enum SegmentState {
    Requested,
    Polled,
    Dropped,
    Resolved,
}

impl<R: VortexReadAt> CoalescedDriver<R> {
    pub fn new(
        read: R,
        footer: Footer,
        events: BoxStream<'static, SegmentEvent>,
        metrics: VortexMetrics,
    ) -> Self {
        Self {
            read,
            footer,
            events,
            metrics,

            first_poll: false,
            state: Default::default(),
            polled: Default::default(),
        }
    }

    pub fn into_stream(self) -> impl Stream<Item = VortexResult<()>> {
        stream::once(async move { Ok(()) })
    }

    fn segment_spec(&self, id: SegmentId) -> &SegmentSpec {
        &self.footer.segment_map()[*id as usize]
    }

    fn on_requested(&mut self, _request: SegmentRequest) {}

    fn on_polled(&mut self, id: SegmentId) {
        // We don't launch _any_ I/O until the first poll.
        self.first_poll = true;

        self.state
            .get_mut(&id)
            .vortex_expect("polled segment does not exist")
            .state = SegmentState::Polled;
        self.polled.push_back(id);
    }

    fn on_dropped(&mut self, _id: SegmentId) {}

    fn on_resolved(&mut self, _id: SegmentId) {}

    fn launch_request(
        &mut self,
        _coalesced: CoalescedSegmentRequest,
    ) -> BoxFuture<'static, VortexResult<()>> {
        todo!()
    }
}

impl<R: VortexReadAt + Unpin> Stream for CoalescedDriver<R> {
    type Item = VortexResult<BoxFuture<'static, VortexResult<()>>>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        // First, we consume any events that have been emitted.
        while let Poll::Ready(event) = this.events.poll_next_unpin(cx) {
            let Some(event) = event else {
                // The event source has shut down, we can end the stream.
                return Poll::Ready(None);
            };

            // Process the event.
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

        todo!()
    }
}
