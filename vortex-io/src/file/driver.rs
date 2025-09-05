// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
use std::collections::{BTreeMap, BTreeSet};
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use pin_project_lite::pin_project;
use vortex_error::VortexExpect;

use crate::file::request::{CoalescedRequest, IoRequest};
use crate::file::{ReadEvent, ReadRequest, RequestId};

pin_project! {
    /// A stream that performs coalescing and prioritization of I/O requests.
    ///
    /// Takes an input stream of [`ReadRequest`]s and buffers all ready requests into local state.
    /// When polled for the next request, this stream will choose the next best request based on
    /// an ordering of `(has_been_polled, insertion_order)`, skipping any canceled requests, and
    /// then coalescing with other nearby requests within the configured `window`.
    ///
    /// The output of this stream is expected to be buffered by the desired I/O concurrency, and
    /// driven to completion.
    pub(crate) struct IoRequestStream<S> {
        #[pin]
        events: S,
        inner_done: bool,
        coalesce_window: Option<u64>,
        state: State,
    }
}

impl<S> IoRequestStream<S> {
    // FIXME(ngates): split this into coalesce_distance and max_read_size. We should keep
    //  expanding the request by coalesce_distance, but stop if we hit max_read_size.
    pub(crate) fn new(events: S, coalesce_window: Option<u64>) -> Self
    where
        S: Stream<Item = ReadEvent> + Unpin + Send + 'static,
    {
        IoRequestStream {
            events,
            inner_done: false,
            coalesce_window,
            state: State::default(),
        }
    }
}

impl<S> Stream for IoRequestStream<S>
where
    S: Stream<Item = ReadEvent> + Unpin + Send + 'static,
{
    type Item = IoRequest;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        // First, try to drain all immediately available requests from the inner stream
        loop {
            match this.events.as_mut().poll_next(cx) {
                Poll::Ready(Some(event)) => {
                    this.state.on_event(event);
                }
                Poll::Ready(None) => {
                    *this.inner_done = true;
                    break;
                }
                Poll::Pending => {
                    break;
                }
            }
        }

        // Try to get a coalesced request
        if let Some(coalesced) = this.state.next(*this.coalesce_window) {
            return Poll::Ready(Some(coalesced));
        }

        // If the inner stream is done, and we have no more requests, we're done
        if *this.inner_done && this.state.is_empty() {
            return Poll::Ready(None);
        }

        // Otherwise, we need more data from the inner stream
        Poll::Pending
    }
}

/// The state of the I/O request stream.
#[derive(Default)]
struct State {
    // Maintains the set of pending requests, ordered by insertion.
    requests: BTreeMap<RequestId, ReadRequest>,

    // Maintains a set of polled requests, ordered by insertion.
    // Note that we intentionally choose a (polled, insertion) priority, such that earlier requests
    // still complete first if both an early and late request have been polled. First-polled
    // and most-recently-polled both have issues of priority inversion for our use-case.
    polled_requests: BTreeMap<RequestId, ReadRequest>,

    // Spatial index - allows us to find nearby requests for coalescing sorted by offset.
    requests_by_offset: BTreeSet<(u64, RequestId)>,
}

impl State {
    fn is_empty(&self) -> bool {
        self.requests_by_offset.is_empty()
    }

    fn on_event(&mut self, event: ReadEvent) {
        log::trace!("Received ReadEvent: {:?}", event);
        match event {
            ReadEvent::Request(req) => {
                self.requests_by_offset.insert((req.offset, req.id));
                self.requests.insert(req.id, req);
            }
            ReadEvent::Polled(req_id) => {
                if let Some(req) = self.requests.remove(&req_id) {
                    self.polled_requests.insert(req_id, req);
                }
            }
            ReadEvent::Dropped(req_id) => {
                if let Some(req) = self.requests.remove(&req_id) {
                    self.requests_by_offset.remove(&(req.offset, req_id));
                    log::trace!("ReadRequest dropped before poll: {:?}", req);
                }
                if let Some(req) = self.polled_requests.remove(&req_id) {
                    self.requests_by_offset.remove(&(req.offset, req_id));
                    log::trace!("ReadRequest dropped after poll: {:?}", req);
                }
            }
        }
    }

    /// Get the next request, if any.
    fn next(&mut self, coalesce_distance: Option<u64>) -> Option<IoRequest> {
        match coalesce_distance {
            None => self.next_uncoalesced().map(IoRequest::new_single),
            Some(distance) => self.next_coalesced(distance).map(IoRequest::new_coalesced),
        }
    }

    /// Find the next uncoalesced request, choosing only polled requests.
    fn next_uncoalesced(&mut self) -> Option<ReadRequest> {
        while let Some((req_id, req)) = self.polled_requests.pop_first() {
            self.requests_by_offset.remove(&(req.offset, req_id));
            if req.callback.is_closed() {
                log::trace!("Dropping canceled request");
                continue;
            }
            return Some(req);
        }
        None
    }

    fn next_coalesced(&mut self, coalesce_distance: u64) -> Option<CoalescedRequest> {
        // Find the next valid request in priority order
        let first_req = self.next_uncoalesced()?;

        let mut requests = vec![first_req];
        let mut current_start = requests[0].offset;
        let mut current_end = requests[0].offset + requests[0].length as u64;
        let alignment = requests[0].alignment.clone();

        // Find the range we should scan for coalescing
        let scan_start = current_start.saturating_sub(coalesce_distance);
        let scan_end = current_end.saturating_add(coalesce_distance);

        // Collect requests that can be coalesced (both before and after our mandatory request)
        let mut keys_to_remove = Vec::new();

        for &(req_offset, req_id) in self
            .requests_by_offset
            .range((scan_start, RequestId::MIN)..=(scan_end, RequestId::MAX))
        {
            let req = self
                .polled_requests
                .get(&req_id)
                .or_else(|| self.requests.get(&req_id))
                .vortex_expect("Missing request in requests_by_offset");

            // Skip any cancelled requests
            if req.callback.is_closed() {
                keys_to_remove.push((req_offset, req_id));
                continue;
            }

            // Check if this request is within coalescing distance of our current range
            let req_end = req_offset + req.length as u64;
            if (req_offset <= current_end + coalesce_distance && req_end >= current_start)
                || (req_end + coalesce_distance >= current_start && req_offset <= current_end)
            {
                current_start = current_start.min(req_offset);
                current_end = current_end.max(req_end);

                let req = self
                    .polled_requests
                    .remove(&req_id)
                    .or_else(|| self.requests.remove(&req_id))
                    .vortex_expect("Missing request in requests_by_offset");

                requests.push(req);
                keys_to_remove.push((req_offset, req_id));
            }
        }

        // Remove any dropped requests
        for (req_offset, req_id) in keys_to_remove {
            self.requests_by_offset.remove(&(req_offset, req_id));
            self.polled_requests
                .remove(&req_id)
                .or_else(|| self.requests.remove(&req_id));
        }

        // Sort requests by offset for correct slicing in resolve
        requests.sort_unstable_by_key(|r| r.offset);

        log::debug!(
            "Coalesced {} requests into range {}-{} (len={})",
            requests.len(),
            current_start,
            current_end,
            current_end - current_start,
        );

        Some(CoalescedRequest {
            range: current_start..current_end,
            alignment,
            requests,
        })
    }
}
