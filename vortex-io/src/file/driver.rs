// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
use std::collections::{BTreeMap, BTreeSet};
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use pin_project_lite::pin_project;
use vortex_error::VortexExpect;

use crate::file::request::{CoalescedRequest, IoRequest, ReadRequest, RequestId};
use crate::file::{CoalesceWindow, ReadEvent};

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
        coalesce_window: Option<CoalesceWindow>,
        state: State,
    }
}

impl<S> IoRequestStream<S> {
    // FIXME(ngates): split this into coalesce_distance and max_read_size. We should keep
    //  expanding the request by coalesce_distance, but stop if we hit max_read_size.
    pub(crate) fn new(events: S, coalesce_window: Option<CoalesceWindow>) -> Self
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
        if let Some(coalesced) = this.state.next(this.coalesce_window.as_ref()) {
            return Poll::Ready(Some(coalesced));
        }

        // If the inner stream is done, and we have no more _polled_ requests, we're done
        if *this.inner_done && this.state.polled_requests.is_empty() {
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
    fn next(&mut self, coalesce_window: Option<&CoalesceWindow>) -> Option<IoRequest> {
        match coalesce_window {
            None => self.next_uncoalesced().map(IoRequest::new_single),
            Some(window) => self.next_coalesced(window).map(IoRequest::new_coalesced),
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

    fn next_coalesced(&mut self, window: &CoalesceWindow) -> Option<CoalescedRequest> {
        // Find the next valid request in priority order
        let first_req = self.next_uncoalesced()?;

        let mut requests = vec![first_req];
        let mut current_start = requests[0].offset;
        let mut current_end = requests[0].offset + requests[0].length as u64;
        let alignment = requests[0].alignment;

        let mut keys_to_remove = Vec::new();
        let mut found_new_requests = true;

        // Keep expanding the window while we can find new requests within constraints
        while found_new_requests {
            found_new_requests = false;

            // Find the range we should scan for coalescing in this iteration
            let scan_start = current_start.saturating_sub(window.distance);
            let scan_end = current_end.saturating_add(window.distance);

            // Look for requests that can be coalesced with our current range
            for &(req_offset, req_id) in self
                .requests_by_offset
                .range((scan_start, RequestId::MIN)..=(scan_end, RequestId::MAX))
            {
                // Skip if we've already marked this request for removal
                if keys_to_remove.iter().any(|&(_, id)| id == req_id) {
                    continue;
                }

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
                if (req_offset <= current_end + window.distance && req_end >= current_start)
                    || (req_end + window.distance >= current_start && req_offset <= current_end)
                {
                    // Calculate what the new range would be if we include this request
                    let new_start = current_start.min(req_offset);
                    let new_end = current_end.max(req_end);
                    let new_total_size = new_end - new_start;

                    // Check if the coalesced request would exceed max_size
                    if new_total_size <= window.max_size {
                        current_start = new_start;
                        current_end = new_end;

                        let req = self
                            .polled_requests
                            .remove(&req_id)
                            .or_else(|| self.requests.remove(&req_id))
                            .vortex_expect("Missing request in requests_by_offset");

                        requests.push(req);
                        keys_to_remove.push((req_offset, req_id));
                        found_new_requests = true;
                    }
                    // If adding this request would exceed max_size, we skip it but don't remove it
                    // so it can be considered for future coalescing operations
                }
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

#[cfg(test)]
mod tests {
    use futures::{StreamExt, stream};
    use vortex_buffer::{Alignment, ByteBuffer};
    use vortex_error::VortexResult;

    use super::*;
    use crate::file::request::ReadRequest;
    use crate::file::{IoRequestInner, ReadEvent};

    fn create_request(
        id: usize,
        offset: u64,
        length: usize,
    ) -> (ReadRequest, oneshot::Receiver<VortexResult<ByteBuffer>>) {
        let (tx, rx) = oneshot::channel();
        (
            ReadRequest {
                id,
                offset,
                length,
                alignment: Alignment::none(),
                callback: tx,
            },
            rx,
        )
    }

    async fn collect_outputs(
        events: Vec<ReadEvent>,
        coalesce_window: Option<CoalesceWindow>,
    ) -> Vec<IoRequest> {
        let event_stream = stream::iter(events);
        let io_stream = IoRequestStream::new(event_stream, coalesce_window);
        io_stream.collect().await
    }

    #[tokio::test]
    async fn test_single_request() {
        let (req, _rx) = create_request(1, 100, 50);
        let events = vec![ReadEvent::Request(req), ReadEvent::Polled(1)];

        let outputs = collect_outputs(events, None).await;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].offset(), 100);
        assert_eq!(outputs[0].len(), 50);
    }

    #[tokio::test]
    async fn test_poll_order_priority() {
        let (req1, _rx1) = create_request(1, 0, 10);
        let (req2, _rx2) = create_request(2, 100, 10);
        let (req3, _rx3) = create_request(3, 200, 10);

        let events = vec![
            // Insert in different order
            ReadEvent::Request(req2),
            ReadEvent::Request(req1),
            ReadEvent::Request(req3),
            // Poll in regular order
            ReadEvent::Polled(1),
            ReadEvent::Polled(2),
            ReadEvent::Polled(3),
        ];

        let outputs = collect_outputs(events, None).await;
        assert_eq!(outputs.len(), 3);

        // Should be in insertion order, not poll order!
        let offsets: Vec<u64> = outputs.iter().map(|req| req.offset()).collect();
        assert_eq!(offsets, vec![0, 100, 200]); // req1, req2, req3
    }

    #[tokio::test]
    async fn test_coalesce_adjacent() {
        let (req1, _rx1) = create_request(1, 0, 10);
        let (req2, _rx2) = create_request(2, 10, 10);
        let (req3, _rx3) = create_request(3, 20, 10);

        let events = vec![
            ReadEvent::Request(req1),
            ReadEvent::Request(req2),
            ReadEvent::Request(req3),
            ReadEvent::Polled(1),
            ReadEvent::Polled(2),
            ReadEvent::Polled(3),
        ];

        let outputs = collect_outputs(
            events,
            Some(CoalesceWindow {
                distance: 0,
                max_size: 1024,
            }),
        )
        .await;
        assert_eq!(outputs.len(), 1);

        match outputs[0].inner() {
            IoRequestInner::Coalesced(coalesced) => {
                assert_eq!(coalesced.range, 0..30);
                assert_eq!(coalesced.requests.len(), 3);
            }
            _ => panic!("Expected coalesced request"),
        }
    }

    #[tokio::test]
    async fn test_coalesce_with_gap() {
        let (req1, _rx1) = create_request(1, 0, 10);
        let (req2, _rx2) = create_request(2, 15, 10); // 5 byte gap

        let events = vec![
            ReadEvent::Request(req1),
            ReadEvent::Request(req2),
            ReadEvent::Polled(1),
        ];

        // Gap is 5, window is 6 - should coalesce
        let outputs = collect_outputs(
            events,
            Some(CoalesceWindow {
                distance: 6,
                max_size: 1024,
            }),
        )
        .await;
        assert_eq!(outputs.len(), 1);
        match outputs[0].inner() {
            IoRequestInner::Coalesced(c) => assert_eq!(c.requests.len(), 2),
            _ => panic!("Expected coalesced"),
        }
    }

    #[tokio::test]
    async fn test_dropped_requests() {
        let (req1, _rx1) = create_request(1, 0, 10);
        let (req2, _rx2) = create_request(2, 100, 10);
        let (req3, _rx3) = create_request(3, 200, 10);

        let events = vec![
            ReadEvent::Request(req1),
            ReadEvent::Request(req2),
            ReadEvent::Request(req3),
            ReadEvent::Dropped(1), // Drop before poll
            ReadEvent::Polled(2),
            ReadEvent::Polled(3),
            ReadEvent::Dropped(3), // Drop after poll
        ];

        let outputs = collect_outputs(events, None).await;
        assert_eq!(outputs.len(), 1); // Only req2 should remain
        assert_eq!(outputs[0].offset(), 100);
    }

    #[tokio::test]
    async fn test_cancelled_requests() {
        let (tx1, rx1) = oneshot::channel();
        let (tx2, _rx2) = oneshot::channel();

        // Drop rx1 to cancel request 1
        drop(rx1);

        let req1 = ReadRequest {
            id: 1,
            offset: 0,
            length: 10,
            alignment: Alignment::none(),
            callback: tx1,
        };
        let req2 = ReadRequest {
            id: 2,
            offset: 100,
            length: 10,
            alignment: Alignment::none(),
            callback: tx2,
        };

        let events = vec![
            ReadEvent::Request(req1),
            ReadEvent::Request(req2),
            ReadEvent::Polled(1),
            ReadEvent::Polled(2),
        ];

        let outputs = collect_outputs(events, None).await;
        assert_eq!(outputs.len(), 1); // Only req2, req1 was cancelled
        assert_eq!(outputs[0].offset(), 100);
    }

    #[tokio::test]
    async fn test_unpolled_requests_ignored() {
        let (req1, _rx1) = create_request(1, 0, 10);
        let (req2, _rx2) = create_request(2, 100, 10);

        let events = vec![
            ReadEvent::Request(req1),
            ReadEvent::Request(req2),
            // No Polled events
        ];

        let outputs = collect_outputs(events, None).await;
        assert_eq!(outputs.len(), 0);
    }

    #[tokio::test]
    async fn test_coalesce_expansion_around_polled() {
        let (req1, _rx1) = create_request(1, 0, 10);
        let (req2, _rx2) = create_request(2, 50, 10); // This one polled first
        let (req3, _rx3) = create_request(3, 100, 10);

        let events = vec![
            ReadEvent::Request(req1),
            ReadEvent::Request(req2),
            ReadEvent::Request(req3),
            ReadEvent::Polled(2), // Poll middle request
        ];

        let outputs = collect_outputs(
            events,
            Some(CoalesceWindow {
                distance: 60,
                max_size: 1024,
            }),
        )
        .await;
        assert_eq!(outputs.len(), 1);

        match outputs[0].inner() {
            IoRequestInner::Coalesced(coalesced) => {
                assert_eq!(coalesced.range, 0..110);
                assert_eq!(coalesced.requests.len(), 3);
                // Should be sorted by offset
                assert_eq!(coalesced.requests[0].offset, 0);
                assert_eq!(coalesced.requests[1].offset, 50);
                assert_eq!(coalesced.requests[2].offset, 100);
            }
            _ => panic!("Expected coalesced request"),
        }
    }

    #[tokio::test]
    async fn test_empty_stream() {
        let outputs = collect_outputs(vec![], None).await;
        assert_eq!(outputs.len(), 0);
    }

    #[tokio::test]
    async fn test_mixed_coalesced_and_single() {
        let (req1, _rx1) = create_request(1, 0, 10);
        let (req2, _rx2) = create_request(2, 10, 10);
        let (req3, _rx3) = create_request(3, 1000, 10);

        let events = vec![
            // First group - will coalesce
            ReadEvent::Request(req1),
            ReadEvent::Request(req2),
            ReadEvent::Polled(1),
            // Second group - single request, far away
            ReadEvent::Request(req3),
            ReadEvent::Polled(3),
        ];

        let outputs = collect_outputs(
            events,
            Some(CoalesceWindow {
                distance: 5,
                max_size: 1024,
            }),
        )
        .await;
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].range(), 0..20);
        assert_eq!(outputs[1].range(), 1000..1010);
    }
}
