// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::pin::Pin;
use std::task::Context;
use std::task::Poll;

use futures::Stream;
use pin_project_lite::pin_project;
use tracing::trace;
use vortex_buffer::Alignment;
use vortex_error::VortexExpect;
use vortex_io::CoalesceConfig;
use vortex_utils::aliases::hash_set::HashSet;

use crate::read::ReadRequest;
use crate::read::RequestId;
use crate::read::request::CoalescedRequest;
use crate::read::request::IoRequest;
use crate::segments::ReadEvent;
use crate::segments::RequestMetrics;

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
        coalesce_window: Option<CoalesceConfig>,
        state: State,
    }
}

impl<S> IoRequestStream<S> {
    // FIXME(ngates): split this into coalesce_distance and max_read_size. We should keep
    //  expanding the request by coalesce_distance, but stop if we hit max_read_size.
    pub(crate) fn new(
        events: S,
        coalesce_window: Option<CoalesceConfig>,
        coalesced_buffer_alignment: Alignment,
        metrics: RequestMetrics,
    ) -> Self
    where
        S: Stream<Item = ReadEvent> + Unpin + Send + 'static,
    {
        IoRequestStream {
            events,
            inner_done: false,
            coalesce_window,
            state: State::new(metrics, coalesced_buffer_alignment),
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

    // Metrics for tracking I/O request patterns
    metrics: RequestMetrics,
    coalesced_buffer_alignment: Alignment,
}

impl State {
    fn new(metrics: RequestMetrics, coalesced_buffer_alignment: Alignment) -> Self {
        Self {
            requests: BTreeMap::new(),
            polled_requests: BTreeMap::new(),
            requests_by_offset: BTreeSet::new(),
            metrics,
            coalesced_buffer_alignment,
        }
    }

    #[allow(clippy::cognitive_complexity)]
    fn on_event(&mut self, event: ReadEvent) {
        trace!(?event, "Received ReadEvent");
        match event {
            ReadEvent::Request(req) => {
                if req.callback.is_closed() {
                    trace!(?req, "ReadRequest dropped before registration");
                    return;
                }
                self.requests_by_offset.insert((req.offset, req.id));
                self.requests.insert(req.id, req);
            }
            ReadEvent::Polled(req_id) => {
                if let Some(req) = self.requests.remove(&req_id) {
                    if req.callback.is_closed() {
                        self.requests_by_offset.remove(&(req.offset, req_id));
                        trace!(?req, "ReadRequest dropped before poll");
                    } else {
                        self.polled_requests.insert(req_id, req);
                    }
                }
            }
            ReadEvent::Dropped(req_id) => {
                if let Some(req) = self.requests.remove(&req_id) {
                    self.requests_by_offset.remove(&(req.offset, req_id));
                    trace!(?req, "ReadRequest dropped before poll");
                }
                if let Some(req) = self.polled_requests.remove(&req_id) {
                    self.requests_by_offset.remove(&(req.offset, req_id));
                    trace!(?req, "ReadRequest dropped after poll");
                }
            }
        }
    }

    /// Get the next request, if any.
    fn next(&mut self, coalesce_window: Option<&CoalesceConfig>) -> Option<IoRequest> {
        match coalesce_window {
            None => self.next_uncoalesced().map(|request| {
                self.metrics.individual_requests.add(1);
                IoRequest::new_single(request)
            }),
            Some(window) => self.next_coalesced(window).map(|request| {
                match request.requests().len() {
                    1 => self.metrics.individual_requests.add(1),
                    num_requests => {
                        self.metrics.coalesced_requests.add(1);
                        self.metrics
                            .num_requests_coalesced
                            .update(num_requests as f64);
                    }
                };
                IoRequest::new_coalesced(request)
            }),
        }
    }

    /// Find the next uncoalesced request, choosing only polled requests.
    fn next_uncoalesced(&mut self) -> Option<ReadRequest> {
        while let Some((req_id, req)) = self.polled_requests.pop_first() {
            self.requests_by_offset.remove(&(req.offset, req_id));
            if req.callback.is_closed() {
                trace!("Dropping canceled request");
                continue;
            }
            return Some(req);
        }
        None
    }

    /// Coalesce nearby requests into a single range while aligning the range start down to the
    /// global maximum segment alignment.
    ///
    /// Example (file offsets):
    /// [x, x, x, x, x, x, A, A, A, A, A, x, B]
    /// A aligned to 2, B aligned to 4
    /// Coalesced range starts at 4, so the buffer is:
    /// [x, x, A, A, A, A, A, x, B]
    /// A stays 2-aligned, B stays 4-aligned
    fn next_coalesced(&mut self, window: &CoalesceConfig) -> Option<CoalescedRequest> {
        // Find the next valid request in priority order
        let first_req = self.next_uncoalesced()?;

        let mut requests = vec![first_req];
        let mut current_start = requests[0].offset;
        let mut current_end = requests[0].offset + requests[0].length as u64;
        let align = *self.coalesced_buffer_alignment as u64;

        // Track requests that we've already decided to remove (or that were cancelled) so that
        // we don't repeatedly process them during range scans.
        let mut keys_to_remove: Vec<(u64, RequestId)> = Vec::new();
        let mut ids_to_remove: HashSet<RequestId> = HashSet::new();
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
                if ids_to_remove.contains(&req_id) {
                    continue;
                }

                let req = self
                    .polled_requests
                    .get(&req_id)
                    .or_else(|| self.requests.get(&req_id))
                    .vortex_expect("Missing request in requests_by_offset");

                // Skip any cancelled requests
                if req.callback.is_closed() {
                    if ids_to_remove.insert(req_id) {
                        keys_to_remove.push((req_offset, req_id));
                    }
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
                    let aligned_start = new_start - (new_start % align);
                    let new_total_size = new_end - aligned_start;

                    if new_total_size > window.max_size {
                        // Skip it but keep it available for future coalescing operations.
                        continue;
                    }

                    current_start = new_start;
                    current_end = new_end;
                    let req = self
                        .polled_requests
                        .remove(&req_id)
                        .or_else(|| self.requests.remove(&req_id))
                        .vortex_expect("Missing request in requests_by_offset");

                    requests.push(req);
                    if ids_to_remove.insert(req_id) {
                        keys_to_remove.push((req_offset, req_id));
                    }
                    found_new_requests = true;
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

        let aligned_start = current_start - (current_start % align);

        trace!(
            "Coalesced {} requests into range {}..{} (len={})",
            requests.len(),
            aligned_start,
            current_end,
            current_end - aligned_start,
        );

        Some(
            CoalescedRequest::try_new(
                aligned_start..current_end,
                self.coalesced_buffer_alignment,
                requests,
            )
            .vortex_expect("each request is correctly constructed and range.start <= range.end"),
        )
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;
    use futures::stream;
    use vortex_array::buffer::BufferHandle;
    use vortex_buffer::Alignment;
    use vortex_error::VortexResult;
    use vortex_metrics::DefaultMetricsRegistry;
    use vortex_metrics::MetricValue;
    use vortex_metrics::MetricsRegistry;

    use super::*;
    use crate::read::request::IoRequestInner;

    fn create_request(
        id: usize,
        offset: u64,
        length: usize,
    ) -> (ReadRequest, oneshot::Receiver<VortexResult<BufferHandle>>) {
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
        coalesce_window: Option<CoalesceConfig>,
    ) -> Vec<IoRequest> {
        collect_outputs_with_alignment(events, coalesce_window, Alignment::none()).await
    }

    async fn collect_outputs_with_alignment(
        events: Vec<ReadEvent>,
        coalesce_window: Option<CoalesceConfig>,
        coalesced_buffer_alignment: Alignment,
    ) -> Vec<IoRequest> {
        let event_stream = stream::iter(events);
        let metrics_registry = DefaultMetricsRegistry::default();
        let metrics = RequestMetrics::new(&metrics_registry, vec![]);
        let io_stream = IoRequestStream::new(
            event_stream,
            coalesce_window,
            coalesced_buffer_alignment,
            metrics,
        );
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
            Some(CoalesceConfig {
                distance: 0,
                max_size: 1024,
            }),
        )
        .await;
        assert_eq!(outputs.len(), 1);

        match outputs[0].inner() {
            IoRequestInner::Coalesced(coalesced) => {
                assert_eq!(*coalesced.range(), 0..30);
                assert_eq!(coalesced.requests().len(), 3);
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
            Some(CoalesceConfig {
                distance: 6,
                max_size: 1024,
            }),
        )
        .await;
        assert_eq!(outputs.len(), 1);
        match outputs[0].inner() {
            IoRequestInner::Coalesced(c) => assert_eq!(c.requests().len(), 2),
            _ => panic!("Expected coalesced"),
        }
    }

    #[tokio::test]
    async fn test_coalesce_alignment_adjustment() {
        let (tx1, _rx1) = oneshot::channel();
        let (tx2, _rx2) = oneshot::channel();

        let req1 = ReadRequest {
            id: 1,
            offset: 6,
            length: 5,
            alignment: Alignment::new(2),
            callback: tx1,
        };
        let req2 = ReadRequest {
            id: 2,
            offset: 12,
            length: 1,
            alignment: Alignment::new(4),
            callback: tx2,
        };

        let events = vec![
            ReadEvent::Request(req1),
            ReadEvent::Request(req2),
            ReadEvent::Polled(1),
            ReadEvent::Polled(2),
        ];

        let outputs = collect_outputs_with_alignment(
            events,
            Some(CoalesceConfig {
                distance: 1,
                max_size: 1024,
            }),
            Alignment::new(4),
        )
        .await;

        assert_eq!(outputs.len(), 1);
        match outputs[0].inner() {
            IoRequestInner::Coalesced(coalesced) => {
                assert_eq!(coalesced.range().start, 4);
                assert_eq!(coalesced.alignment(), Alignment::new(4));
                for req in coalesced.requests() {
                    let rel = req.offset - coalesced.range().start;
                    assert_eq!(rel % *req.alignment as u64, 0);
                }
            }
            _ => panic!("Expected coalesced request"),
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

    #[test]
    fn test_on_event_cleans_cancelled_requests_early() {
        let metrics_registry = DefaultMetricsRegistry::default();
        let metrics = RequestMetrics::new(&metrics_registry, vec![]);
        let mut state = State::new(metrics, Alignment::none());

        // Cancel before registration: request should never enter state indexes.
        let (tx1, rx1) = oneshot::channel();
        drop(rx1);
        let req1 = ReadRequest {
            id: 1,
            offset: 10,
            length: 4,
            alignment: Alignment::none(),
            callback: tx1,
        };
        state.on_event(ReadEvent::Request(req1));
        assert!(state.requests.is_empty());
        assert!(state.polled_requests.is_empty());
        assert!(state.requests_by_offset.is_empty());

        // Cancel after registration but before first poll: should be removed on Polled event.
        let (tx2, rx2) = oneshot::channel();
        let req2 = ReadRequest {
            id: 2,
            offset: 20,
            length: 8,
            alignment: Alignment::none(),
            callback: tx2,
        };
        state.on_event(ReadEvent::Request(req2));
        drop(rx2);
        state.on_event(ReadEvent::Polled(2));
        assert!(state.requests.is_empty());
        assert!(state.polled_requests.is_empty());
        assert!(state.requests_by_offset.is_empty());
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
            Some(CoalesceConfig {
                distance: 60,
                max_size: 1024,
            }),
        )
        .await;
        assert_eq!(outputs.len(), 1);

        match outputs[0].inner() {
            IoRequestInner::Coalesced(coalesced) => {
                assert_eq!(*coalesced.range(), 0..110);
                assert_eq!(coalesced.requests().len(), 3);
                // Should be sorted by offset
                assert_eq!(coalesced.requests()[0].offset, 0);
                assert_eq!(coalesced.requests()[1].offset, 50);
                assert_eq!(coalesced.requests()[2].offset, 100);
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
            Some(CoalesceConfig {
                distance: 5,
                max_size: 1024,
            }),
        )
        .await;
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].range(), 0..20);
        assert_eq!(outputs[1].range(), 1000..1010);
    }

    #[tokio::test]
    async fn test_metrics_tracking() {
        let (req1, _rx1) = create_request(1, 0, 10);
        let (req2, _rx2) = create_request(2, 10, 10);
        let (req3, _rx3) = create_request(3, 1000, 10);

        let events = vec![
            // First group - will coalesce (2 requests)
            ReadEvent::Request(req1),
            ReadEvent::Request(req2),
            ReadEvent::Polled(1),
            ReadEvent::Polled(2),
            // Second group - single request, far away
            ReadEvent::Request(req3),
            ReadEvent::Polled(3),
        ];

        let event_stream = stream::iter(events);
        let metrics_registry = DefaultMetricsRegistry::default();
        let metrics = RequestMetrics::new(&metrics_registry, vec![]);
        let io_stream = IoRequestStream::new(
            event_stream,
            Some(CoalesceConfig {
                distance: 5,
                max_size: 1024,
            }),
            Alignment::none(),
            metrics,
        );

        let outputs: Vec<IoRequest> = io_stream.collect().await;
        assert_eq!(outputs.len(), 2);

        let snapshot = metrics_registry.snapshot();
        let mut individual_count = 0u64;
        let mut coalesced_operations = 0u64;
        let mut coalesced_histogram_count = 0usize;

        for metric in snapshot.iter() {
            match metric.value() {
                MetricValue::Counter(counter) => {
                    if metric.name() == "io.requests.individual" {
                        individual_count = counter.value();
                    } else if metric.name() == "io.requests.coalesced" {
                        coalesced_operations = counter.value();
                    }
                }
                MetricValue::Histogram(histogram) => {
                    if metric.name() == "io.requests.coalesced.num_coalesced" {
                        coalesced_histogram_count = histogram.count();
                    }
                }
                _ => {}
            }
        }

        // Should have 1 individual request (req3) and 1 coalesced operation (req1+req2)
        assert_eq!(individual_count, 1, "Expected 1 individual request");
        assert_eq!(coalesced_operations, 1, "Expected 1 coalesced operation");
        assert_eq!(
            coalesced_histogram_count, 1,
            "Expected 1 histogram entry for coalesced count"
        );
    }

    #[tokio::test]
    async fn test_metrics_individual_only() {
        let (req1, _rx1) = create_request(1, 0, 10);
        let (req2, _rx2) = create_request(2, 100, 10);

        let events = vec![
            ReadEvent::Request(req1),
            ReadEvent::Request(req2),
            ReadEvent::Polled(1),
            ReadEvent::Polled(2),
        ];

        let event_stream = stream::iter(events);
        let metrics_registry = DefaultMetricsRegistry::default();
        let metrics = RequestMetrics::new(&metrics_registry, vec![]);
        // No coalescing window - should be individual requests
        let io_stream = IoRequestStream::new(event_stream, None, Alignment::none(), metrics);

        let outputs: Vec<IoRequest> = io_stream.collect().await;
        assert_eq!(outputs.len(), 2);

        // Check metrics
        let snapshot = metrics_registry.snapshot();
        let mut individual_count = 0_u64;
        let mut coalesced_operations = 0_u64;

        for metric in snapshot.iter() {
            if let MetricValue::Counter(counter) = metric.value() {
                if metric.name() == "io.requests.individual" {
                    individual_count = counter.value();
                } else if metric.name() == "io.requests.coalesced.num_coalesced" {
                    coalesced_operations = counter.value();
                }
            }
        }

        // Should have 2 individual requests and no coalesced operations
        assert_eq!(individual_count, 2, "Expected 2 individual requests");
        assert_eq!(coalesced_operations, 0, "Expected 0 coalesced operations");
    }
}
