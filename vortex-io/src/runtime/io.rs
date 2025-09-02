// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::ReadRequest;
use futures::Stream;
use pin_project_lite::pin_project;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fmt::{Debug, Formatter};
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{VortexError, VortexExpect, VortexResult};

/// An I/O request, either a single read or a coalesced set of reads.
pub enum IoRequest {
    Single(ReadRequest),
    Coalesced(CoalescedRequest),
}

impl IoRequest {
    pub fn offset(&self) -> u64 {
        match self {
            IoRequest::Single(r) => r.offset,
            IoRequest::Coalesced(r) => r.range.start,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            IoRequest::Single(r) => r.length,
            IoRequest::Coalesced(r) => usize::try_from(r.range.end - r.range.start)
                .vortex_expect("range too big for usize"),
        }
    }

    pub fn is_canceled(&self) -> bool {
        match self {
            IoRequest::Single(req) => req.completion.is_canceled(),
            IoRequest::Coalesced(req) => req.requests.iter().all(|r| r.completion.is_canceled()),
        }
    }
}

/// A set of I/O requests that have been coalesced into a single larger request.
pub struct CoalescedRequest {
    pub range: Range<u64>,
    pub alignment: Alignment, // The alignment of the first request in the coalesced range.
    pub requests: Vec<ReadRequest>, // TODO(ngates): we could have enum of Single/Many to avoid Vec.
}

impl Debug for CoalescedRequest {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoalescedRequest")
            .field("#", &self.requests.len())
            .field("length", &(self.range.end - self.range.start))
            .field("range", &self.range)
            .field("alignment", &self.alignment)
            .finish()
    }
}

impl CoalescedRequest {
    pub fn resolve(self, result: VortexResult<ByteBuffer>) {
        match result {
            Ok(buffer) => {
                let buffer = buffer.aligned(Alignment::none());
                for req in self.requests.into_iter() {
                    let start = (req.offset - self.range.start) as usize;
                    let end = start + req.length;
                    let slice = buffer.slice(start..end).aligned(req.alignment);
                    req.completion.complete(Ok(slice));
                }
            }
            Err(e) => {
                let e = Arc::new(e);
                for req in self.requests.into_iter() {
                    req.completion.complete(Err(VortexError::from(e.clone())));
                }
            }
        }
    }
}

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
        inner: S,
        inner_done: bool,
        coalesce_window: Option<u64>,
        state: State,
    }
}

impl<S> IoRequestStream<S> {
    // FIXME(ngates): split this into coalesce_distance and max_read_size. We should keep
    //  expanding the request by coalesce_distance, but stop if we hit max_read_size.
    pub(crate) fn new(inner: S, coalesce_window: Option<u64>) -> Self
    where
        S: Stream<Item = ReadRequest> + Unpin + Send + 'static,
    {
        IoRequestStream {
            inner,
            inner_done: false,
            coalesce_window,
            state: State::default(),
        }
    }
}

impl<S> Stream for IoRequestStream<S>
where
    S: Stream<Item = ReadRequest> + Unpin + Send + 'static,
{
    type Item = IoRequest;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        // First, try to drain all immediately available requests from the inner stream
        loop {
            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(req)) => {
                    this.state.push_req(req);
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
    requests: BTreeMap<ReqId, ReadRequest>,

    // Maintains a set of polled requests, ordered by insertion.
    // Note that we intentionally choose a (polled, insertion) priority, such that earlier requests
    // still complete first if both an early and late request have been polled. First-polled
    // and most-recently-polled both have issues of priority inversion for our use-case.
    polled_requests: BTreeMap<ReqId, ReadRequest>,

    // Spatial index - allows us to find nearby requests for coalescing sorted by offset.
    requests_by_offset: BTreeSet<(u64, ReqId)>,

    // Next request ID to assign
    next_id: ReqId,
}

type ReqId = usize;

impl State {
    fn is_empty(&self) -> bool {
        self.requests_by_offset.is_empty()
    }

    fn push_req(&mut self, req: ReadRequest) {
        let req_id = self.next_id;
        self.next_id += 1;

        self.requests_by_offset.insert((req.offset, req_id));
        if req.completion.is_polled() {
            self.polled_requests.insert(req_id, req);
        } else {
            self.requests.insert(req_id, req);
        }
    }

    /// Get the next request, if any.
    fn next(&mut self, coalesce_distance: Option<u64>) -> Option<IoRequest> {
        // First, we move any polled requests to the polled_requests map.
        let mut to_move = Vec::new();
        for (&req_id, req) in self.requests.iter() {
            if req.completion.is_polled() {
                to_move.push(req_id);
            }
        }
        for req_id in to_move {
            if let Some(req) = self.requests.remove(&req_id) {
                self.polled_requests.insert(req_id, req);
            }
        }

        match coalesce_distance {
            None => self.next_uncoalesced().map(IoRequest::Single),
            Some(distance) => self.next_coalesced(distance).map(IoRequest::Coalesced),
        }
    }

    /// Find the next uncoalesced request, prioritizing polled requests first.
    fn next_uncoalesced(&mut self) -> Option<ReadRequest> {
        for queue in [&mut self.polled_requests, &mut self.requests] {
            // for queue in [&mut self.polled_requests] {
            while let Some((req_id, req)) = queue.pop_first() {
                self.requests_by_offset.remove(&(req.offset, req_id));
                if req.completion.is_canceled() {
                    log::trace!("Dropping canceled request");
                    continue;
                }
                return Some(req);
            }
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
            .range((scan_start, ReqId::MIN)..=(scan_end, ReqId::MAX))
        {
            let req = self
                .polled_requests
                .get(&req_id)
                .or_else(|| self.requests.get(&req_id))
                .vortex_expect("Missing request in requests_by_offset");

            // Skip any cancelled requests
            if req.completion.is_canceled() {
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
