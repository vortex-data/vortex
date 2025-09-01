// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::ReadRequest;
use futures::Stream;
use pin_project_lite::pin_project;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt::{Debug, Formatter};
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{VortexError, VortexResult};

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
                    req.callback.complete(Ok(slice));
                }
            }
            Err(e) => {
                let e = Arc::new(e);
                for req in self.requests.into_iter() {
                    req.callback.complete(Err(VortexError::from(e.clone())));
                }
            }
        }
    }
}

/// An extension trait for coalescing streams of I/O requests.
pub trait CoalescedStreamExt: Stream<Item = ReadRequest> {
    /// Coalesce nearby requests into a single larger request given a window in bytes.
    fn coalesce(self, window: u64) -> CoalescedStream<Self>
    where
        Self: Sized + Unpin + Send + 'static,
    {
        CoalescedStream {
            inner: Box::new(self),
            window,
            requests: CoalescedRequests::default(),
            inner_done: false,
        }
    }
}

impl<S> CoalescedStreamExt for S where S: Stream<Item = ReadRequest> {}

pin_project! {
    pub struct CoalescedStream<S> {
        #[pin]
        inner: Box<S>,
        window: u64,
        requests: CoalescedRequests,
        inner_done: bool,
    }
}

impl<S> Stream for CoalescedStream<S>
where
    S: Stream<Item = ReadRequest> + Unpin + Send + 'static,
{
    type Item = CoalescedRequest;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        // First, try to drain all immediately available requests from the inner stream
        loop {
            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(req)) => {
                    this.requests.push_req(req);
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
        if let Some(coalesced) = this.requests.next_coalesced(*this.window) {
            return Poll::Ready(Some(coalesced));
        }

        // If the inner stream is done and we have no more requests, we're done
        if *this.inner_done && this.requests.is_empty() {
            return Poll::Ready(None);
        }

        // Otherwise, we need more data from the inner stream
        Poll::Pending
    }
}

/// A utility for coalescing requests
#[derive(Default)]
struct CoalescedRequests {
    // Maintains the order in which we should process requests
    priority_queue: VecDeque<usize>,
    // Spatial index - allows us to find nearby requests for coalescing
    requests_by_offset: BTreeMap<(u64, usize), ReadRequest>,
    // Map request ID to its key in the BTreeMap
    id_to_key: HashMap<usize, (u64, usize)>,
    // Next request ID to assign
    next_id: usize,
}

impl CoalescedRequests {
    fn is_empty(&self) -> bool {
        self.requests_by_offset.is_empty()
    }

    fn push_req(&mut self, req: ReadRequest) {
        let req_id = self.next_id;
        self.next_id += 1;

        let key = (req.offset, req_id);

        // Add to priority queue (FIFO order)
        self.priority_queue.push_back(req_id);

        // Add to spatial index
        self.id_to_key.insert(req_id, key);
        self.requests_by_offset.insert(key, req);
    }

    /// Get the next coalesced request, if any.
    // FIXME(ngates): split this into coalesce_distance and max_read_size. We should keep
    //  expanding the request by coalesce_distance, but stop if we hit max_read_size.
    fn next_coalesced(&mut self, coalesce_distance: u64) -> Option<CoalescedRequest> {
        // Find the next valid request in priority order
        let mut next_valid_key = None;
        while let Some(next_id) = self.priority_queue.pop_front() {
            if let Some(&key) = self.id_to_key.get(&next_id) {
                next_valid_key = Some(key);

                // Skip any cancelled requests
                if let Some(req) = self.requests_by_offset.get(&key) {
                    // Throw away any requests that have been canceled
                    if !req.callback.is_canceled() {
                        break;
                    }
                }
            }
            // Request was already coalesced, continue looking
        }
        let key = next_valid_key?;
        let (start_offset, start_id) = key;

        // Remove the initial request
        let first_req = self
            .requests_by_offset
            .remove(&key)
            .expect("key should exist");
        self.id_to_key.remove(&start_id);

        let mut requests = vec![first_req];
        let mut current_start = requests[0].offset;
        let mut current_end = requests[0].offset + requests[0].length as u64;
        let alignment = requests[0].alignment.clone();

        // Find the range we should scan for coalescing
        let scan_start = start_offset.saturating_sub(coalesce_distance);
        let scan_end = start_offset + requests[0].length as u64 + coalesce_distance;

        // Collect requests that can be coalesced (both before and after our mandatory request)
        let mut keys_to_remove = Vec::new();

        for (&key, req) in self
            .requests_by_offset
            .range((scan_start, 0)..=(scan_end, usize::MAX))
        {
            // Skip any cancelled requests
            if req.callback.is_canceled() {
                keys_to_remove.push(key);
                continue;
            }

            let (req_offset, _req_id) = key;
            let req_end = req_offset + req.length as u64;

            // Check if this request is within coalescing distance of our current range
            if (req_offset <= current_end + coalesce_distance && req_end >= current_start)
                || (req_end + coalesce_distance >= current_start && req_offset <= current_end)
            {
                keys_to_remove.push(key);
                current_start = current_start.min(req_offset);
                current_end = current_end.max(req_end);
            }
        }

        // Remove the coalesced requests
        for key in keys_to_remove {
            let (_, req_id) = key;
            if let Some(req) = self.requests_by_offset.remove(&key) {
                requests.push(req);
                self.id_to_key.remove(&req_id);
                // Remove from priority queue (this is O(n) but queue should be small)
                self.priority_queue.retain(|&id| id != req_id);
            }
        }

        // Sort requests by offset for correct slicing in resolve
        requests.sort_unstable_by_key(|r| r.offset);

        log::debug!(
            "Coalesced {} requests into range {}-{} (len={})",
            requests.len(),
            current_start,
            current_end,
            current_end - current_start
        );

        Some(CoalescedRequest {
            range: current_start..current_end,
            alignment,
            requests,
        })
    }
}
