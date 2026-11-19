// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! IO requests, results, and the scheduler that fulfils them.
//!
//! A request names a registered source, a segment of that source, and the bytes of the segment it
//! wants. Where a segment's bytes live is construction-time data the scheduler holds, so every
//! read keeps its name from the operator that asks for it to the source that serves it.

use std::ops::Range;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use vortex_array::buffer::BufferHandle;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

mod retain;
mod scheduler;

pub use retain::DEFAULT_RETAINED_BYTES;
pub use scheduler::IoScheduler;
pub use scheduler::PlacedSegments;
pub use scheduler::RegisteredSource;
pub use scheduler::SegmentReader;

pub use crate::segments::SegmentId;

/// Consumer-chosen label for one request, unique within the issuing object.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct IoRequestId(pub u32);

/// A source registered with an [`IoScheduler`], unique among the scheduler's live sources.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SourceId(pub u32);

/// What to fetch from a source: bytes of one of its segments.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IoTarget {
    /// The segment, local to the request's source.
    pub segment: SegmentId,
    /// Which bytes of the segment.
    pub bytes: SegmentBytes,
}

impl IoTarget {
    /// The whole of `segment`.
    pub fn segment(segment: SegmentId) -> Self {
        Self {
            segment,
            bytes: SegmentBytes::All,
        }
    }
}

/// The bytes of a segment a request wants. Either way they are delivered as one buffer at the
/// segment's alignment.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SegmentBytes {
    /// The whole segment.
    All,
    /// Sorted, disjoint ranges, relative to the start of the segment, delivered concatenated in
    /// order with no padding. A consumer that rounds each range out to the segment's alignment
    /// gets every buffer inside it aligned, because a segment's buffers are padded relative to
    /// its start. One buffer per range is one request per range.
    Ranges(Arc<[Range<u64>]>),
}

/// A stage's reason for registering bytes. Optional hints may be declined by the service.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum IoIntent {
    /// Required bytes, delivered to the requesting object.
    Fetch,
    /// Optional bytes eligible for an independent background read.
    Prefetch,
    /// Coalescing interest only; never starts an independent read.
    Announce,
}

/// One request: a consumer-chosen id, the source it reads, and the target it names.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IoRequest {
    /// Required delivery, background prefetch, or coalescing-only interest.
    pub intent: IoIntent,
    /// Identifies the request within the issuing object.
    pub request: IoRequestId,
    /// The source to read.
    pub source: SourceId,
    /// What to fetch.
    pub target: IoTarget,
}

/// Requests an object publishes together, each exactly once.
pub type IoBatch = Vec<IoRequest>;

/// Opaque owner identity chosen by the caller, unique within a registration scope.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct IoOwnerId(pub u64);

/// One finished request: its owner, its request identity, and the answer.
pub struct Completion {
    /// The owner that submitted the request.
    pub owner: IoOwnerId,
    /// The request's consumer-chosen id.
    pub request: IoRequestId,
    /// The requested bytes, possibly not yet resident on the host, or the read error, which
    /// fails the registration scope.
    pub result: VortexResult<BufferHandle>,
}

/// IO registration and dispatch for one registration scope, such as the work descended from one
/// root planner. Services may decline optional hints. Accepted optional work belongs to the
/// source, not to the publishing stage, and lives until the source is cleared or dropped.
pub trait IoSource: Send + Sync {
    /// Registers a whole batch before any of its reads become eligible for dispatch.
    /// Registration does not perform IO and never blocks.
    fn submit(&self, owner: IoOwnerId, batch: IoBatch) -> VortexResult<()>;

    /// Advances eligible IO without blocking and returns a required completion if ready.
    /// Optional successes stay in the source; any read failure fails the run.
    fn poll(&self) -> VortexResult<Option<Completion>>;

    /// Like [`poll`](Self::poll), registering `cx` to be woken when a required completion is
    /// ready. Fails when no fetch is in flight, since nothing would ever wake the caller.
    fn poll_completion(&self, cx: &mut Context<'_>) -> Poll<VortexResult<Completion>>;

    /// Blocks until a required completion is available or a read fails.
    fn wait(&self) -> VortexResult<Completion>;

    /// Releases a retired object's fetch subscriptions, preserving optional announcements.
    fn release(&self, owner: IoOwnerId);

    /// Cancels the scope's registrations and drops retained bytes and pending completions.
    fn clear(&self);
}

/// An IO service, shared by everything that reads its sources.
///
/// Each registration scope reads through its own [`IoSource`], and the service is free to share
/// and coalesce the reads of all of them. [`IoScheduler`] is the default implementation.
pub trait IoService: Send + Sync {
    /// A fresh source for one registration scope.
    fn session(&self) -> Arc<dyn IoSource>;
}

/// Receives fulfilled requests. Implementations only store; they never compute here.
pub trait IoConsumer {
    /// Stores the bytes fetched for `request`.
    ///
    /// The driver only delivers fetches the consumer published, so anything else is a driver
    /// bug and the consumer may panic.
    fn set_io_result(&mut self, request: IoRequestId, result: BufferHandle);
}

/// One outstanding request at a time, with ids that never repeat within the owner.
///
/// A consumer that issues requests one after another keeps its id allocation, its in-flight
/// request, and the delivered result here, so its own state carries none of them.
#[derive(Default)]
pub struct IoSlot {
    next_id: u32,
    request: Option<IoRequest>,
    result: Option<BufferHandle>,
}

impl IoSlot {
    /// Issues a request for `target` of `source` with a fresh id. Panics if one is already
    /// outstanding or its result has not been taken, since that is a bug in the owner.
    pub fn issue(&mut self, source: SourceId, target: IoTarget) -> IoBatch {
        if self.request.is_some() || self.result.is_some() {
            vortex_panic!("IoSlot: issue while a request is outstanding");
        }
        let request = IoRequest {
            intent: IoIntent::Fetch,
            request: IoRequestId(self.next_id),
            source,
            target,
        };
        self.next_id += 1;
        self.request = Some(request.clone());
        vec![request]
    }

    /// Whether the issued request is still undelivered.
    pub fn is_waiting(&self) -> bool {
        self.request.is_some()
    }

    /// Stores the result for the in-flight request. Panics on an unknown id, since the driver
    /// only delivers requests that are outstanding.
    pub fn deliver(&mut self, id: IoRequestId, result: BufferHandle) {
        if self
            .request
            .take_if(|request| request.request == id)
            .is_none()
        {
            vortex_panic!("IoSlot: delivery of {id:?}, which is not outstanding");
        }
        self.result = Some(result);
    }

    /// The delivered result, once.
    pub fn take(&mut self) -> Option<BufferHandle> {
        self.result.take()
    }
}

#[cfg(test)]
mod tests;
