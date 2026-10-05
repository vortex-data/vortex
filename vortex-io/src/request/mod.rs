// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! IO requests, results, and the source that fulfils them.
//!
//! The protocol moves bytes at a range. A source has exactly one other property a consumer may
//! ask for, its length, because a Vortex file cannot be located from its start without it.
//! Nothing else is a request: anything a stage needs that is not bytes at a range is
//! construction-time data for that stage.

use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

mod source;

pub use source::ReadAtIoSource;

/// Consumer-chosen label for one request, unique within the issuing object.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct IoRequestId(pub u32);

/// What to fetch from the source: its length, or bytes at a range. This enum is closed; see the
/// module documentation.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum IoTarget {
    /// Total length of the source in bytes.
    Size,
    /// An absolute byte range.
    Range {
        /// Absolute byte offset of the first byte.
        offset: u64,
        /// Number of bytes to read.
        len: usize,
        /// The alignment the delivered bytes have at least.
        alignment: Alignment,
    },
}

impl IoTarget {
    /// A byte range whose bytes need no particular alignment.
    pub fn range(offset: u64, len: usize) -> Self {
        Self::Range {
            offset,
            len,
            alignment: Alignment::none(),
        }
    }
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

/// One request: a consumer-chosen id and the target it names.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct IoRequest {
    /// Required delivery, background prefetch, or coalescing-only interest.
    pub intent: IoIntent,
    /// Identifies the request within the issuing object.
    pub request: IoRequestId,
    /// What to fetch.
    pub target: IoTarget,
}

/// Requests an object publishes together, each exactly once.
pub type IoBatch = Vec<IoRequest>;

/// Result for one request. The variant matches the target: `Size` for `Size`, `Bytes` for
/// `Range`. The driver checks this before delivering, so a consumer never sees a mismatch.
#[derive(Debug, Clone)]
pub enum IoResult {
    /// Total length of the source in bytes.
    Size(u64),
    /// The bytes of a range request, possibly not yet resident on the host.
    Bytes(BufferHandle),
}

impl IoResult {
    /// Whether this is the variant `target` must be answered with.
    pub fn matches(&self, target: &IoTarget) -> bool {
        matches!(
            (self, target),
            (Self::Size(_), IoTarget::Size) | (Self::Bytes(_), IoTarget::Range { .. })
        )
    }

    /// The variant name, for messages.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Size(_) => "size",
            Self::Bytes(_) => "bytes",
        }
    }
}

/// Opaque owner identity chosen by the caller, unique within a registration scope.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct IoOwnerId(pub u64);

/// One finished request: its owner, its request identity, and the answer.
pub struct Completion {
    /// The owner that submitted the request.
    pub owner: IoOwnerId,
    /// The request's consumer-chosen id.
    pub request: IoRequestId,
    /// The answer, or the read error, which fails the registration scope.
    pub result: VortexResult<IoResult>,
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

/// A byte source's IO service, shared by everything that reads the source.
///
/// Each registration scope reads through its own [`IoSource`], and the service is free to share
/// and coalesce the reads of all of them.
pub trait IoService: Send + Sync {
    /// A fresh source for one registration scope.
    fn session(&self) -> Arc<dyn IoSource>;
}

/// Receives fulfilled requests. Implementations only store; they never compute here.
pub trait IoConsumer {
    /// Stores the result for `request`.
    ///
    /// The driver only delivers fetches the consumer published and results that match their
    /// targets, so anything else is a driver bug and the consumer may panic.
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult);
}

/// One outstanding request at a time, with ids that never repeat within the owner.
///
/// A consumer that issues requests one after another keeps its id allocation, its in-flight
/// request, and the delivered result here, so its own state carries none of them.
#[derive(Default)]
pub struct IoSlot {
    next_id: u32,
    request: Option<IoRequest>,
    result: Option<IoResult>,
}

impl IoSlot {
    /// Issues a request for `target` with a fresh id. Panics if one is already outstanding or
    /// its result has not been taken, since that is a bug in the owner.
    pub fn issue(&mut self, target: IoTarget) -> IoBatch {
        if self.request.is_some() || self.result.is_some() {
            vortex_panic!("IoSlot: issue while a request is outstanding");
        }
        let request = IoRequest {
            intent: IoIntent::Fetch,
            request: IoRequestId(self.next_id),
            target,
        };
        self.next_id += 1;
        self.request = Some(request);
        vec![request]
    }

    /// Whether the issued request is still undelivered.
    pub fn is_waiting(&self) -> bool {
        self.request.is_some()
    }

    /// Stores the result for the in-flight request. Panics on an unknown id or a result that
    /// does not match the target, since the driver guarantees both.
    pub fn deliver(&mut self, id: IoRequestId, result: IoResult) {
        let Some(request) = self.request.take_if(|request| request.request == id) else {
            vortex_panic!("IoSlot: delivery of {id:?}, which is not outstanding");
        };
        if !result.matches(&request.target) {
            vortex_panic!(
                "IoSlot: {id:?} for {:?} answered with {}",
                request.target,
                result.kind()
            );
        }
        self.result = Some(result);
    }

    /// The delivered result, once.
    pub fn take(&mut self) -> Option<IoResult> {
        self.result.take()
    }
}

#[cfg(test)]
mod tests;
