// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! IO requests, results, and the source that fulfils them.
//!
//! The protocol moves bytes at a range. A source has exactly one other property a consumer may
//! ask for, its length, because a Vortex file cannot be located from its start without it.
//! Nothing else is a request: anything a stage needs that is not bytes at a range is
//! construction-time data for that stage.

use vortex_array::buffer::BufferHandle;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

/// Consumer-chosen label for one request, unique within the issuing object.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct IoRequestId(pub u32);

/// What to fetch from the source: its length, or bytes at a range. This enum is closed; see the
/// module documentation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum IoTarget {
    /// Total length of the source in bytes.
    Size,
    /// An absolute byte range.
    Range {
        /// Absolute byte offset of the first byte.
        offset: u64,
        /// Number of bytes to read.
        len: usize,
    },
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
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IoRequest {
    /// Required delivery, background prefetch, or coalescing-only interest.
    pub intent: IoIntent,
    /// Identifies the request within the issuing object.
    pub request: IoRequestId,
    /// What to fetch.
    pub target: IoTarget,
}

/// The whole currently known set of outstanding requests for one object.
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

/// IO registration and dispatch. A source belongs to one registration scope at a time.
/// Services may decline optional hints. Accepted optional work belongs to the service,
/// not to the publishing stage.
pub trait IoSource: Send + Sync {
    /// Registers a whole batch before any of its reads become eligible for dispatch.
    /// Registration does not perform IO. Repeated identities must name the same target.
    fn submit(&self, owner: IoOwnerId, batch: IoBatch) -> VortexResult<()>;

    /// Advances eligible IO without blocking and returns a required completion if ready.
    /// Optional successes stay in the service; any read failure fails the run.
    fn poll(&self) -> VortexResult<Option<Completion>>;

    /// Blocks until a required completion is available or a read fails.
    fn wait(&self) -> VortexResult<Completion>;

    /// Releases a retired object's fetch subscriptions, preserving optional announcements.
    fn release(&self, owner: IoOwnerId);

    /// Cancels the run's registrations and drops retained bytes and pending completions.
    fn clear(&self);
}

/// Receives fulfilled requests. Implementations only store; they never compute here.
pub trait IoConsumer {
    /// Stores the result for `request`.
    ///
    /// The driver only delivers ids the consumer listed and results that match their targets,
    /// so anything else is a driver bug and the consumer may panic. After delivery the id must
    /// not reappear in `state()`.
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
        self.request = Some(request.clone());
        vec![request]
    }

    /// The in-flight request as a batch, while it is undelivered.
    pub fn batch(&self) -> Option<IoBatch> {
        self.request.as_ref().map(|request| vec![request.clone()])
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
