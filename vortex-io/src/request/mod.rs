// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! IO requests, results, and the source that fulfils them.
//!
//! The protocol moves bytes at a range of a byte source. A source has exactly one other property
//! a consumer may ask for, its length, because a Vortex file cannot be located from its start
//! without it. The one request that is not about an existing source opens a new one: a planner
//! that discovers a file mid-plan asks for it to be [opened](IoTarget::Open), and is answered with
//! the [`SourceId`] its later reads name. Anything else a stage needs is construction-time data
//! for that stage.
//!
//! Every request names the source it reads. Ids are minted by a [`SourcesIoService`] when a source
//! is registered or opened, never defaulted, so a plan reads any number of files through one
//! service.

use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

mod open;
mod read_at;
mod source;
mod sources;

pub use open::FileSystemOpenable;
pub use open::Openable;
pub use open::ReadAtOpenable;
pub use read_at::ReadAtIoService;
pub use source::ReadAtIoSource;
pub use sources::SourcesIoService;

/// The byte source a request reads, such as one file of a table.
///
/// Only a [`SourcesIoService`] mints ids, when a source is registered with it or opened through
/// it. There is no default and no public constructor, so no planner can name a source it was not
/// given.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceId(u32);

impl SourceId {
    /// The id's number, unique within the service that minted it. For messages and ordering.
    pub fn index(self) -> u32 {
        self.0
    }
}

impl fmt::Display for SourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "source {}", self.0)
    }
}

/// Consumer-chosen label for one request, unique within the issuing object.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct IoRequestId(pub u32);

/// What to fetch from the source: its length, bytes at a range, or a new source to open. This
/// enum is closed; see the module documentation.
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
        /// The alignment the delivered bytes have at least.
        alignment: Alignment,
    },
    /// Opens a new byte source, answered with [`IoResult::Opened`]. Two opens with the same
    /// [key](Openable::key) are answered with the same id. Addressed to the
    /// [service's own id](SourcesIoService::id), not to a source.
    Open(OpenTarget),
}

/// The thing an [`IoTarget::Open`] opens. Compared and hashed by its [key](Openable::key).
#[derive(Clone)]
pub struct OpenTarget(pub Arc<dyn Openable>);

impl OpenTarget {
    /// A target opening `openable`.
    pub fn new(openable: impl Openable) -> Self {
        Self(Arc::new(openable))
    }
}

impl fmt::Debug for OpenTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Open").field(self.0.key()).finish()
    }
}

impl PartialEq for OpenTarget {
    fn eq(&self, other: &Self) -> bool {
        self.0.key() == other.0.key()
    }
}

impl Eq for OpenTarget {}

impl Hash for OpenTarget {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.key().hash(state);
    }
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

/// One request: the source it reads, a consumer-chosen id, and the target it names.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IoRequest {
    /// The source the request reads, or for an [`IoTarget::Open`], the service that opens it.
    pub source: SourceId,
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
/// `Range`, `Opened` for `Open`. The driver checks this before delivering, so a consumer never sees a mismatch.
#[derive(Debug, Clone)]
pub enum IoResult {
    /// Total length of the source in bytes.
    Size(u64),
    /// The bytes of a range request, possibly not yet resident on the host.
    Bytes(BufferHandle),
    /// The id later requests name to read the source an [`IoTarget::Open`] opened.
    Opened(SourceId),
}

impl IoResult {
    /// Whether this is the variant `target` must be answered with.
    pub fn matches(&self, target: &IoTarget) -> bool {
        matches!(
            (self, target),
            (Self::Size(_), IoTarget::Size)
                | (Self::Bytes(_), IoTarget::Range { .. })
                | (Self::Opened(_), IoTarget::Open(_))
        )
    }

    /// The variant name, for messages.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Size(_) => "size",
            Self::Bytes(_) => "bytes",
            Self::Opened(_) => "opened",
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
pub struct IoSlot {
    source: SourceId,
    next_id: u32,
    request: Option<IoRequest>,
    result: Option<IoResult>,
}

impl IoSlot {
    /// A slot whose requests read `source`.
    pub fn new(source: SourceId) -> Self {
        Self {
            source,
            next_id: 0,
            request: None,
            result: None,
        }
    }

    /// The source this slot's requests read.
    pub fn source(&self) -> SourceId {
        self.source
    }

    /// Issues a request for `target` with a fresh id. Panics if one is already outstanding or
    /// its result has not been taken, since that is a bug in the owner.
    pub fn issue(&mut self, target: IoTarget) -> IoBatch {
        self.issue_to(self.source, target)
    }

    /// Like [`issue`](Self::issue), naming `source` instead of the slot's own, such as the
    /// service that answers an [`IoTarget::Open`].
    pub fn issue_to(&mut self, source: SourceId, target: IoTarget) -> IoBatch {
        if self.request.is_some() || self.result.is_some() {
            vortex_panic!("IoSlot: issue while a request is outstanding");
        }
        let request = IoRequest {
            source,
            intent: IoIntent::Fetch,
            request: IoRequestId(self.next_id),
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
