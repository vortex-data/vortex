// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! IO requests, results, and the source that fulfils them.
//!
//! The protocol moves bytes at a range. A source has exactly one other property a consumer may
//! ask for, its length, because a Vortex file cannot be located from its start without it.
//! Nothing else is a request: anything a stage needs that is not bytes at a range is
//! construction-time data for that stage.

use std::sync::Arc;

use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_io::VortexReadAt;
use vortex_io::runtime::BlockingRuntime;

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

/// One request: a consumer-chosen id and the target it names.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IoRequest {
    /// Identifies the request within the issuing object.
    pub request: IoRequestId,
    /// What to fetch.
    pub target: IoTarget,
}

/// The whole currently known set of outstanding requests for one object.
pub type IoBatch = Vec<IoRequest>;

/// Result for one request. The variant matches the target: `Size` for `Size`, `Bytes` for
/// `Range`. The driver checks this before delivering, so a consumer never sees a mismatch.
#[derive(Debug)]
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

/// Fulfils one target. Blocking in this prototype.
pub trait IoSource: Send + Sync {
    /// Performs one read against the source and returns its result.
    ///
    /// Errors propagate unchanged to the caller of the driver.
    fn perform(&self, target: &IoTarget) -> VortexResult<IoResult>;
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

/// An [`IoSource`] over a [`VortexReadAt`], blocking on the given runtime.
pub struct ReadAtIoSource<R: BlockingRuntime> {
    read: Arc<dyn VortexReadAt>,
    runtime: Arc<R>,
}

impl<R: BlockingRuntime> ReadAtIoSource<R> {
    /// Creates a source that answers every target by blocking on `runtime`.
    pub fn new(read: Arc<dyn VortexReadAt>, runtime: Arc<R>) -> Self {
        Self { read, runtime }
    }
}

impl<R: BlockingRuntime + Send + Sync> IoSource for ReadAtIoSource<R> {
    fn perform(&self, target: &IoTarget) -> VortexResult<IoResult> {
        match *target {
            IoTarget::Size => Ok(IoResult::Size(self.runtime.block_on(self.read.size())?)),
            IoTarget::Range { offset, len } => Ok(IoResult::Bytes(
                self.runtime
                    .block_on(self.read.read_at(offset, len, Alignment::none()))?,
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_utils::aliases::hash_map::HashMap;

    use super::*;

    #[test]
    fn request_id_is_a_map_key() {
        let mut pending: HashMap<IoRequestId, IoTarget> = HashMap::default();
        pending.insert(IoRequestId(0), IoTarget::Size);
        pending.insert(IoRequestId(1), IoTarget::Range { offset: 8, len: 16 });
        assert_eq!(pending.remove(&IoRequestId(0)), Some(IoTarget::Size));
        assert!(pending.contains_key(&IoRequestId(1)));
        assert!(!pending.contains_key(&IoRequestId(0)));
    }

    #[test]
    fn batches_compare_structurally() {
        let batch: IoBatch = vec![IoRequest {
            request: IoRequestId(3),
            target: IoTarget::Range { offset: 0, len: 4 },
        }];
        assert_eq!(batch, batch.clone());
        assert_ne!(batch[0].target, IoTarget::Range { offset: 0, len: 5 });
        assert_eq!(format!("{:?}", IoRequestId(3)), "IoRequestId(3)");
    }

    #[test]
    fn slot_allocates_fresh_ids_and_hands_back_one_result() {
        let mut slot = IoSlot::default();
        assert!(slot.batch().is_none());
        let batch = slot.issue(IoTarget::Size);
        assert_eq!(batch[0].request, IoRequestId(0));
        assert_eq!(slot.batch(), Some(batch));
        slot.deliver(IoRequestId(0), IoResult::Size(9));
        assert!(slot.batch().is_none());
        assert!(matches!(slot.take(), Some(IoResult::Size(9))));
        assert!(slot.take().is_none());
        let batch = slot.issue(IoTarget::Range { offset: 0, len: 1 });
        assert_eq!(batch[0].request, IoRequestId(1));
    }

    #[test]
    #[should_panic(expected = "not outstanding")]
    fn slot_rejects_unknown_ids() {
        let mut slot = IoSlot::default();
        slot.issue(IoTarget::Size);
        slot.deliver(IoRequestId(7), IoResult::Size(1));
    }

    #[test]
    #[should_panic(expected = "answered with bytes")]
    fn slot_rejects_mismatched_variants() {
        let mut slot = IoSlot::default();
        slot.issue(IoTarget::Size);
        slot.deliver(
            IoRequestId(0),
            IoResult::Bytes(BufferHandle::new_host(vortex_buffer::ByteBuffer::empty())),
        );
    }
}
