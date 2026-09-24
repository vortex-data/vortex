// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! IO requests, results, and the source that fulfils them.
//!
//! Prototype deviation from `TRAITS.md`: requests are keyed by [`IoTarget`] alone, with a
//! `Size` target for source-length discovery, and results arrive as [`IoResult`] rather than a
//! bare buffer handle so a size answer can travel the same path as bytes.

use std::sync::Arc;

use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;
use vortex_io::VortexReadAt;
use vortex_io::runtime::BlockingRuntime;

/// Consumer-chosen label for one request, unique within the issuing object.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct IoRequestId(pub u32);

/// What to fetch from the source.
///
/// Prototype deviation: `TRAITS.md` keys requests by segment source and segment id; this slice
/// only needs sizes and byte ranges.
#[derive(Clone, Debug, PartialEq, Eq)]
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IoRequest {
    /// Identifies the request within the issuing object.
    pub request: IoRequestId,
    /// What to fetch.
    pub target: IoTarget,
}

/// The whole currently known set of outstanding requests for one object.
pub type IoBatch = Vec<IoRequest>;

/// Result for one request. The variant matches the target: `Size` for `Size`, `Bytes` for
/// `Range`.
#[derive(Debug)]
pub enum IoResult {
    /// Total length of the source in bytes.
    Size(u64),
    /// The bytes of a range request, possibly not yet resident on the host.
    Bytes(BufferHandle),
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
    /// An unknown id is a protocol error the consumer surfaces from its next `compute()`, as is
    /// a result whose variant does not match the request's target. After delivery the id must
    /// not reappear in `state()`.
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult);
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
}
