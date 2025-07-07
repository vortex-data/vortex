// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::TryStreamExt as _;
use parking_lot::Mutex;
use vortex_array::stream::SendableArrayStream;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::segments::SegmentId;
use crate::sequence::{SequenceId, SequencePointer};
use crate::{SendableSequentialStream, SequentialStreamAdapter, SequentialStreamExt as _};

pub trait SegmentWriter: Send + Sync {
    /// Write the given data into a segment and associate it with the given segment identifier.
    /// The provided buffers are concatenated together to form the segment.
    ///
    // TODO(ngates): in order to support aligned Direct I/O, it is preferable for all segments to
    //  be aligned to the logical block size (typically 512, but could be 4096). For this reason,
    //  if we know we're going to read an entire FlatLayout together, then we should probably
    //  serialize it into a single segment that is 512 byte aligned? Or else, we should guarantee
    //  to align the the first segment to 512, and then assume that coalescing captures the rest.
    fn put(&mut self, segment_id: SegmentId, buffer: Vec<ByteBuffer>) -> VortexResult<()>;
}

/// Utility struct to associate SequenceId's with
/// chunks in an array stream. It wraps a [SegmentWriter]
/// and enforces SegmentId's sent to it are monotonically
/// increasing.
/// See [SequenceId] docs for more information.
#[derive(Clone)]
pub struct SequenceWriter {
    state: Arc<Mutex<State>>,
}

struct State {
    segment_writer: Box<dyn SegmentWriter>,
    eof_pointer: SequencePointer,
}

impl SequenceWriter {
    pub fn new(segment_writer: Box<dyn SegmentWriter>) -> Self {
        let eof_pointer = SequenceId::root();
        Self {
            state: Arc::new(Mutex::new(State {
                segment_writer,
                eof_pointer,
            })),
        }
    }

    /// Wait until the given SequenceId is the first non dropped
    /// instance among all that self::new_sequential is created.
    /// Calls [SegmentWriter::put] with the resulting SegmentId.
    /// See [SequenceId::collapse] docs for more information.
    pub async fn put(
        &self,
        sequence_id: SequenceId,
        buffer: Vec<ByteBuffer>,
    ) -> VortexResult<SegmentId> {
        let segment_id = sequence_id.collapse().await;
        self.state.lock().segment_writer.put(segment_id, buffer)?;
        Ok(segment_id)
    }

    /// Annotate an array stream with sequence identifiers.
    ///
    /// Each sequence id on an item
    /// in the stream would come after all others that come before it.
    ///
    /// Consecutive calls would guarantee that all sequence id's associated
    /// on the latter stream would come after all that were associated
    /// with the former stream.
    pub fn new_sequential(&self, stream: SendableArrayStream) -> SendableSequentialStream {
        let mut sequence_pointer = self.tail();
        SequentialStreamAdapter::new(
            stream.dtype().clone(),
            stream.map_ok(move |chunk| (sequence_pointer.advance(), chunk)),
        )
        .sendable()
    }

    fn tail(&self) -> SequencePointer {
        let mut guard = self.state.lock();
        let head = guard.eof_pointer.advance();
        head.descend()
    }
}
