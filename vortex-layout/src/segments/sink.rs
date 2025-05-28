use std::future;
use std::sync::Arc;

use futures::StreamExt as _;
use parking_lot::Mutex;
use vortex_array::stream::SendableArrayStream;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::segments::SegmentId;
use crate::sequence::{SequenceId, SequencePointer};
use crate::{SendableSequentialStream, SequentialStreamAdapter, SequentialStreamExt as _};

pub trait SegmentWriter: Send + Sync {
    /// Write the given data into a segment and return its identifier.
    /// The provided buffers are concatenated together to form the segment.
    ///
    // TODO(ngates): in order to support aligned Direct I/O, it is preferable for all segments to
    //  be aligned to the logical block size (typically 512, but could be 4096). For this reason,
    //  if we know we're going to read an entire FlatLayout together, then we should probably
    //  serialize it into a single segment that is 512 byte aligned? Or else, we should guarantee
    //  to align the the first segment to 512, and then assume that coalescing captures the rest.
    fn put(&self, segment_id: SegmentId, buffer: Vec<ByteBuffer>) -> VortexResult<()>;
}

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

    pub async fn put(
        &self,
        sequence_id: SequenceId,
        buffer: Vec<ByteBuffer>,
    ) -> VortexResult<SegmentId> {
        let segment_id = sequence_id.collapse().await;
        self.state.lock().segment_writer.put(segment_id, buffer)?;
        Ok(segment_id)
    }

    pub fn new_sequential(&self, stream: SendableArrayStream) -> SendableSequentialStream {
        let sequence_pointer = self.tail();
        SequentialStreamAdapter::new(
            stream.dtype().clone(),
            stream.scan(sequence_pointer, |pointer, item| match item {
                Ok(chunk) => future::ready(Some(Ok((pointer.advance(), chunk)))),
                Err(e) => future::ready(Some(Err(e))),
            }),
        )
        .sendable()
    }

    fn tail(&self) -> SequencePointer {
        let mut guard = self.state.lock();
        let head = guard.eof_pointer.advance();
        head.descend()
    }
}
