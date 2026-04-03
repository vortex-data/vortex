// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A recorded variant of a BufferHandle.
//!
//! RecordingBufferHandle will log all operations and store them for later querying.

use std::collections::VecDeque;
use std::ops::Range;

use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;

#[derive(Debug, Clone, Hash)]
pub(super) enum BufferOp {
    /// A slice range. This is in bytes, not elements.
    Slice(Range<usize>),
}

/// A recording variant of the `BufferHandle`.
///
/// All operations are handled lazily, and instead of being logged, we simply record and query the operations.
#[derive(Debug, Clone)]
pub struct RecordingBuffer {
    pub(super) alignment: Alignment,
    /// Length of the original buffer.
    pub(super) original_len: usize,
    /// Current len after all the stored operations are applied over the original.
    pub(super) len: usize,
    /// A vector of operations that are applied in-order over the original buffer.
    pub(super) operations: VecDeque<BufferOp>,
}

impl RecordingBuffer {
    /// Create a new recorded buffer, with no operations applied.
    pub fn new(alignment: Alignment, original_len: usize, len: usize) -> Self {
        RecordingBuffer {
            alignment,
            original_len,
            len,
            operations: VecDeque::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn aligned(mut self, alignment: Alignment) -> Self {
        self.alignment = alignment;
        self
    }

    pub fn alignment(&self) -> Alignment {
        self.alignment
    }

    pub fn slice(&self, range: Range<usize>) -> Self {
        assert!(
            range.start.is_multiple_of(*self.alignment),
            "RecordedBuffer can only be sliced to multiple of its alignment"
        );

        let new_len = range.end - range.start;
        assert!(
            new_len <= self.len,
            "RecordedBuffer cannot be sliced beyond current len"
        );

        let mut operations = self.operations.clone();
        operations.push_back(BufferOp::Slice(range));

        Self {
            len: new_len,
            original_len: self.original_len,
            operations,
            alignment: self.alignment,
        }
    }

    /// Resolve a buffer down into pre- and post-fetch components.
    ///
    /// The only pre-fetch information we need is the final resolved byte range _within_ the
    /// original segment. The post-fetch information is a boxed closure of operations that
    /// is saved off to be applied after the subsegment data has been fetched.
    pub fn resolve(self) -> ResolvedOperations {
        resolve(self.original_len, self.operations)
    }
}

/// The output of resolving a `RecordedBuffer`.
///
/// `RecordedBuffer` implements the `BufferHandle` protocol by saving operations to its memory. Once the optimizer is called,
/// we know exactly how many elements we want to portray here.
pub struct ResolvedOperations {
    pub byte_range: Range<usize>,
    pub callback: Box<dyn FnOnce(ByteBuffer) -> ByteBuffer>,
}

pub(super) fn resolve(
    original_len: usize,
    mut operations: VecDeque<BufferOp>,
) -> ResolvedOperations {
    // Initialize our state with the start and end ranges of the full buffer.
    let mut range_start = 0;
    let mut range_end = original_len;

    // Create a set of operations that must be applied after the fact, once we've materialized
    // the real ByteBuffer.
    let mut post_ops: Vec<Box<dyn FnOnce(ByteBuffer) -> ByteBuffer>> = Vec::new();

    while let Some(op) = operations.pop_front() {
        match op {
            BufferOp::Slice(range) => {
                range_start += range.start;
                range_end = range_start + range.end;
            }
        }
    }

    ResolvedOperations {
        byte_range: range_start..range_end,
        callback: Box::new(move |bytes: ByteBuffer| {
            let mut bytes = bytes;
            for op in post_ops {
                bytes = op(bytes);
            }
            bytes
        }),
    }
}
