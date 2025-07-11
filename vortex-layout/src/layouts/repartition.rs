// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;

use arcref::ArcRef;
use async_stream::try_stream;
use futures::{StreamExt as _, pin_mut};
use vortex_array::arrays::ChunkedArray;
use vortex_array::{Array, ArrayContext, ArrayRef, IntoArray};
use vortex_error::{VortexExpect, VortexResult};

use crate::segments::SequenceWriter;
use crate::{
    LayoutStrategy, SendableLayoutFuture, SendableSequentialStream, SequentialStreamAdapter,
    SequentialStreamExt,
};

#[derive(Clone)]
pub struct RepartitionWriterOptions {
    /// The minimum uncompressed size in bytes for a block.
    pub block_size_minimum: usize,
    /// The multiple of the number of rows in each block.
    pub block_len_multiple: usize,
}

/// Repartition a stream of arrays into blocks.
///
/// Each emitted block (except the last) is at least `block_size_minimum` bytes and contains a
/// multiple of `block_len_multiple` rows.
pub struct RepartitionStrategy {
    options: RepartitionWriterOptions,
    child: ArcRef<dyn LayoutStrategy>,
}

impl RepartitionStrategy {
    pub fn new(child: ArcRef<dyn LayoutStrategy>, options: RepartitionWriterOptions) -> Self {
        Self { options, child }
    }
}

impl LayoutStrategy for RepartitionStrategy {
    fn write_stream(
        &self,
        ctx: &ArrayContext,
        sequence_writer: SequenceWriter,
        stream: SendableSequentialStream,
    ) -> SendableLayoutFuture {
        // TODO(os): spawn stream below like:
        // canon_stream = stream.map(async {to_canonical}).map(spawn).buffered(parallelism)
        let dtype = stream.dtype().clone();
        let canonical_stream = SequentialStreamAdapter::new(
            dtype.clone(),
            stream.map(|chunk| {
                let (sequence_id, chunk) = chunk?;
                VortexResult::Ok((sequence_id, chunk.to_canonical()?.into_array()))
            }),
        )
        .sendable();

        let dtype_clone = dtype.clone();
        let options = self.options.clone();
        let repartitioned_stream = try_stream! {
            let canonical_stream = canonical_stream.peekable();
            pin_mut!(canonical_stream);
            let mut chunks = ChunksBuffer::new(options.clone());
            while let Some(chunk) = canonical_stream.as_mut().next().await {
                let (sequence_id, chunk) = chunk?;
                let mut sequence_pointer = sequence_id.descend();
                let mut offset = 0;
                while offset < chunk.len() {
                    let end = (offset + options.block_len_multiple).min(chunk.len());
                    let sliced = chunk.slice(offset, end)?;
                    chunks.push_back(sliced);
                    offset = end;

                    if chunks.have_enough() {
                        let output_chunks = chunks.collect_exact_blocks()?;
                        assert!(!output_chunks.is_empty());
                        let chunked =
                            ChunkedArray::new_unchecked(output_chunks, dtype_clone.clone());
                        if !chunked.is_empty() {
                            yield (
                                sequence_pointer.advance(),
                                chunked.to_canonical()?.into_array(),
                            )
                        }
                    }
                }
                if canonical_stream.as_mut().peek().await.is_none() {
                    let to_flush = ChunkedArray::new_unchecked(
                        chunks.data.drain(..).collect(),
                        dtype_clone.clone(),
                    );
                    if !to_flush.is_empty() {
                        yield (
                            sequence_pointer.advance(),
                            to_flush.to_canonical()?.into_array(),
                        )
                    }
                }
            }
        };

        self.child.write_stream(
            ctx,
            sequence_writer,
            SequentialStreamAdapter::new(dtype, repartitioned_stream).sendable(),
        )
    }
}

struct ChunksBuffer {
    data: VecDeque<ArrayRef>,
    row_count: usize,
    nbytes: usize,
    options: RepartitionWriterOptions,
}

impl ChunksBuffer {
    fn new(options: RepartitionWriterOptions) -> Self {
        Self {
            data: Default::default(),
            row_count: 0,
            nbytes: 0,
            options,
        }
    }

    fn have_enough(&self) -> bool {
        self.nbytes >= self.options.block_size_minimum
            && self.row_count >= self.options.block_len_multiple
    }

    fn collect_exact_blocks(&mut self) -> VortexResult<Vec<ArrayRef>> {
        let nblocks = self.row_count / self.options.block_len_multiple;
        let mut res = Vec::with_capacity(self.data.len());
        let mut remaining = nblocks * self.options.block_len_multiple;
        while remaining > 0 {
            let chunk = self
                .pop_front()
                .vortex_expect("must have at least one chunk");
            let len = chunk.len();

            if len > remaining {
                let left = chunk.slice(0, remaining)?;
                let right = chunk.slice(remaining, len)?;
                self.push_front(right);
                res.push(left);
                remaining = 0;
            } else {
                res.push(chunk);
                remaining -= len;
            }
        }
        Ok(res)
    }

    fn push_back(&mut self, chunk: ArrayRef) {
        self.row_count += chunk.len();
        self.nbytes += chunk.nbytes();
        self.data.push_back(chunk);
    }

    fn push_front(&mut self, chunk: ArrayRef) {
        self.row_count += chunk.len();
        self.nbytes += chunk.nbytes();
        self.data.push_front(chunk);
    }

    fn pop_front(&mut self) -> Option<ArrayRef> {
        let res = self.data.pop_front();
        if let Some(chunk) = res.as_ref() {
            self.row_count -= chunk.len();
            self.nbytes -= chunk.nbytes();
        }
        res
    }
}
