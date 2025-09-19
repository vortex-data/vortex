// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
use std::sync::Arc;

use async_stream::try_stream;
use async_trait::async_trait;
use futures::{StreamExt as _, pin_mut};
use vortex_array::arrays::ChunkedArray;
use vortex_array::{Array, ArrayContext, ArrayRef, IntoArray};
use vortex_error::{VortexExpect, VortexResult};
use vortex_io::runtime::Handle;

use crate::segments::SegmentSinkRef;
use crate::sequence::{
    SendableSequentialStream, SequencePointer, SequentialStreamAdapter, SequentialStreamExt,
};
use crate::{LayoutRef, LayoutStrategy};

#[derive(Clone)]
pub struct RepartitionWriterOptions {
    /// The minimum uncompressed size in bytes for a block (coalescing threshold).
    /// Chunks smaller than this will be coalesced with others.
    pub block_size_min_bound: u64,
    /// The maximum uncompressed size in bytes for a block (splitting threshold).
    /// Chunks larger than this will be split using heuristic splitting.
    pub block_size_max_bound: u64,
    /// The multiple of the number of rows in each block.
    pub block_len_multiple: usize,
    pub canonicalize: bool,
}

/// Repartition a stream of arrays into blocks.
///
/// Each emitted block (except the last) is at least `block_size_minimum` bytes and contains a
/// multiple of `block_len_multiple` rows.
#[derive(Clone)]
pub struct RepartitionStrategy {
    child: Arc<dyn LayoutStrategy>,
    options: RepartitionWriterOptions,
}

impl RepartitionStrategy {
    pub fn new<S: LayoutStrategy>(child: S, options: RepartitionWriterOptions) -> Self {
        Self {
            child: Arc::new(child),
            options,
        }
    }
}

#[async_trait]
impl LayoutStrategy for RepartitionStrategy {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        eof: SequencePointer,
        handle: Handle,
    ) -> VortexResult<LayoutRef> {
        // TODO(os): spawn stream below like:
        // canon_stream = stream.map(async {to_canonical}).map(spawn).buffered(parallelism)
        let dtype = stream.dtype().clone();
        let stream = if self.options.canonicalize {
            SequentialStreamAdapter::new(
                dtype.clone(),
                stream.map(|chunk| {
                    let (sequence_id, chunk) = chunk?;
                    VortexResult::Ok((sequence_id, chunk.to_canonical().into_array()))
                }),
            )
            .sendable()
        } else {
            stream
        };

        let dtype_clone = dtype.clone();
        let options = self.options.clone();
        let repartitioned_stream = try_stream! {
            let canonical_stream = stream.peekable();
            pin_mut!(canonical_stream);

            let mut chunks = ChunksBuffer::new(options.clone());
            while let Some(chunk) = canonical_stream.as_mut().next().await {
                let (sequence_id, chunk) = chunk?;
                let mut sequence_pointer = sequence_id.descend();

                // Check if we should split this chunk first
                let chunks_to_process = if chunks.should_split_chunk(&chunk) {
                    chunks.split_chunk_heuristic(chunk)?
                } else {
                    vec![chunk]
                };

                // Process each chunk (original or split)
                for chunk_part in chunks_to_process {
                    let mut offset = 0;
                    while offset < chunk_part.len() {
                        let end = (offset + options.block_len_multiple).min(chunk_part.len());
                        let sliced = chunk_part.slice(offset..end);
                        chunks.push_back(sliced);
                        offset = end;

                        if chunks.have_enough() {
                            let output_chunks = chunks.collect_exact_blocks()?;
                            assert!(!output_chunks.is_empty());
                            let chunked =
                                ChunkedArray::try_new(output_chunks, dtype_clone.clone())?;
                            if !chunked.is_empty() {
                                let canonical = chunked.to_canonical();
                                let compacted = canonical.compact()?;
                                yield (
                                    sequence_pointer.advance(),
                                    compacted.into_array(),
                                )
                            }
                        }
                    }
                }

                if canonical_stream.as_mut().peek().await.is_none() {
                    let to_flush = ChunkedArray::try_new(
                        chunks.data.drain(..).collect(),
                        dtype_clone.clone(),
                    )?;
                    if !to_flush.is_empty() {
                        let canonical = to_flush.to_canonical();
                        let compacted = canonical.compact()?;
                        yield (
                            sequence_pointer.advance(),
                            compacted.into_array(),
                        )
                    }
                }
            }
        };

        self.child
            .write_stream(
                ctx,
                segment_sink,
                SequentialStreamAdapter::new(dtype, repartitioned_stream).sendable(),
                eof,
                handle,
            )
            .await
    }

    fn buffered_bytes(&self) -> u64 {
        // TODO(os): we should probably add the buffered bytes from this strategy on top,
        // it is currently better to not add it at all because these buffered arrays are
        // potentially sliced and uncompressed. They would overestimate the actual bytes
        // that will end up in the file when flushed.
        self.child.buffered_bytes()
    }
}

struct ChunksBuffer {
    data: VecDeque<ArrayRef>,
    row_count: usize,
    nbytes: u64,
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
        self.nbytes >= self.options.block_size_min_bound
            && self.row_count >= self.options.block_len_multiple
    }

    /// Check if a chunk is too large and should be split based on max bound
    fn should_split_chunk(&self, chunk: &ArrayRef) -> bool {
        chunk.nbytes() > self.options.block_size_max_bound
    }

    /// Split a chunk using heuristic approach based on estimated bytes per row
    fn split_chunk_heuristic(&self, chunk: ArrayRef) -> VortexResult<Vec<ArrayRef>> {
        let chunk_nbytes = chunk.nbytes();
        let chunk_len = chunk.len();

        if chunk_len == 0 || chunk_nbytes <= self.options.block_size_max_bound {
            return Ok(vec![chunk]);
        }

        // Estimate bytes per row
        let bytes_per_row = chunk_nbytes / chunk_len as u64;
        if bytes_per_row == 0 {
            return Ok(vec![chunk]); // Avoid division by zero
        }

        // Calculate how many rows should fit in a target size (use the midpoint between min and max)
        let target_size =
            (self.options.block_size_min_bound + self.options.block_size_max_bound) / 2;
        let estimated_rows_per_chunk = match usize::try_from(target_size / bytes_per_row) {
            Ok(rows) if rows > 0 => rows,
            _ => return Ok(vec![chunk]), // Single row is too large or conversion failed
        };

        // Split into chunks of estimated_rows_per_chunk size
        let mut result = Vec::new();
        let mut offset = 0;

        while offset < chunk_len {
            let end = (offset + estimated_rows_per_chunk).min(chunk_len);
            result.push(chunk.slice(offset..end));
            offset = end;
        }

        Ok(result)
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
                let left = chunk.slice(0..remaining);
                let right = chunk.slice(remaining..len);
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
