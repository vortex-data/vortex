// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
use std::sync::Arc;

use async_stream::try_stream;
use async_trait::async_trait;
use futures::{StreamExt as _, pin_mut};
use vortex_array::arrays::ChunkedArray;
use vortex_array::{Array, ArrayContext, ArrayRef, IntoArray, ToCanonical};
use vortex_error::{VortexExpect, VortexResult};
use vortex_io::runtime::Handle;

use crate::segments::SegmentSinkRef;
use crate::sequence::{
    SendableSequentialStream, SequencePointer, SequentialStreamAdapter, SequentialStreamExt,
};
use crate::{LayoutRef, LayoutStrategy};

#[derive(Clone)]
pub struct RepartitionWriterOptions {
    /// The minimum uncompressed size in bytes for a block.
    pub block_size_minimum: u64,
    /// The maximum uncompressed size in bytes for a block (soft limit, respects block_len_multiple).
    pub block_size_maximum: Option<u64>,
    /// The multiple of the number of rows in each block.
    pub block_len_multiple: usize,
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
        let canonical_stream = SequentialStreamAdapter::new(
            dtype.clone(),
            stream.map(|chunk| {
                let (sequence_id, chunk) = chunk?;
                VortexResult::Ok((sequence_id, chunk.to_canonical().into_array()))
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
                    let sliced = chunk.slice(offset..end);
                    chunks.push_back(sliced);
                    offset = end;

                    if chunks.have_enough() || chunks.is_oversized() {
                        let output_chunks = chunks.collect_exact_blocks()?;
                        assert!(!output_chunks.is_empty());
                        let chunked =
                            ChunkedArray::try_new(output_chunks, dtype_clone.clone())?;
                        if !chunked.is_empty() {
                            yield (
                                sequence_pointer.advance(),
                                chunked.to_canonical().into_array(),
                            )
                        }
                    }
                }
                if canonical_stream.as_mut().peek().await.is_none() {
                    let to_flush = ChunkedArray::try_new(
                        chunks.data.drain(..).collect(),
                        dtype_clone.clone(),
                    )?;
                    if !to_flush.is_empty() {
                        yield (
                            sequence_pointer.advance(),
                            to_flush.to_canonical().into_array(),
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
        self.nbytes >= self.options.block_size_minimum
            && self.row_count >= self.options.block_len_multiple
    }

    fn is_oversized(&self) -> bool {
        if let Some(max_size) = self.options.block_size_maximum {
            self.nbytes > max_size && self.row_count >= self.options.block_len_multiple
        } else {
            false
        }
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
                // Check if this chunk is oversized and needs subdivision
                if let Some(max_size) = self.options.block_size_maximum {
                    let chunk_size = chunk.to_canonical().compacted_size();
                    if chunk_size > max_size && len >= self.options.block_len_multiple {
                        // Use size-aware subdivision that respects zone boundaries
                        let subdivided = self.subdivide_chunk_size_aware(chunk, max_size)?;
                        for subchunk in subdivided {
                            let subchunk_len = subchunk.len();
                            if subchunk_len <= remaining {
                                res.push(subchunk);
                                remaining -= subchunk_len;
                            } else {
                                // This subchunk is too big for the remaining space
                                let left = subchunk.slice(0..remaining);
                                let right = subchunk.slice(remaining..subchunk_len);
                                self.push_front(right);
                                res.push(left);
                                remaining = 0;
                                break;
                            }
                        }
                    } else {
                        res.push(chunk);
                        remaining -= len;
                    }
                } else {
                    res.push(chunk);
                    remaining -= len;
                }
            }
        }
        Ok(res)
    }

    /// Subdivide a chunk that exceeds the maximum size using size-aware row selection
    /// while maintaining zone boundary alignment
    fn subdivide_chunk_size_aware(
        &self,
        chunk: ArrayRef,
        max_size: u64,
    ) -> VortexResult<Vec<ArrayRef>> {
        let mut result = Vec::new();
        let mut offset = 0;
        let chunk_len = chunk.len();

        while offset < chunk_len {
            let remaining_chunk = chunk.slice(offset..chunk_len);
            let canonical_chunk = remaining_chunk.to_canonical();

            let rows_that_fit =
                canonical_chunk.rows_for_target_size(max_size, self.options.block_len_multiple);

            if rows_that_fit == 0 {
                // No rows fit, but we need to make progress to avoid infinite loop
                // Take at least one zone boundary worth of rows
                let min_rows = self.options.block_len_multiple.min(chunk_len - offset);
                let subchunk = chunk.slice(offset..offset + min_rows);
                result.push(subchunk);
                offset += min_rows;
            } else {
                let end = (offset + rows_that_fit).min(chunk_len);
                let subchunk = chunk.slice(offset..end);
                result.push(subchunk);
                offset = end;
            }
        }

        Ok(result)
    }

    fn push_back(&mut self, chunk: ArrayRef) {
        self.row_count += chunk.len();
        self.nbytes += chunk.to_canonical().compacted_size();
        self.data.push_back(chunk);
    }

    fn push_front(&mut self, chunk: ArrayRef) {
        self.row_count += chunk.len();
        self.nbytes += chunk.to_canonical().compacted_size();
        self.data.push_front(chunk);
    }

    fn pop_front(&mut self) -> Option<ArrayRef> {
        let res = self.data.pop_front();
        if let Some(chunk) = res.as_ref() {
            self.row_count -= chunk.len();
            self.nbytes -= chunk.to_canonical().compacted_size();
        }
        res
    }
}
