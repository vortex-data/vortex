use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;

use async_stream::try_stream;
use futures::StreamExt as _;
use arcref::ArcRef;
use vortex_array::arrays::ChunkedArray;
use vortex_array::{Array, ArrayContext, ArrayRef, IntoArray};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};

use crate::segments::SegmentWriter;
use crate::{LayoutStrategy, LayoutWriter, SequentialArrayStream};

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
        dtype: &DType,
        segment_writer: Arc<dyn SegmentWriter>,
        stream: SequentialArrayStream,
    ) -> Pin<Box<dyn LayoutWriter>> {
        // TODO(os): spawn stream below like:
        // canon_stream = stream.map(async {to_canonical}).map(spawn).buffered(parallelism)
        let mut canonical_stream = stream.map(|chunk| {
            let (sequence_id, chunk) = chunk?;
            VortexResult::Ok((sequence_id, chunk.to_canonical()?.into_array()))
        });

        let dtype_clone = dtype.clone();
        let options = self.options.clone();
        let repartitioned_stream = try_stream! {
            let mut last_sequence_pointer = None;
            let mut chunks = ChunksBuffer::new(options.clone());
            while let Some(chunk) = canonical_stream.next().await {
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
                        let chunked_array = to_canonical_chunked(output_chunks, &dtype_clone)?;
                        yield (sequence_pointer.advance(), chunked_array)
                    }
                }
                last_sequence_pointer = Some(sequence_pointer);
            }
            // stream is consumed, flush remaining chunks if any
            let Some(mut sequence_pointer) = last_sequence_pointer else {
                assert!(chunks.data.is_empty());
                return;
            };
            let to_flush = to_canonical_chunked(chunks.data.drain(..).collect(), &dtype_clone)?;
            yield (sequence_pointer.advance(), to_flush)
        };

        self.child
            .write_stream(&ctx, &dtype, segment_writer, Box::pin(repartitioned_stream))
    }
}

fn to_canonical_chunked(chunks: Vec<ArrayRef>, dtype: &DType) -> VortexResult<ArrayRef> {
    Ok(ChunkedArray::new_unchecked(chunks, dtype.clone())
        .to_canonical()?
        .into_array())
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
        self.nbytes > self.options.block_size_minimum
            && self.row_count > self.options.block_len_multiple
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
        self.nbytes = chunk.nbytes();
        self.data.push_back(chunk);
    }

    fn push_front(&mut self, chunk: ArrayRef) {
        self.row_count += chunk.len();
        self.nbytes = chunk.nbytes();
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
