use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;

use async_stream::try_stream;
use async_trait::async_trait;
use futures::StreamExt as _;
use arcref::ArcRef;
use vortex_array::arrays::ChunkedArray;
use vortex_array::{Array, ArrayContext, ArrayRef, IntoArray};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};

use crate::segments::{ConcurrentSegmentWriter, NewSegmentWriter};
use crate::{
    LayoutRef, LayoutStrategy, LayoutWriter, LayoutWriterExt, NewLayoutStrategy, NewLayoutWriter,
    SequentialArrayStream,
};

pub struct RepartitionStrategy {
    pub options: RepartitionWriterOptions,
    pub child: ArcRef<dyn LayoutStrategy>,
}

impl LayoutStrategy for RepartitionStrategy {
    fn new_writer(&self, ctx: &ArrayContext, dtype: &DType) -> VortexResult<Box<dyn LayoutWriter>> {
        Ok(RepartitionWriter::new(
            dtype.clone(),
            self.child.new_writer(ctx, dtype)?,
            self.options.clone(),
        )
        .boxed())
    }
}

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
pub struct RepartitionWriter {
    dtype: DType,
    chunks: VecDeque<ArrayRef>,
    row_count: usize,
    nbytes: usize,
    writer: Box<dyn LayoutWriter>,
    options: RepartitionWriterOptions,
}

impl RepartitionWriter {
    pub fn new(
        dtype: DType,
        writer: Box<dyn LayoutWriter>,
        options: RepartitionWriterOptions,
    ) -> Self {
        Self {
            dtype,
            chunks: VecDeque::new(),
            row_count: 0,
            nbytes: 0,
            writer,
            options,
        }
    }

    async fn maybe_flush_chunk(
        &mut self,
        segments: &mut dyn ConcurrentSegmentWriter,
    ) -> VortexResult<()> {
        if self.nbytes >= self.options.block_size_minimum {
            let nblocks = self.row_count / self.options.block_len_multiple;

            // If we don't have a full block, then wait for more
            if nblocks == 0 {
                return Ok(());
            }

            let mut chunks = Vec::with_capacity(self.chunks.len());
            let mut remaining = nblocks * self.options.block_len_multiple;

            while remaining > 0 {
                let chunk = self.chunks.pop_front().vortex_expect("chunk is missing");
                self.row_count -= chunk.len();
                self.nbytes -= chunk.nbytes();

                let len = chunk.len();

                if len > remaining {
                    let left = chunk.slice(0, remaining)?;
                    let right = chunk.slice(remaining, len)?;
                    self.row_count += right.len();
                    self.nbytes += right.nbytes();
                    self.chunks.push_front(right);

                    chunks.push(left);
                    remaining = 0;
                } else {
                    chunks.push(chunk);
                    remaining -= len;
                }
            }

            // Combine the chunks to and flush them to the layout.
            assert!(!chunks.is_empty());
            let chunk = ChunkedArray::new_unchecked(chunks, self.dtype.clone())
                .to_canonical()?
                .into_array();

            self.writer.push_chunk(segments, chunk).await?;
        }

        Ok(())
    }
}

#[async_trait]
impl LayoutWriter for RepartitionWriter {
    async fn push_chunk(
        &mut self,
        segment_writer: &mut dyn ConcurrentSegmentWriter,
        chunk: ArrayRef,
    ) -> VortexResult<()> {
        assert_eq!(
            chunk.dtype(),
            &self.dtype,
            "Can't push chunks of the wrong dtype into a LayoutWriter. Pushed {} but expected {}.",
            chunk.dtype(),
            self.dtype
        );
        // We make sure the chunks are canonical so our nbytes measurement is accurate.
        let chunk = chunk.to_canonical()?.into_array();

        // Split chunks into 8192 blocks to make sure we don't over-size them.
        let mut offset = 0;
        while offset < chunk.len() {
            let end = (offset + self.options.block_len_multiple).min(chunk.len());
            let c = chunk.slice(offset, end)?;
            self.row_count += c.len();
            self.nbytes += c.nbytes();
            self.chunks.push_back(c);
            offset = end;

            self.maybe_flush_chunk(segment_writer).await?;
        }

        Ok(())
    }

    async fn flush(
        &mut self,
        segment_writer: &mut dyn ConcurrentSegmentWriter,
    ) -> VortexResult<()> {
        let chunk =
            ChunkedArray::new_unchecked(self.chunks.drain(..).collect(), self.dtype.clone())
                .to_canonical()?
                .into_array();
        self.writer.push_chunk(segment_writer, chunk).await?;
        self.writer.flush(segment_writer).await
    }

    async fn finish(
        &mut self,
        segment_writer: &mut dyn ConcurrentSegmentWriter,
    ) -> VortexResult<LayoutRef> {
        self.writer.finish(segment_writer).await
    }
}

struct NewRepartitionStrategy {
    options: RepartitionWriterOptions,
    child: ArcRef<dyn NewLayoutStrategy>,
}

impl NewLayoutStrategy for NewRepartitionStrategy {
    fn write_stream(
        &self,
        ctx: &ArrayContext,
        dtype: &DType,
        segment_writer: Arc<dyn NewSegmentWriter>,
        stream: SequentialArrayStream,
    ) -> Pin<Box<dyn NewLayoutWriter>> {
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
