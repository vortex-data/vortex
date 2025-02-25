use std::collections::VecDeque;

use vortex_array::arrays::ChunkedArray;
use vortex_array::compute::slice;
use vortex_array::nbytes::NBytes;
use vortex_array::{Array, ArrayRef, IntoArray};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};

use crate::segments::SegmentWriter;
use crate::{Layout, LayoutWriter};

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

    fn flush(&mut self, segments: &mut dyn SegmentWriter) -> VortexResult<()> {
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
                    let left = slice(&chunk, 0, remaining)?;
                    let right = slice(&chunk, remaining, len)?;
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

            self.writer.push_chunk(segments, chunk)?;
        }

        Ok(())
    }
}

impl LayoutWriter for RepartitionWriter {
    fn push_chunk(
        &mut self,
        segments: &mut dyn SegmentWriter,
        chunk: ArrayRef,
    ) -> VortexResult<()> {
        // We make sure the chunks are canonical so our nbytes measurement is accurate.
        let chunk = chunk.to_canonical()?.into_array();

        // Split chunks into 8192 blocks to make sure we don't over-size them.
        let mut offset = 0;
        while offset < chunk.len() {
            let end = (offset + self.options.block_len_multiple).min(chunk.len());
            let c = slice(&chunk, offset, end)?;
            self.row_count += c.len();
            self.nbytes += c.nbytes();
            self.chunks.push_back(c);
            offset = end;

            self.flush(segments)?;
        }

        Ok(())
    }

    fn finish(&mut self, segments: &mut dyn SegmentWriter) -> VortexResult<Layout> {
        let chunk =
            ChunkedArray::new_unchecked(self.chunks.drain(..).collect(), self.dtype.clone())
                .to_canonical()?
                .into_array();
        self.writer.push_chunk(segments, chunk)?;
        self.writer.finish(segments)
    }
}
