//! This module defines the default layout strategy for a Vortex file.

use std::collections::VecDeque;
use std::sync::{Arc, LazyLock};

use vortex_array::array::ChunkedArray;
use vortex_array::compute::slice;
use vortex_array::stats::STATS_TO_WRITE;
use vortex_array::{Array, IntoArray, IntoCanonical};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};
use vortex_layout::layouts::chunked::writer::{ChunkedLayoutOptions, ChunkedLayoutWriter};
use vortex_layout::layouts::flat::writer::FlatLayoutOptions;
use vortex_layout::layouts::struct_::writer::StructLayoutWriter;
use vortex_layout::segments::SegmentWriter;
use vortex_layout::{Layout, LayoutStrategy, LayoutWriter, LayoutWriterExt};
use vortex_sampling_compressor::compressors::CompressionTree;
use vortex_sampling_compressor::{SamplingCompressor, DEFAULT_COMPRESSORS};

static COMPRESSOR: LazyLock<Arc<SamplingCompressor<'static>>> =
    LazyLock::new(|| Arc::new(SamplingCompressor::new(DEFAULT_COMPRESSORS)));

/// The default Vortex file layout strategy.
#[derive(Clone)]
pub struct VortexLayoutStrategy {
    /// The minimum size of a block in bytes. The block will be cut at the next multiple
    /// of the `block_len` larger than this size.
    pub minimum_block_size: usize,
    /// The divisor for the number of rows in a block.
    pub block_len_multiple: usize,
}

impl Default for VortexLayoutStrategy {
    fn default() -> Self {
        Self {
            minimum_block_size: 8 * (1 << 20), // 8MB
            block_len_multiple: 8 * 1024,      // 8192 rows
        }
    }
}

impl LayoutStrategy for VortexLayoutStrategy {
    fn new_writer(&self, dtype: &DType) -> VortexResult<Box<dyn LayoutWriter>> {
        // First, we unwrap struct arrays into their components.
        if dtype.is_struct() {
            return StructLayoutWriter::try_new_with_factory(dtype, self.clone())
                .map(|w| w.boxed());
        }

        // Then we re-chunk each column per our strategy...
        Ok(ColumnChunker::new(
            dtype.clone(),
            // ...compress each chunk using a sampling compressor...
            SamplingCompressorWriter {
                compressor: COMPRESSOR.clone(),
                compress_like: None,
                child: ChunkedLayoutWriter::new(
                    dtype,
                    ChunkedLayoutOptions {
                        // ...and write each chunk as a flat layout.
                        chunk_strategy: Arc::new(FlatLayoutOptions::default()),
                        ..Default::default()
                    },
                )
                .boxed(),
            }
            .boxed(),
            self.clone(),
        )
        .boxed())
    }
}

/// Each column is chunked into multiples of 8096 values, of at least 1MB in uncompressed size.
struct ColumnChunker {
    dtype: DType,
    chunks: VecDeque<Array>,
    row_count: usize,
    nbytes: usize,
    writer: Box<dyn LayoutWriter>,
    options: VortexLayoutStrategy,
}

impl ColumnChunker {
    pub fn new(dtype: DType, writer: Box<dyn LayoutWriter>, options: VortexLayoutStrategy) -> Self {
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
        if self.nbytes >= self.options.minimum_block_size {
            let nblocks = self.row_count / self.options.block_len_multiple;

            // If we don't have a full block, then continue anyway.
            if nblocks == 0 {
                // TODO(ngates): if we exceed a maximum block size, regardless of row count we should
                //  flush the chunk. This can happen for columns with very large cells.
                return Ok(());
            }

            if nblocks > 1 {
                // TODO(ngates): if we have _too_ many blocks, then we might look into slicing
                //  the chunks to be smaller blocks.
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
            let chunk = ChunkedArray::try_new(chunks, self.dtype.clone())
                .vortex_expect("failed to create chunked array")
                .into_canonical()?
                .into_array();

            self.writer.push_chunk(segments, chunk)?;
        }

        Ok(())
    }
}

impl LayoutWriter for ColumnChunker {
    fn push_chunk(&mut self, segments: &mut dyn SegmentWriter, chunk: Array) -> VortexResult<()> {
        // We make sure the chunks are canonical so our nbytes measurement is accurate.
        let chunk = chunk.into_canonical()?.into_array();

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
        let chunk = ChunkedArray::try_new(self.chunks.drain(..).collect(), self.dtype.clone())
            .vortex_expect("failed to create chunked array")
            .into_canonical()?
            .into_array();
        self.writer.push_chunk(segments, chunk)?;
        self.writer.finish(segments)
    }
}

/// A layout writer that compresses chunks using a sampling compressor, and re-uses the previous
/// compressed chunk as a hint for the next.
struct SamplingCompressorWriter {
    compressor: Arc<SamplingCompressor<'static>>,
    compress_like: Option<CompressionTree<'static>>,
    child: Box<dyn LayoutWriter>,
}

impl LayoutWriter for SamplingCompressorWriter {
    fn push_chunk(&mut self, segments: &mut dyn SegmentWriter, chunk: Array) -> VortexResult<()> {
        // Compute the stats for the chunk prior to compression
        chunk.statistics().compute_all(STATS_TO_WRITE)?;

        let (compressed, tree) = self
            .compressor
            .compress(&chunk, self.compress_like.as_ref())?
            .into_parts();
        self.compress_like = tree;
        self.child.push_chunk(segments, compressed)
    }

    fn finish(&mut self, segments: &mut dyn SegmentWriter) -> VortexResult<Layout> {
        self.child.finish(segments)
    }
}
