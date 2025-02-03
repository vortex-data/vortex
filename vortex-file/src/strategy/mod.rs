//! This module defines the default layout strategy for a Vortex file.

use std::collections::VecDeque;
use std::sync::{Arc, LazyLock};

use vortex_array::array::ChunkedArray;
use vortex_array::compute::slice;
use vortex_array::{Array, IntoArray, IntoCanonical};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};
use vortex_layout::layouts::chunked::writer::{ChunkedLayoutOptions, ChunkedLayoutWriter};
use vortex_layout::layouts::flat::writer::FlatLayoutWriter;
use vortex_layout::layouts::struct_::writer::StructLayoutWriter;
use vortex_layout::segments::SegmentWriter;
use vortex_layout::{Layout, LayoutStrategy, LayoutWriter, LayoutWriterExt};
use vortex_sampling_compressor::{SamplingCompressor, DEFAULT_COMPRESSORS};

static COMPRESSOR: LazyLock<Arc<SamplingCompressor<'static>>> =
    LazyLock::new(|| Arc::new(SamplingCompressor::new(DEFAULT_COMPRESSORS)));

struct SamplingCompressorStrategy(Arc<SamplingCompressor<'static>>);

impl LayoutStrategy for SamplingCompressorStrategy {
    fn new_writer(&self, dtype: &DType) -> VortexResult<Box<dyn LayoutWriter>> {
        Ok(SamplingCompressorWriter {
            compressor: self.0.clone(),
            child: Box::new(FlatLayoutWriter::new(dtype.clone(), Default::default())),
        }
        .boxed())
    }
}

struct SamplingCompressorWriter {
    compressor: Arc<SamplingCompressor<'static>>,
    child: Box<dyn LayoutWriter>,
}

impl LayoutWriter for SamplingCompressorWriter {
    fn push_chunk(&mut self, segments: &mut dyn SegmentWriter, chunk: Array) -> VortexResult<()> {
        self.child.push_chunk(
            segments,
            self.compressor
                .compress(&chunk, None)
                .map(|a| a.into_array())?,
        )
    }

    fn finish(&mut self, segments: &mut dyn SegmentWriter) -> VortexResult<Layout> {
        self.child.finish(segments)
    }
}

/// The default Vortex file layout strategy.
pub struct VortexLayoutStrategy;

impl LayoutStrategy for VortexLayoutStrategy {
    fn new_writer(&self, dtype: &DType) -> VortexResult<Box<dyn LayoutWriter>> {
        if dtype.is_struct() {
            StructLayoutWriter::try_new_with_factory(dtype, VortexLayoutStrategy).map(|w| w.boxed())
        } else {
            Ok(ColumnChunker::new(
                dtype.clone(),
                ChunkedLayoutWriter::new(
                    dtype,
                    ChunkedLayoutOptions {
                        chunk_strategy: Arc::new(SamplingCompressorStrategy(COMPRESSOR.clone())),
                        ..Default::default()
                    },
                )
                .boxed(),
            )
            .boxed())
        }
    }
}

/// Each column is chunked into multiples of 8096 values, of at least 1MB in uncompressed size.
struct ColumnChunker {
    dtype: DType,
    chunks: VecDeque<Array>,
    row_count: usize,
    nbytes: usize,
    writer: Box<dyn LayoutWriter>,
}

impl ColumnChunker {
    const BLOCK_LEN: usize = 8192;
    const BLOCK_SIZE: usize = 1 << 20; // 1MB

    pub fn new(dtype: DType, writer: Box<dyn LayoutWriter>) -> Self {
        Self {
            dtype,
            chunks: VecDeque::new(),
            row_count: 0,
            nbytes: 0,
            writer,
        }
    }

    fn flush(&mut self, segments: &mut dyn SegmentWriter) -> VortexResult<()> {
        if self.nbytes >= Self::BLOCK_SIZE {
            let nblocks = self.row_count / Self::BLOCK_LEN;

            // If we don't have a full block, then continue anyway.
            if nblocks == 0 {
                // TODO(ngates): if we exceed a maximum block size, regardless of row count we should
                //  flush the chunk. This can happen for columns with very large cells.
                return Ok(());
            }

            let mut chunks = Vec::with_capacity(self.chunks.len());
            let mut remaining = nblocks * Self::BLOCK_LEN;

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
            assert!(chunks.len() > 0);
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

        self.row_count += chunk.len();
        self.nbytes += chunk.nbytes();
        self.chunks.push_back(chunk);
        self.flush(segments)
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
