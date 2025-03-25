//! This module defines the default layout strategy for a Vortex file.

use std::collections::VecDeque;
use std::sync::Arc;

use vortex_array::arcref::ArcRef;
use vortex_array::nbytes::NBytes;
use vortex_array::stats::{PRUNING_STATS, STATS_TO_WRITE};
use vortex_array::{Array, ArrayContext, ArrayRef};
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex_layout::layouts::flat::FlatLayout;
use vortex_layout::layouts::stats::writer::{StatsLayoutOptions, StatsLayoutWriter};
use vortex_layout::layouts::struct_::writer::StructLayoutWriter;
use vortex_layout::segments::SegmentWriter;
use vortex_layout::writers::{RepartitionWriter, RepartitionWriterOptions};
use vortex_layout::{Layout, LayoutStrategy, LayoutWriter, LayoutWriterExt};

const ROW_BLOCK_SIZE: usize = 8192;

/// The default Vortex file layout strategy.
#[derive(Clone, Debug, Default)]
pub struct VortexLayoutStrategy;

impl LayoutStrategy for VortexLayoutStrategy {
    fn new_writer(&self, ctx: &ArrayContext, dtype: &DType) -> VortexResult<Box<dyn LayoutWriter>> {
        // First, we unwrap struct arrays into their components.
        if dtype.is_struct() {
            return Ok(
                StructLayoutWriter::try_new_with_strategy(ctx, dtype, self.clone())?.boxed(),
            );
        }

        // We buffer arrays per column, before flushing them into a chunked layout.
        // This helps to keep consecutive chunks of a column adjacent for more efficient reads.
        let strategy: ArcRef<dyn LayoutStrategy> = ArcRef::new_arc(Arc::new(BufferedStrategy {
            child: ArcRef::new_arc(Arc::new(ChunkedLayoutStrategy::default()) as _),
            // TODO(ngates): this should really be amortized by the number of fields? Maybe the
            //  strategy could keep track of how many writers were created?
            buffer_size: 2 << 20, // 2MB
        }) as _);

        // Compress each chunk with btrblocks.
        let writer = BtrBlocksCompressedWriter {
            child: strategy.new_writer(ctx, dtype)?,
        }
        .boxed();

        // Prior to compression, re-partition into size-based chunks.
        let writer = RepartitionWriter::new(
            dtype.clone(),
            writer,
            RepartitionWriterOptions {
                block_size_minimum: 8 * (1 << 20),  // 1 MB
                block_len_multiple: ROW_BLOCK_SIZE, // 8K rows
            },
        )
        .boxed();

        // Prior to repartitioning, we record statistics
        let stats_writer = StatsLayoutWriter::try_new(
            ctx.clone(),
            dtype,
            writer,
            ArcRef::new_arc(Arc::new(BtrBlocksCompressedStrategy {
                child: ArcRef::new_ref(&FlatLayout),
            })),
            StatsLayoutOptions {
                block_size: ROW_BLOCK_SIZE,
                stats: PRUNING_STATS.into(),
            },
        )?
        .boxed();

        let writer = RepartitionWriter::new(
            dtype.clone(),
            stats_writer,
            RepartitionWriterOptions {
                // No minimum block size in bytes
                block_size_minimum: 0,
                // Always repartition into 8K row blocks
                block_len_multiple: ROW_BLOCK_SIZE,
            },
        )
        .boxed();

        Ok(writer)
    }
}

struct BtrBlocksCompressedStrategy {
    child: ArcRef<dyn LayoutStrategy>,
}

impl LayoutStrategy for BtrBlocksCompressedStrategy {
    fn new_writer(&self, ctx: &ArrayContext, dtype: &DType) -> VortexResult<Box<dyn LayoutWriter>> {
        let child = self.child.new_writer(ctx, dtype)?;
        Ok(BtrBlocksCompressedWriter { child }.boxed())
    }
}

/// A layout writer that compresses chunks using a sampling compressor, and re-uses the previous
/// compressed chunk as a hint for the next.
struct BtrBlocksCompressedWriter {
    child: Box<dyn LayoutWriter>,
}

impl LayoutWriter for BtrBlocksCompressedWriter {
    fn push_chunk(
        &mut self,
        segment_writer: &mut dyn SegmentWriter,
        chunk: ArrayRef,
    ) -> VortexResult<()> {
        // Compute the stats for the chunk prior to compression
        chunk.statistics().compute_all(STATS_TO_WRITE)?;

        let compressed = BtrBlocksCompressor.compress(&chunk)?;
        self.child.push_chunk(segment_writer, compressed)
    }

    fn flush(&mut self, segment_writer: &mut dyn SegmentWriter) -> VortexResult<()> {
        self.child.flush(segment_writer)
    }

    fn finish(&mut self, segment_writer: &mut dyn SegmentWriter) -> VortexResult<Layout> {
        self.child.finish(segment_writer)
    }
}

struct BufferedStrategy {
    child: ArcRef<dyn LayoutStrategy>,
    buffer_size: u64,
}

impl LayoutStrategy for BufferedStrategy {
    fn new_writer(&self, ctx: &ArrayContext, dtype: &DType) -> VortexResult<Box<dyn LayoutWriter>> {
        let child = self.child.new_writer(ctx, dtype)?;
        Ok(BufferedWriter {
            chunks: Default::default(),
            nbytes: 0,
            buffer_size: self.buffer_size,
            child,
        }
        .boxed())
    }
}

struct BufferedWriter {
    chunks: VecDeque<ArrayRef>,
    nbytes: u64,
    buffer_size: u64,
    child: Box<dyn LayoutWriter>,
}

impl LayoutWriter for BufferedWriter {
    fn push_chunk(
        &mut self,
        segment_writer: &mut dyn SegmentWriter,
        chunk: ArrayRef,
    ) -> VortexResult<()> {
        self.nbytes += chunk.nbytes() as u64;
        self.chunks.push_back(chunk);
        // Wait until we're at 2x the buffer size before flushing 1x the buffer size
        // This avoids small tail stragglers being flushed at the end of the file.
        if self.nbytes >= 2 * self.buffer_size {
            while self.nbytes > self.buffer_size {
                if let Some(chunk) = self.chunks.pop_front() {
                    self.nbytes -= chunk.nbytes() as u64;
                    self.child.push_chunk(segment_writer, chunk)?;
                } else {
                    break;
                }
            }
        }
        Ok(())
    }

    fn flush(&mut self, segment_writer: &mut dyn SegmentWriter) -> VortexResult<()> {
        for chunk in self.chunks.drain(..) {
            self.child.push_chunk(segment_writer, chunk)?;
        }
        self.child.flush(segment_writer)
    }

    fn finish(&mut self, segment_writer: &mut dyn SegmentWriter) -> VortexResult<Layout> {
        self.child.finish(segment_writer)
    }
}
