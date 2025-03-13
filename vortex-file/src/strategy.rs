//! This module defines the default layout strategy for a Vortex file.

use std::sync::Arc;

use vortex_array::arcref::ArcRef;
use vortex_array::stats::{PRUNING_STATS, STATS_TO_WRITE};
use vortex_array::{Array, ArrayContext, ArrayRef};
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_layout::layouts::chunked::writer::{ChunkedLayoutOptions, ChunkedLayoutWriter};
use vortex_layout::layouts::flat::FlatLayout;
use vortex_layout::layouts::flat::writer::FlatLayoutOptions;
use vortex_layout::layouts::stats::writer::{StatsLayoutOptions, StatsLayoutWriter};
use vortex_layout::layouts::struct_::writer::StructLayoutWriter;
use vortex_layout::segments::SegmentWriter;
use vortex_layout::writers::{RepartitionWriter, RepartitionWriterOptions};
use vortex_layout::{Layout, LayoutStrategy, LayoutWriter, LayoutWriterExt};

/// The default Vortex file layout strategy.
#[derive(Clone, Debug, Default)]
pub struct VortexLayoutStrategy;

impl LayoutStrategy for VortexLayoutStrategy {
    fn new_writer(&self, ctx: &ArrayContext, dtype: &DType) -> VortexResult<Box<dyn LayoutWriter>> {
        // First, we unwrap struct arrays into their components.
        if dtype.is_struct() {
            return StructLayoutWriter::try_new_with_factory(ctx, dtype, self.clone())
                .map(|w| w.boxed());
        }

        // Otherwise, we finish with compressing the chunks
        let writer = BtrBlocksCompressedWriter {
            child: ChunkedLayoutWriter::new(
                ctx.clone(),
                &DType::Null,
                ChunkedLayoutOptions {
                    chunk_strategy: ArcRef::new_arc(Arc::new(FlatLayoutOptions::default()) as _),
                },
            )
            .boxed(),
        }
        .boxed();

        // Prior to compression, re-partition into size-based chunks.
        let writer = RepartitionWriter::new(
            dtype.clone(),
            writer,
            RepartitionWriterOptions {
                block_size_minimum: 8 * (1 << 20), // 1 MB
                block_len_multiple: 8192,          // 8K rows
            },
        )
        .boxed();

        // Prior to repartitioning, we record statistics
        let writer = RepartitionWriter::new(
            dtype.clone(),
            StatsLayoutWriter::try_new(
                ctx.clone(),
                dtype,
                writer,
                ArcRef::new_arc(Arc::new(BtrBlocksCompressedStrategy {
                    child: ArcRef::new_ref(&FlatLayout),
                })),
                StatsLayoutOptions {
                    block_size: 8192,
                    stats: PRUNING_STATS.into(),
                },
            )?
            .boxed(),
            RepartitionWriterOptions {
                // No minimum block size in bytes
                block_size_minimum: 0,
                // Always repartition into 8K row blocks
                block_len_multiple: 8192,
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
        segments: &mut dyn SegmentWriter,
        chunk: ArrayRef,
    ) -> VortexResult<()> {
        // Compute the stats for the chunk prior to compression
        chunk.statistics().compute_all(STATS_TO_WRITE)?;

        let compressed = BtrBlocksCompressor.compress(&chunk)?;
        self.child.push_chunk(segments, compressed)
    }

    fn finish(&mut self, segments: &mut dyn SegmentWriter) -> VortexResult<Layout> {
        self.child.finish(segments)
    }
}
