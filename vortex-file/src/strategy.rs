//! This module defines the default layout strategy for a Vortex file.

use std::sync::Arc;

use vortex_array::arcref::ArcRef;
use vortex_array::nbytes::NBytes;
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

        // Otherwise, we finish with compressing the chunks
        let writer = BtrBlocksCompressedWriter {
            previous_chunk: None,
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
        Ok(BtrBlocksCompressedWriter {
            child,
            previous_chunk: None,
        }
        .boxed())
    }
}

struct PreviousCompression {
    chunk: ArrayRef,
    ratio: f64,
}

const COMPRESSION_DRIFT_THRESHOLD: f64 = 2.0;

/// A layout writer that compresses chunks using a sampling compressor, and re-uses the previous
/// compressed chunk as a hint for the next.
struct BtrBlocksCompressedWriter {
    child: Box<dyn LayoutWriter>,
    previous_chunk: Option<PreviousCompression>,
}

impl LayoutWriter for BtrBlocksCompressedWriter {
    fn push_chunk(
        &mut self,
        segment_writer: &mut dyn SegmentWriter,
        chunk: ArrayRef,
    ) -> VortexResult<()> {
        // Compute the stats for the chunk prior to compression
        chunk.statistics().compute_all(STATS_TO_WRITE)?;

        let mut compressed_array = None;

        if let Some(prev_compression) = self.previous_chunk.as_ref() {
            let prev = prev_compression.chunk.clone();
            let prev_vtable = prev.vtable();
            let canonical = chunk.to_canonical()?;
            let encoded = prev_vtable.encode(&canonical, Some(&prev))?;

            let prev_children = prev.children();
            let encoded_children = encoded.children();

            let new_children = prev_children
                .into_iter()
                .zip(encoded_children.into_iter())
                .map(|(prev, encoded)| {
                    let encoded = encoded.to_canonical()?;
                    let encoded = prev.vtable().encode(&encoded, Some(&prev))?;

                    Ok(encoded)
                })
                .collect::<VortexResult<Vec<_>>>()?;

            let new_array = prev_vtable.replace_children(encoded, new_children)?;
            let ratio = canonical.as_ref().nbytes() as f64 / new_array.nbytes() as f64;

            // not sure this condition is right, but the idea is to make sure the ratio is within the expected drift.
            // If it isn't we  fall back to the compressor.
            if ratio < prev_compression.ratio * COMPRESSION_DRIFT_THRESHOLD {
                compressed_array = Some(new_array);
            }
        }

        let compressed = match compressed_array {
            Some(array) => array,
            None => {
                let original_size = chunk.nbytes() as f64;
                let compressed = BtrBlocksCompressor.compress(&chunk)?;
                self.previous_chunk = Some(PreviousCompression {
                    chunk: compressed.clone(),
                    ratio: compressed.nbytes() as f64 / original_size,
                });
                compressed
            }
        };

        self.child.push_chunk(segment_writer, compressed)
    }

    fn finish(&mut self, segment_writer: &mut dyn SegmentWriter) -> VortexResult<Layout> {
        self.child.finish(segment_writer)
    }
}
