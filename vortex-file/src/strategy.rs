//! This module defines the default layout strategy for a Vortex file.

use std::collections::VecDeque;
use std::sync::Arc;

use itertools::Itertools;
use sketches_ddsketch::DDSketch;
use vortex_array::arcref::ArcRef;
use vortex_array::arrays::ConstantArray;
use vortex_array::nbytes::NBytes;
use vortex_array::stats::{PRUNING_STATS, STATS_TO_WRITE};
use vortex_array::{Array, ArrayContext, ArrayRef, IntoArray};
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_dtype::DType;
use vortex_error::{VortexResult, VortexUnwrap, vortex_err};
use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::repartition::{RepartitionWriter, RepartitionWriterOptions};
use vortex_layout::layouts::stats::writer::{StatsLayoutOptions, StatsLayoutWriter};
use vortex_layout::layouts::struct_::writer::StructLayoutWriter;
use vortex_layout::segments::SegmentWriter;
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
            prev_compressed: None,
            child: strategy.new_writer(ctx, dtype)?,
            ratio: Default::default(),
        }
        .boxed();

        // Prior to compression, re-partition into size-based chunks.
        let writer = RepartitionWriter::new(
            dtype.clone(),
            writer,
            RepartitionWriterOptions {
                block_size_minimum: 1 << 20,        // 1 MB
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
                child: ArcRef::new_arc(Arc::new(FlatLayoutStrategy::default())),
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
            ratio: DDSketch::default(),
            prev_compressed: None,
        }
        .boxed())
    }
}

const COMPRESSION_DRIFT_THRESHOLD: f64 = 1.2;

/// A layout writer that compresses chunks using a sampling compressor, and re-uses the previous
/// compressed chunk as a hint for the next.
struct BtrBlocksCompressedWriter {
    child: Box<dyn LayoutWriter>,
    ratio: DDSketch,
    prev_compressed: Option<ArrayRef>,
}

impl LayoutWriter for BtrBlocksCompressedWriter {
    fn push_chunk(
        &mut self,
        segment_writer: &mut dyn SegmentWriter,
        chunk: ArrayRef,
    ) -> VortexResult<()> {
        // Compute the stats for the chunk prior to compression
        chunk.statistics().compute_all(STATS_TO_WRITE)?;

        // Short circuit the decision if the chunk is constant
        let compressed_chunk = if let Some(constant) = chunk.as_constant() {
            Some(ConstantArray::new(constant, chunk.len()).into_array())
        }
        // If we have information about the data from the previous chunk
        else if let Some(prev_compressed) = self.prev_compressed.as_ref() {
            let uncompressed_chunk = chunk.to_canonical()?;
            let uncompressed_nbytes = uncompressed_chunk.as_ref().nbytes();

            if let Some(compressed_chunk) =
                encode_children_like(uncompressed_chunk.into_array(), prev_compressed)?
            {
                let ratio = compressed_chunk.nbytes() as f64 / uncompressed_nbytes as f64;

                // Make sure the ratio is within the expected drift, if it isn't we fall back to the compressor.
                if let Some(avg_ratio) = self
                    .ratio
                    .quantile(0.5)
                    .map_err(|_| vortex_err!("Quantile outside 0.0 -> 1.0"))
                    .vortex_unwrap()
                {
                    if ratio < avg_ratio * COMPRESSION_DRIFT_THRESHOLD {
                        log::trace!(
                            "Compressed-like to a ratio of {ratio}, which is below the threshold of {}",
                            avg_ratio * COMPRESSION_DRIFT_THRESHOLD
                        );
                        Some(compressed_chunk)
                    } else {
                        log::trace!(
                            "Compressed to a ratio of {ratio}, which is above the threshold of {}",
                            avg_ratio * COMPRESSION_DRIFT_THRESHOLD
                        );
                        None
                    }
                } else {
                    // No ratios published yet
                    None
                }
            } else {
                log::debug!("Couldn't re-encode children");
                None
            }
        } else {
            None
        };

        let compressed_chunk = match compressed_chunk {
            Some(array) => array,
            None => {
                let uncompressed_chunk = chunk.to_canonical()?;
                let uncompressed_nbytes = uncompressed_chunk.as_ref().nbytes();

                let compressed_chunk =
                    BtrBlocksCompressor.compress_canonical(uncompressed_chunk)?;
                let compressed_nbytes = compressed_chunk.nbytes();

                let ratio = compressed_nbytes as f64 / uncompressed_nbytes as f64;
                self.ratio.add(ratio);
                self.prev_compressed = Some(compressed_chunk.clone());

                compressed_chunk
            }
        };

        compressed_chunk.statistics().inherit(chunk.statistics());

        self.child.push_chunk(segment_writer, compressed_chunk)
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

fn encode_children_like(current: ArrayRef, previous: &dyn Array) -> VortexResult<Option<ArrayRef>> {
    if let Some(constant) = current.as_constant() {
        Ok(Some(
            ConstantArray::new(constant, current.len()).into_array(),
        ))
    } else if let Some(encoded) = previous
        .vtable()
        .encode(&current.to_canonical()?, Some(previous))?
    {
        let previous_children = previous.children();
        let encoded_children = encoded.children();

        if previous_children.len() != encoded_children.len() {
            log::trace!(
                "Children count mismatch {} and {}",
                previous_children.len(),
                encoded_children.len()
            );
            return Ok(Some(encoded));
        }

        let mut new_children: Vec<Arc<dyn Array>> = Vec::with_capacity(encoded_children.len());

        for (p, e) in previous_children
            .into_iter()
            .zip_eq(encoded_children.into_iter())
        {
            new_children.push(encode_children_like(e.clone(), p.as_ref())?.unwrap_or(e));
        }

        Ok(Some(encoded.with_children(&new_children)?))
    } else {
        Ok(None)
    }
}
