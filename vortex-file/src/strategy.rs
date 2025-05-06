//! This module defines the default layout strategy for a Vortex file.

use std::collections::VecDeque;
use std::sync::Arc;

use arcref::ArcRef;
use itertools::Itertools;
use vortex_array::arrays::ConstantArray;
use vortex_array::stats::{PRUNING_STATS, Stat};
use vortex_array::{Array, ArrayContext, ArrayRef, IntoArray};
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex_layout::layouts::dict::writer::DictStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::repartition::{
    RepartitionStrategy, RepartitionWriter, RepartitionWriterOptions,
};
use vortex_layout::layouts::struct_::writer::StructLayoutWriter;
use vortex_layout::layouts::zoned::writer::{ZonedLayoutOptions, ZonedLayoutWriter};
use vortex_layout::segments::SegmentWriter;
use vortex_layout::{LayoutRef, LayoutStrategy, LayoutWriter, LayoutWriterExt};

const ROW_BLOCK_SIZE: usize = 8192;

/// The default Vortex file layout strategy.
#[derive(Clone, Debug, Default)]
pub struct VortexLayoutStrategy;

impl LayoutStrategy for VortexLayoutStrategy {
    fn new_writer(&self, ctx: &ArrayContext, dtype: &DType) -> VortexResult<Box<dyn LayoutWriter>> {
        // First, we unwrap struct arrays into their components.
        if dtype.is_struct() {
            return Ok(StructLayoutWriter::try_new_with_strategy(ctx, dtype, self)?.boxed());
        }

        // We buffer arrays per column, before flushing them into a chunked layout.
        // This helps to keep consecutive chunks of a column adjacent for more efficient reads.
        let buffered_strategy: ArcRef<dyn LayoutStrategy> =
            ArcRef::new_arc(Arc::new(BufferedStrategy {
                child: ArcRef::new_arc(Arc::new(ChunkedLayoutStrategy::default()) as _),
                // TODO(ngates): this should really be amortized by the number of fields? Maybe the
                //  strategy could keep track of how many writers were created?
                buffer_size: 2 << 20, // 2MB
            }) as _);

        // Prior to compression, re-partition into size-based chunks.
        let coalescing_strategy = Arc::new(RepartitionStrategy {
            options: RepartitionWriterOptions {
                block_size_minimum: 1 << 20,        // 1 MB
                block_len_multiple: ROW_BLOCK_SIZE, // 8K rows
            },
            child: ArcRef::new_arc(Arc::new(BtrBlocksCompressedStrategy {
                child: buffered_strategy,
            })),
        });

        let dict_strategy = DictStrategy {
            codes: ArcRef::new_arc(coalescing_strategy.clone()),
            values: ArcRef::new_arc(Arc::new(BtrBlocksCompressedStrategy {
                child: ArcRef::new_arc(Arc::new(FlatLayoutStrategy::default())),
            })),
            fallback: ArcRef::new_arc(coalescing_strategy),
            options: Default::default(),
        };

        let writer = dict_strategy.new_writer(ctx, dtype)?;

        // Prior to repartitioning, we create a zone map
        let zoned_writer = ZonedLayoutWriter::new(
            ctx.clone(),
            dtype,
            writer,
            ArcRef::new_arc(Arc::new(BtrBlocksCompressedStrategy {
                child: ArcRef::new_arc(Arc::new(FlatLayoutStrategy::default())),
            })),
            ZonedLayoutOptions {
                block_size: ROW_BLOCK_SIZE,
                stats: PRUNING_STATS.into(),
                max_variable_length_statistics_size: 64,
            },
        )
        .boxed();

        let writer = RepartitionWriter::new(
            dtype.clone(),
            zoned_writer,
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

const COMPRESSION_DRIFT_THRESHOLD: f64 = 1.2;

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
        let chunk = chunk.to_canonical()?.into_array();

        // Compute the stats for the chunk prior to compression
        chunk
            .statistics()
            .compute_all(&Stat::all().collect::<Vec<_>>())?;

        // If we have information about the data from the previous chunk
        let compressed_chunk = if let Some(constant) = chunk.as_constant() {
            Some(ConstantArray::new(constant, chunk.len()).into_array())
        } else if let Some(prev_compression) = self.previous_chunk.as_ref() {
            let prev_chunk = prev_compression.chunk.clone();

            let canonical_nbytes = chunk.as_ref().nbytes();

            if let Some(encoded_chunk) = encode_children_like(chunk.clone(), prev_chunk)? {
                let ratio = canonical_nbytes as f64 / encoded_chunk.nbytes() as f64;

                // Make sure the ratio is within the expected drift, if it isn't we  fall back to the compressor.
                if ratio > (prev_compression.ratio / COMPRESSION_DRIFT_THRESHOLD) {
                    Some(encoded_chunk)
                } else {
                    log::trace!(
                        "Compressed to a ratio of {ratio}, which is below the threshold of {}",
                        prev_compression.ratio / COMPRESSION_DRIFT_THRESHOLD
                    );
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
                let canonical_size = chunk.nbytes() as f64;
                let compressed = BtrBlocksCompressor.compress(&chunk)?;

                if compressed.is_canonical()
                    || ((canonical_size / compressed.nbytes() as f64) < COMPRESSION_DRIFT_THRESHOLD)
                {
                    self.previous_chunk = None;
                } else {
                    self.previous_chunk = Some(PreviousCompression {
                        chunk: compressed.clone(),
                        ratio: canonical_size / compressed.nbytes() as f64,
                    });
                }

                compressed
            }
        };

        compressed_chunk.statistics().inherit(chunk.statistics());

        self.child.push_chunk(segment_writer, compressed_chunk)
    }

    fn flush(&mut self, segment_writer: &mut dyn SegmentWriter) -> VortexResult<()> {
        self.child.flush(segment_writer)
    }

    fn finish(&mut self, segment_writer: &mut dyn SegmentWriter) -> VortexResult<LayoutRef> {
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

    fn finish(&mut self, segment_writer: &mut dyn SegmentWriter) -> VortexResult<LayoutRef> {
        self.child.finish(segment_writer)
    }
}

fn encode_children_like(current: ArrayRef, previous: ArrayRef) -> VortexResult<Option<ArrayRef>> {
    if let Some(constant) = current.as_constant() {
        Ok(Some(
            ConstantArray::new(constant, current.len()).into_array(),
        ))
    } else if let Some(encoded) = previous
        .encoding()
        .encode(&current.to_canonical()?, Some(&previous))?
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
            new_children.push(encode_children_like(e.clone(), p)?.unwrap_or(e));
        }

        Ok(Some(encoded.with_children(&new_children)?))
    } else {
        Ok(None)
    }
}
