//! This module defines the default layout strategy for a Vortex file.

use std::sync::Arc;

use itertools::Itertools;
use vortex_array::aliases::hash_map::HashMap;
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

        // If we have information about the previous chunk
        if let Some(prev_compression) = self.previous_chunk.as_ref() {
            let prev_chunk = prev_compression.chunk.clone();
            let prev_vtable = prev_chunk.vtable();
            let canonical = chunk.to_canonical()?;

            log::debug!(
                "Trying to encode {} array into {}",
                canonical.as_ref().encoding(),
                prev_vtable.id()
            );
            match prev_vtable.encode(&canonical, Some(&prev_chunk))? {
                // Encoding isn't supported, so we remove the previous chunk state to let the compressor to try again.
                None => {
                    self.previous_chunk.take();
                }
                Some(encoded) => {
                    let prev_children = prev_compression.chunk.named_children();
                    let encoded_children_names = encoded.named_children();

                    let mut new_map = HashMap::new();
                    for (child_name, child) in encoded_children_names.into_iter() {
                        // If there's a matching child, we try and encode the child
                        if let Some((_, prev_child)) =
                            prev_children.iter().find(|(name, _)| name == &child_name)
                        {
                            if let Some(new_encoded_child) = prev_child
                                .vtable()
                                .encode(&child.to_canonical()?, Some(prev_child))?
                            {
                                new_map.insert(child_name.clone(), new_encoded_child);
                            } else if prev_child.encoding() != child.encoding() {
                                log::warn!(
                                    "Couldn't encode {} array as {}",
                                    child.encoding(),
                                    prev_child.encoding()
                                )
                            }
                        }

                        // If we didn't encode the child, we keep the existing one.
                        if !new_map.contains_key(&child_name) {
                            new_map.insert(child_name, child);
                        }
                    }

                    // We turn the map into a children vec, keeping the order of children.
                    let new_children = encoded
                        .children_names()
                        .iter()
                        .map(|name| new_map[name].clone())
                        .collect_vec();

                    let new_array = encoded.with_children(&new_children)?;
                    let ratio = new_array.nbytes() as f64 / canonical.as_ref().nbytes() as f64;

                    log::debug!("Array compressed with compression ratio of {ratio}");

                    if ratio < prev_compression.ratio * COMPRESSION_DRIFT_THRESHOLD {
                        compressed_array = Some(new_array);
                    } else {
                        log::debug!(
                            "Compression ratio of {ratio} which is above the accepted threshold of {accepted}, falling back to the compressor.",
                            accepted = prev_compression.ratio * COMPRESSION_DRIFT_THRESHOLD
                        );
                    }
                }
            }
        }

        let compressed = match compressed_array {
            Some(array) => array,
            None => {
                let compressed = BtrBlocksCompressor.compress(&chunk)?;
                self.previous_chunk = Some(PreviousCompression {
                    chunk: compressed.clone(),
                    ratio: compressed.nbytes() as f64 / chunk.nbytes() as f64,
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
