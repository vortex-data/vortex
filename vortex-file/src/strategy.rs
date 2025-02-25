//! This module defines the default layout strategy for a Vortex file.

use std::sync::Arc;

use vortex_array::stats::{PRUNING_STATS, STATS_TO_WRITE};
use vortex_array::{Array, ArrayRef};
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
use vortex_sampling_compressor::compressors::CompressionTree;
use vortex_sampling_compressor::{DEFAULT_COMPRESSORS, SamplingCompressor};

/// The default Vortex file layout strategy.
#[derive(Clone, Debug, Default)]
pub struct VortexLayoutStrategy {
    options: StrategyOptions,
}

/// Compressor to use for chunks.
#[derive(Default, Copy, Clone, Debug, PartialEq, Eq)]
pub enum Compressor {
    /// BtrBlocks-style sampling compression that compresses in two passes.
    ///
    /// Better for wide-tables with many columns that heavily compress with Dict, RLE and Frequency.
    #[default]
    BtrBlocks,
    /// A different sampling compressor that only examines the first chunk of data to determine
    /// the best compression strategy for all chunks.
    ///
    /// This compressor performs better for long, skinny tables with relatively homogenous data
    /// distributions.
    Sampling,
}

/// Options to send into the layout strategy.
#[derive(Default, Clone, Debug)]
pub struct StrategyOptions {
    pub compressor: Compressor,
}

impl VortexLayoutStrategy {
    /// Create a new layout writer with the default layout selection and chunk compressor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new layout writer with the specified options, e.g. the chunk compressor.
    pub fn new_with(options: StrategyOptions) -> Self {
        Self { options }
    }

    fn new_compressed_writer(&self) -> Box<dyn LayoutWriter> {
        match self.options.compressor {
            Compressor::BtrBlocks => BtrBlocksCompressedWriter {
                child: ChunkedLayoutWriter::new(
                    &DType::Null,
                    ChunkedLayoutOptions {
                        chunk_strategy: Arc::new(FlatLayoutOptions::default()),
                    },
                )
                .boxed(),
            }
            .boxed(),
            Compressor::Sampling => SamplingCompressorWriter {
                compressor: Arc::new(SamplingCompressor::new(DEFAULT_COMPRESSORS)),
                compress_like: None,
                child: ChunkedLayoutWriter::new(
                    &DType::Null,
                    ChunkedLayoutOptions {
                        chunk_strategy: Arc::new(FlatLayoutOptions::default()),
                    },
                )
                .boxed(),
            }
            .boxed(),
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

        // Otherwise, we finish with compressing the chunks
        let writer = self.new_compressed_writer();

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
                dtype,
                writer,
                Arc::new(FlatLayout),
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

/// A layout writer that compresses chunks using a sampling compressor, and re-uses the previous
/// compressed chunk as a hint for the next.
#[allow(dead_code)]
struct SamplingCompressorWriter {
    compressor: Arc<SamplingCompressor<'static>>,
    compress_like: Option<CompressionTree<'static>>,
    child: Box<dyn LayoutWriter>,
}

impl LayoutWriter for SamplingCompressorWriter {
    fn push_chunk(
        &mut self,
        segments: &mut dyn SegmentWriter,
        chunk: ArrayRef,
    ) -> VortexResult<()> {
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
