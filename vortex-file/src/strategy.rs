// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module defines the default layout strategy for a Vortex file.

use std::sync::Arc;

use arcref::ArcRef;
use vortex_array::stats::PRUNING_STATS;
use vortex_array::ArrayContext;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_layout::layouts::buffered::BufferedStrategy;
use vortex_layout::layouts::chunked::writer::ChunkedStrategy;
use vortex_layout::layouts::compressed::BtrBlocksCompressedStrategy;
use vortex_layout::layouts::dict::writer::DictStrategy;
use vortex_layout::layouts::flat::writer::{FlatLayoutStrategy, DEFAULT_FLAT_STRATEGY};
use vortex_layout::layouts::repartition::{RepartitionStrategy, RepartitionWriterOptions};
use vortex_layout::layouts::struct_::writer::StructStrategy;
use vortex_layout::layouts::view::writer::ViewStrategy;
use vortex_layout::layouts::zoned::writer::{ZonedLayoutOptions, ZonedStrategy};
use vortex_layout::segments::SequenceWriter;
use vortex_layout::{LayoutRef, LayoutStrategy, LayoutStrategyExt, SendableLayoutFuture, SendableSequentialStream, TaskExecutor};

const ROW_BLOCK_SIZE: usize = 8192;

pub struct VortexLayoutStrategy;

/// Fixed-size column type strategy.
pub struct FixedSizeStrategy {
    /// The data type of the array chunks.
    dtype: DType,
}

/// A layout strategy that generates layouts differently for fixed-size and variable-size types.
pub struct SelectingLayoutStrategy {
    fixed_size: Arc<dyn LayoutStrategy>,
    variable_size: Arc<dyn LayoutStrategy>,
}

impl LayoutStrategy for SelectingLayoutStrategy {
    fn write_stream(&self, ctx: &ArrayContext, sequence_writer: SequenceWriter, stream: SendableSequentialStream) -> SendableLayoutFuture {
        todo!()
    }
}

const FLAT_STRATEGY: ArcRef<dyn LayoutStrategy> = ArcRef::new_ref(&DEFAULT_FLAT_STRATEGY);

/// Write a stream of chunks of variable-width data to the sink, emitting a layout that covers
/// it all.
async fn write_variable_size(
    ctx: &ArrayContext,
    sink: SequenceWriter,
    source: SendableSequentialStream,
    executor: &Arc<dyn TaskExecutor>,
) -> VortexResult<LayoutRef> {
    let zoned = ZonedStrategy::new(
        FLAT_STRATEGY.clone(),
        FLAT_STRATEGY.clone(),
        ZonedLayoutOptions::default(),
        executor.clone(),
    );

    let chunked = ChunkedStrategy::default()
        .buffered(2 << 20);
    let buffered = chunked.buffered(2 << 20);



    zoned.write_stream(
        ctx,
        sink,
        source
    ).await
}

impl VortexLayoutStrategy {
    pub fn with_executor(executor: Arc<dyn TaskExecutor>) -> ArcRef<dyn LayoutStrategy> {
        // 7. for each chunk create a flat layout
        let buffered = ArcRef::new_arc(ChunkedStrategy::default().buffered(2 << 20));
        // 5. compress each chunk
        let compressing = arcref(BtrBlocksCompressedStrategy::new(
            buffered,
            executor.clone(),
            16,
        ));

        // 4. prior to compression, coalesce up to a minimum size
        let coalescing = arcref(RepartitionStrategy::new(
            compressing,
            RepartitionWriterOptions {
                block_size_minimum: 1 << 20,
                block_len_multiple: ROW_BLOCK_SIZE,
            },
        ));

        // 2.1. | 3.1. compress stats tables and dict values.
        let compress_then_flat = arcref(BtrBlocksCompressedStrategy::new(
            ArcRef::new_ref(&DEFAULT_FLAT_STRATEGY),
            executor.clone(),
            1,
        ));

        // 3. apply dict encoding or fallback
        let dict = arcref(DictStrategy::new(
            coalescing.clone(),
            arcref(ViewStrategy::new(
                ArcRef::new_arc(Arc::new(FlatLayoutStrategy::default())),
                coalescing.clone(),
            )),
            coalescing.clone(),
            Default::default(),
            executor.clone(),
        ));

        // 2. calculate stats for each row group
        let stats = arcref(ZonedStrategy::new(
            dict,
            compress_then_flat.clone(),
            ZonedLayoutOptions::default(),
            executor.clone(),
        ));

        // 1. repartition each column to fixed row counts
        let repartition = arcref(RepartitionStrategy::new(
            stats,
            RepartitionWriterOptions {
                // No minimum block size in bytes
                block_size_minimum: 0,
                // Always repartition into 8K row blocks
                block_len_multiple: ROW_BLOCK_SIZE,
            },
        ));

        // 0. start with splitting columns
        arcref(StructStrategy::new(repartition))
    }

    #[cfg(feature = "zstd")]
    pub fn compact_with_executor(
        executor: Arc<dyn TaskExecutor>,
        compressor: vortex_layout::layouts::compact::CompactCompressor,
    ) -> ArcRef<dyn LayoutStrategy> {
        use vortex_layout::layouts::compact::CompactCompressedStrategy;

        // 6. for each chunk create a flat layout
        let chunked = arcref(ChunkedStrategy::default());
        // 5. buffer chunks so they end up with closer segment ids physically
        let buffered = arcref(BufferedStrategy::new(chunked, 2 << 20)); // 2MB
        // 4. compress each chunk
        let compressing = arcref(CompactCompressedStrategy::new(
            buffered,
            executor.clone(),
            16,
            compressor.clone(),
        ));

        // 3. prior to compression, coalesce up to a minimum size
        let coalescing = arcref(RepartitionStrategy::new(
            compressing,
            RepartitionWriterOptions {
                block_size_minimum: 1 << 20,
                block_len_multiple: ROW_BLOCK_SIZE,
            },
        ));

        // 2.1. compress stats tables
        let compress_then_flat = arcref(CompactCompressedStrategy::new(
            arcref(FlatLayoutStrategy::default()),
            executor.clone(),
            1,
            compressor,
        ));

        // TODO: start applying dictionary encoding for variable-length fields
        // when helpful. It is probably best to avoid doing this for small
        // fixed-length fields like numbers.

        // 2. calculate stats for each row group
        let stats = arcref(ZonedStrategy::new(
            coalescing,
            compress_then_flat.clone(),
            ZonedLayoutOptions {
                block_size: ROW_BLOCK_SIZE,
                stats: PRUNING_STATS.into(),
                max_variable_length_statistics_size: 64,
                parallelism: 16,
            },
            executor.clone(),
        ));

        // 1. repartition each column to fixed row counts
        let repartition = arcref(RepartitionStrategy::new(
            stats,
            RepartitionWriterOptions {
                // No minimum block size in bytes
                block_size_minimum: 0,
                // Always repartition into 8K row blocks
                block_len_multiple: ROW_BLOCK_SIZE,
            },
        ));

        // 0. start with splitting columns
        arcref(StructStrategy::new(repartition))
    }
}

fn arcref(item: impl LayoutStrategy) -> ArcRef<dyn LayoutStrategy> {
    ArcRef::new_arc(Arc::new(item))
}

/// Variable-size data is usually really two distinct pieces of data: the variable-sized
/// buffers, and some fixed-width index structures used to know how to seek into the data. We can
/// take advantage of some of these fixed-width index structures to prune reading of some of the
/// fixed-size datasets.
pub struct VariableSizeStrategy;

impl LayoutStrategy for VariableSizeStrategy {
    fn write_stream(
        &self,
        ctx: &ArrayContext,
        sequence_writer: SequenceWriter,
        stream: SendableSequentialStream,
    ) -> SendableLayoutFuture {
        todo!()
    }
}
