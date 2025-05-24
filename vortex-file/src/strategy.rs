//! This module defines the default layout strategy for a Vortex file.

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;

use async_stream::try_stream;
use arcref::ArcRef;
use futures::{FutureExt, StreamExt, pin_mut};
use itertools::Itertools;
use vortex_array::arrays::ConstantArray;
use vortex_array::stats::{{PRUNING_STATS, STATS_TO_WRITE}, Stat};
use vortex_array::{Array, ArrayContext, ArrayRef, IntoArray};
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_dtype::DType;
use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex_layout::layouts::dict::writer::DictStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::repartition::{RepartitionStrategy, RepartitionWriterOptions};
use vortex_layout::layouts::stats::writer::{StatsLayoutOptions, StatsStrategy};
use vortex_layout::layouts::struct_::writer::StructStrategy;
use vortex_layout::scan::{TaskExecutor, TaskExecutorExt};
use vortex_layout::layouts::zoned::writer::{ZonedLayoutOptions, ZonedLayoutWriter};
use vortex_layout::segments::{ConcurrentSegmentWriter, NewSegmentWriter};
use vortex_layout::{
    LayoutRef, LayoutStrategy, LayoutWriter, LayoutWriterExt, NewLayoutStrategy, NewLayoutWriter,
    SequentialArrayStream,
};
use vortex_layout::segments::SegmentWriter;
use vortex_layout::sequence::SequencePointer;
use vortex_layout::{LayoutStrategy, LayoutWriter, SequentialArrayStream};

const ROW_BLOCK_SIZE: usize = 8192;

pub struct VortexLayoutStrategy;

impl VortexLayoutStrategy {
    pub fn multi_threaded(
        executor: Arc<dyn TaskExecutor>,
        end_of_file: SequencePointer,
    ) -> ArcRef<dyn LayoutStrategy> {
        // 7. for each chunk create a flat layout
        let chunked = arcref(ChunkedLayoutStrategy::default());
        // 6. buffer chunks so they end up with closer segment ids physically
        let buffered = arcref(BufferedStrategy::new(chunked, 2 << 20)); // 2MB
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
            arcref(FlatLayoutStrategy::default()),
            executor.clone(),
            1,
        ));

        // 3. apply dict encoding or fallback
        let dict = arcref(DictStrategy::new(
            coalescing.clone(),
            compress_then_flat.clone(),
            coalescing,
            Default::default(),
        ));

        // 2. calculate stats for each row group
        let stats = arcref(StatsStrategy::new(
            dict,
            compress_then_flat.clone(),
            StatsLayoutOptions {
                block_size: ROW_BLOCK_SIZE,
                stats: PRUNING_STATS.into(),
                max_variable_length_statistics_size: 64,
            },
            end_of_file,
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

/// A layout writer that compresses chunks using a sampling compressor.
struct BtrBlocksCompressedStrategy {
    child: ArcRef<dyn LayoutStrategy>,
    executor: Arc<dyn TaskExecutor>,
    parallelism: usize,
}

impl BtrBlocksCompressedStrategy {
    pub fn new(
        child: ArcRef<dyn LayoutStrategy>,
        executor: Arc<dyn TaskExecutor>,
        parallelism: usize,
    ) -> Self {
        Self {
            child,
            executor,
            parallelism,
        }
    }
}

impl LayoutStrategy for BtrBlocksCompressedStrategy {
    fn write_stream(
        &self,
        ctx: &ArrayContext,
        dtype: &DType,
        segment_writer: Arc<dyn SegmentWriter>,
        stream: SequentialArrayStream,
    ) -> Pin<Box<dyn LayoutWriter>> {
        let executor = self.executor.clone();

        let stream = stream
            .map(|chunk| {
                async {
                    let (sequence_id, chunk) = chunk?;
                    chunk.statistics().compute_all(STATS_TO_WRITE)?;
                    Ok((sequence_id, BtrBlocksCompressor.compress(&chunk)?))
                }
                .boxed()
            })
            .map(move |compress_future| executor.spawn(compress_future))
            .buffered(self.parallelism);

        self.child
            .write_stream(ctx, dtype, segment_writer, Box::pin(stream))
    }
}

struct BufferedStrategy {
    child: ArcRef<dyn LayoutStrategy>,
    buffer_size: u64,
}

impl BufferedStrategy {
    pub fn new(child: ArcRef<dyn LayoutStrategy>, buffer_size: u64) -> Self {
        Self { child, buffer_size }
    }
}

impl LayoutStrategy for BufferedStrategy {
    fn write_stream(
        &self,
        ctx: &ArrayContext,
        dtype: &DType,
        segment_writer: Arc<dyn SegmentWriter>,
        stream: SequentialArrayStream,
    ) -> Pin<Box<dyn LayoutWriter>> {
        let buffer_size = self.buffer_size;
        let buffered_stream = try_stream! {
            let stream = stream.peekable();
            pin_mut!(stream);

            let mut nbytes = 0u64;
            let mut chunks = VecDeque::new();

            while let Some(chunk) = stream.as_mut().next().await {
                let (sequence_id, chunk) = chunk?;
                nbytes += chunk.nbytes() as u64;
                chunks.push_back(chunk);

                // if this is the last element, flush everything
                if let None = stream.as_mut().peek().await {
                    let mut sequence_pointer = sequence_id.descend();
                    while let Some(chunk) = chunks.pop_front() {
                        yield (sequence_pointer.advance(), chunk)
                    }
                    break;
                }

                if nbytes < 2 * buffer_size {
                    continue;
                };
                // Wait until we're at 2x the buffer size before flushing 1x the buffer size
                // This avoids small tail stragglers being flushed at the end of the file.
                let mut sequence_pointer = sequence_id.descend();
                while nbytes >= 2 * buffer_size {
                    let Some(chunk) = chunks.pop_front() else {
                        break;
                    };
                    nbytes -= chunk.nbytes() as u64;
                    yield (sequence_pointer.advance(), chunk)
                }
            }
        };
        self.child
            .write_stream(&ctx, &dtype, segment_writer, Box::pin(buffered_stream))
    }
}

