// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future;
use std::sync::Arc;

use async_trait::async_trait;
use futures::{FutureExt, StreamExt as _};
use parking_lot::Mutex;
use vortex_array::stats::{PRUNING_STATS, Stat};
use vortex_array::{ArrayContext, ArrayRef};
use vortex_error::VortexResult;

use crate::layouts::zoned::ZonedLayout;
use crate::layouts::zoned::zone_map::StatsAccumulator;
use crate::segments::SegmentSink;
use crate::sequence::{SequenceId, SequencePointer};
use crate::{
    ArrayStreamSequentialExt, IntoLayout, LayoutRef, LayoutStrategy, SequentialArrayStream,
    SequentialStreamAdapter, SequentialStreamExt, TaskExecutor, TaskExecutorExt,
};

pub struct ZonedLayoutOptions {
    /// The size of a statistics block
    pub block_size: usize,
    /// The statistics to collect for each block.
    pub stats: Arc<[Stat]>,
    /// Maximum length of a variable length statistics
    pub max_variable_length_statistics_size: usize,
    /// Number of chunks to compute in parallel.
    pub parallelism: usize,
}

impl Default for ZonedLayoutOptions {
    fn default() -> Self {
        Self {
            block_size: 8192,
            stats: PRUNING_STATS.into(),
            max_variable_length_statistics_size: 64,
            parallelism: 16,
        }
    }
}

pub struct ZonedStrategy {
    child: Arc<dyn LayoutStrategy>,
    stats: Arc<dyn LayoutStrategy>,
    options: ZonedLayoutOptions,
    executor: Arc<dyn TaskExecutor>,
    eof: Mutex<SequencePointer>,
}

impl ZonedStrategy {
    pub fn new(
        child: Arc<dyn LayoutStrategy>,
        stats: Arc<dyn LayoutStrategy>,
        options: ZonedLayoutOptions,
        executor: Arc<dyn TaskExecutor>,
        // Pointer to the end of the sequence, used to put stats tables at the back of the file.
        eof: SequencePointer,
    ) -> Self {
        Self {
            child,
            stats,
            options,
            executor,
            eof: Mutex::new(eof),
        }
    }
}

#[async_trait(?Send)]
impl LayoutStrategy for ZonedStrategy {
    async fn write_stream(
        &self,
        ctx: &ArrayContext,
        segment_sink: &dyn SegmentSink,
        stream: SequentialArrayStream,
    ) -> VortexResult<LayoutRef> {
        let executor = self.executor.clone();
        let stats = self.options.stats.clone();
        let precomputed_stream = SequentialStreamAdapter::new(
            stream.dtype().clone(),
            stream
                .map(move |chunk| {
                    let stats = stats.clone();
                    async move {
                        let (sequence_id, chunk) = chunk?;
                        chunk.statistics().compute_all(&stats)?;
                        VortexResult::Ok((sequence_id, chunk))
                    }
                    .boxed()
                })
                .map(move |stats_future| executor.spawn(stats_future))
                .buffered(self.options.parallelism),
        )
        .sendable();

        let stats_accumulator = Arc::new(Mutex::new(StatsAccumulator::new(
            precomputed_stream.dtype(),
            &self.options.stats,
            self.options.max_variable_length_statistics_size,
        )));
        let stream = SequentialStreamAdapter::new(
            precomputed_stream.dtype().clone(),
            precomputed_stream.scan(stats_accumulator.clone(), |acc, item| {
                future::ready(Some(accumulate_stats(acc, item)))
            }),
        )
        .sendable();

        let ctx = ctx.clone();
        let child = self.child.clone();
        let stats_strategy = self.stats.clone();
        let block_size = self.options.block_size;

        let data_layout = child.write_stream(&ctx, segment_sink, stream).await?;

        let Some(stats_table) = stats_accumulator.lock().as_stats_table() else {
            // If we have no stats (e.g. the DType doesn't support them), then we just return the
            // child layout.
            return Ok(data_layout);
        };

        // We must defer creating the stats table LayoutWriter until now, because the DType of
        // the table depends on which stats were successfully computed.
        let stats_stream = stats_table
            .array()
            .to_array_stream()
            // TODO(ngates): we need to fix the API for SequencePointers to make it less error
            //  prone. The order in which this is called is also not deterministic.
            .sequenced(self.eof.lock().advance().descend());

        let zones_layout = stats_strategy
            .write_stream(&ctx, segment_sink, stats_stream)
            .await?;

        Ok(ZonedLayout::new(
            data_layout,
            zones_layout,
            block_size,
            stats_table.present_stats().clone(),
        )
        .into_layout())
    }
}

fn accumulate_stats(
    stats_accumulator: &mut Arc<Mutex<StatsAccumulator>>,
    item: VortexResult<(SequenceId, ArrayRef)>,
) -> VortexResult<(SequenceId, ArrayRef)> {
    let (sequence_id, chunk) = item?;
    stats_accumulator.lock().push_chunk(&chunk)?;
    Ok((sequence_id, chunk))
}
