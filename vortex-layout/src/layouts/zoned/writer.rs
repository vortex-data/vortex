// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::once;
use futures::{FutureExt, StreamExt as _};
use parking_lot::Mutex;
use vortex_array::stats::{PRUNING_STATS, Stat};
use vortex_array::stream::{ArrayStreamAdapter, ArrayStreamExt};
use vortex_array::{ArrayContext, ArrayRef};
use vortex_error::VortexResult;

use crate::layouts::zoned::ZonedLayout;
use crate::layouts::zoned::zone_map::StatsAccumulator;
use crate::segments::SequenceWriter;
use crate::sequence::SequenceId;
use crate::{
    IntoLayout, LayoutRef, LayoutStrategy, SendableSequentialStream, SequentialStreamAdapter,
    SequentialStreamExt, TaskExecutor, TaskExecutorExt,
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

pub struct ZonedStrategy<Child, Stats> {
    child: Child,
    stats: Stats,
    options: ZonedLayoutOptions,
    executor: Arc<dyn TaskExecutor>,
}

impl<Child, Stats> ZonedStrategy<Child, Stats>
where
    Child: LayoutStrategy,
    Stats: LayoutStrategy,
{
    pub fn new(
        child: Child,
        stats: Stats,
        options: ZonedLayoutOptions,
        executor: Arc<dyn TaskExecutor>,
    ) -> Self {
        Self {
            child,
            stats,
            options,
            executor,
        }
    }
}

#[async_trait]
impl<Child, Stats> LayoutStrategy for ZonedStrategy<Child, Stats>
where
    Child: LayoutStrategy,
    Stats: LayoutStrategy,
{
    async fn write_stream(
        &self,
        ctx: &ArrayContext,
        sequence_writer: SequenceWriter,
        stream: SendableSequentialStream,
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
        let block_size = self.options.block_size;
        let data_layout = self
            .child
            .write_stream(&ctx, sequence_writer.clone(), stream)
            .await?;

        let Some(stats_table) = stats_accumulator.lock().as_stats_table() else {
            // If we have no stats (e.g. the DType doesn't support them), then we just return the
            // child layout.
            return Ok(data_layout);
        };
        // We must defer creating the stats table LayoutWriter until now, because the DType of
        // the table depends on which stats were successfully computed.
        let stats_array = stats_table.array().to_array().clone();

        let stats_stream = sequence_writer.new_sequential(ArrayStreamExt::boxed(
            ArrayStreamAdapter::new(stats_array.dtype().clone(), once(async { Ok(stats_array) })),
        ));

        let zones_layout = self
            .stats
            .write_stream(&ctx, sequence_writer, stats_stream)
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
