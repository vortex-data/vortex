// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt as _;
use parking_lot::Mutex;
use vortex_array::stats::{PRUNING_STATS, Stat};
use vortex_array::{ArrayContext, ArrayRef};
use vortex_error::VortexResult;
use vortex_io::runtime::Handle;

use crate::layouts::zoned::ZonedLayout;
use crate::layouts::zoned::zone_map::StatsAccumulator;
use crate::segments::SegmentSinkRef;
use crate::sequence::{
    SendableSequentialStream, SequenceId, SequencePointer, SequentialArrayStreamExt,
    SequentialStreamAdapter, SequentialStreamExt,
};
use crate::{IntoLayout, LayoutRef, LayoutStrategy};

pub struct ZonedLayoutOptions {
    /// The size of a statistics block
    pub block_size: usize,
    /// The statistics to collect for each block.
    pub stats: Arc<[Stat]>,
    /// Maximum length of a variable length statistics
    pub max_variable_length_statistics_size: usize,
    /// Number of chunks to compute in parallel.
    pub concurrency: usize,
}

impl Default for ZonedLayoutOptions {
    fn default() -> Self {
        Self {
            block_size: 8192,
            stats: PRUNING_STATS.into(),
            max_variable_length_statistics_size: 64,
            concurrency: 16,
        }
    }
}

pub struct ZonedStrategy {
    child: Arc<dyn LayoutStrategy>,
    stats: Arc<dyn LayoutStrategy>,
    options: ZonedLayoutOptions,
}

impl ZonedStrategy {
    pub fn new<Child: LayoutStrategy, Stats: LayoutStrategy>(
        child: Child,
        stats: Stats,
        options: ZonedLayoutOptions,
    ) -> Self {
        Self {
            child: Arc::new(child),
            stats: Arc::new(stats),
            options,
        }
    }
}

#[async_trait]
impl LayoutStrategy for ZonedStrategy {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        eof: SequencePointer,
        handle: Handle,
    ) -> VortexResult<LayoutRef> {
        let stats = self.options.stats.clone();
        let handle2 = handle.clone();

        let stats_accumulator = Arc::new(Mutex::new(StatsAccumulator::new(
            stream.dtype(),
            &stats,
            self.options.max_variable_length_statistics_size,
        )));

        // We can compute per-chunk statistics in parallel, so we spawn tasks for each chunk
        let stream = SequentialStreamAdapter::new(
            stream.dtype().clone(),
            stream
                .map(move |chunk| {
                    let stats = stats.clone();
                    handle2.spawn_cpu(move || {
                        let (sequence_id, chunk) = chunk?;
                        chunk.statistics().compute_all(&stats)?;
                        VortexResult::Ok((sequence_id, chunk))
                    })
                })
                .buffered(self.options.concurrency),
        )
        .sendable();

        // Now we accumulate the stats we computed above, this time we cannot spawn because we
        // need to feed the accumulator an ordered stream.
        let stream = SequentialStreamAdapter::new(
            stream.dtype().clone(),
            stream.scan(stats_accumulator.clone(), |acc, item| {
                future::ready(Some(accumulate_stats(acc, item)))
            }),
        )
        .sendable();

        let block_size = self.options.block_size;

        // We create a new SequencePointer for the stats table so that we can write it just
        // before the end of the file.
        let (data_eof, stats_eof) = eof.split();

        let data_layout = self
            .child
            .write_stream(
                ctx.clone(),
                segment_sink.clone(),
                stream,
                data_eof,
                handle.clone(),
            )
            .await?;

        let Some(stats_table) = stats_accumulator.lock().as_stats_table() else {
            // If we have no stats (e.g. the DType doesn't support them), then we just return the
            // child layout.
            return Ok(data_layout);
        };

        // We must defer creating the stats table LayoutWriter until now, because the DType of
        // the table depends on which stats were successfully computed.
        let (stats_ptr, stats_eof) = stats_eof.split();
        let stats_stream = stats_table.array().to_array_stream().sequenced(stats_ptr);
        let zones_layout = self
            .stats
            .write_stream(ctx, segment_sink.clone(), stats_stream, stats_eof, handle)
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
    // We have already computed per-chunk statistics, so avoid trying again for any that failed.
    stats_accumulator
        .lock()
        .push_chunk_without_compute(&chunk)?;
    Ok((sequence_id, chunk))
}
