// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt as _;
use parking_lot::Mutex;
use vortex_array::ArrayContext;
use vortex_array::IntoArray;
use vortex_array::expr::stats::Stat;
use vortex_array::stats::PRUNING_STATS;
use vortex_error::VortexResult;
use vortex_io::runtime::Handle;

use crate::IntoLayout;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::layouts::zoned::ZonedLayout;
use crate::layouts::zoned::zone_map::StatsAccumulator;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialArrayStreamExt;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

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
            concurrency: std::thread::available_parallelism()
                .map(|v| v.get())
                .unwrap_or(1),
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
        mut eof: SequencePointer,
        handle: Handle,
    ) -> VortexResult<LayoutRef> {
        let stats = Arc::clone(&self.options.stats);
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
                    let stats = Arc::clone(&stats);
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
        let stats_accumulator2 = Arc::clone(&stats_accumulator);
        let stream = SequentialStreamAdapter::new(
            stream.dtype().clone(),
            stream.map(move |item| {
                let (sequence_id, chunk) = item?;
                // We have already computed per-chunk statistics, so avoid trying again for any that failed.
                stats_accumulator2
                    .lock()
                    .push_chunk_without_compute(&chunk)?;
                Ok((sequence_id, chunk))
            }),
        )
        .sendable();

        let block_size = self.options.block_size;

        // The eof used for the data child should appear _before_ our own stats tables.
        let data_eof = eof.split_off();
        let data_layout = self
            .child
            .write_stream(
                ctx.clone(),
                Arc::clone(&segment_sink),
                stream,
                data_eof,
                handle.clone(),
            )
            .await?;

        let Some(stats_table) = stats_accumulator.lock().as_stats_table()? else {
            // If we have no stats (e.g. the DType doesn't support them), then we just return the
            // child layout.
            return Ok(data_layout);
        };

        // We must defer creating the stats table LayoutWriter until now, because the DType of
        // the table depends on which stats were successfully computed.
        let stats_stream = stats_table
            .array()
            .clone()
            .into_array()
            .to_array_stream()
            .sequenced(eof.split_off());
        let zones_layout = self
            .stats
            .write_stream(ctx, Arc::clone(&segment_sink), stats_stream, eof, handle)
            .await?;

        Ok(ZonedLayout::new(
            data_layout,
            zones_layout,
            block_size,
            Arc::clone(stats_table.present_stats()),
        )
        .into_layout())
    }

    fn buffered_bytes(&self) -> u64 {
        self.child.buffered_bytes() + self.stats.buffered_bytes()
    }
}
