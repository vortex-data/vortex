// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem;
use std::ops::DerefMut;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt as _;
use parking_lot::Mutex;
use vortex_array::stats::{PRUNING_STATS, Stat};
use vortex_array::{Array, ArrayContext};
use vortex_error::VortexResult;
use vortex_io::runtime::Handle;

use crate::layouts::dict::bloom::builder::BloomFilterAccumulator;
use crate::layouts::zoned::ZonedLayout;
use crate::layouts::zoned::accumulator::Accumulator;
use crate::layouts::zoned::zone_map::StatsAccumulator;
use crate::segments::SegmentSinkRef;
use crate::sequence::{
    SendableSequentialStream, SequencePointer, SequentialArrayStreamExt, SequentialStreamAdapter,
    SequentialStreamExt,
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

#[derive(Default)]
struct ZonesAccumulator {
    stats: StatsAccumulator,
    bloom_filter: BloomFilterAccumulator,
}

impl ZonesAccumulator {
    pub fn push_chunk(&mut self, chunk: &dyn Array) -> VortexResult<()> {
        self.stats.push_chunk(chunk)?;
        self.bloom_filter.push_chunk(chunk)?;
        Ok(())
    }

    pub fn finish(self) -> (StatsAccumulator, BloomFilterAccumulator) {
        (self.stats, self.bloom_filter)
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
        let stats = self.options.stats.clone();
        let handle2 = handle.clone();

        // We can compute per-chunk statistics in parallel, so we spawn tasks for each chunk
        let dtype = stream.dtype().clone();
        let stream = SequentialStreamAdapter::new(
            dtype.clone(),
            stream
                .map(move |chunk| {
                    let stats = stats.clone();
                    handle2.spawn_cpu(move || {
                        let (sequence_id, chunk) = chunk?;
                        chunk.statistics().compute_all(&stats)?;
                        Ok((sequence_id, chunk))
                    })
                })
                .buffered(self.options.concurrency),
        )
        .sendable();

        // Now we accumulate the stats we computed above, this time we cannot spawn because we
        // need to feed the accumulator an ordered stream.
        // Create an accumulator that is pushed down into chunks to update any configured
        // indices for the zones.
        let zone_accumulator = Arc::new(Mutex::new(ZonesAccumulator {
            stats: StatsAccumulator::new(
                &dtype,
                &self.options.stats,
                self.options.max_variable_length_statistics_size,
            ),
            bloom_filter: BloomFilterAccumulator::default(),
        }));

        let accum = zone_accumulator.clone();

        // Map together the stream of

        let stream = SequentialStreamAdapter::new(
            stream.dtype().clone(),
            stream.map(move |item| {
                let (sequence_id, chunk) = item?;
                // We have already computed per-chunk statistics, so avoid trying again for any that failed.
                accum.lock().push_chunk(&chunk)?;
                Ok((sequence_id, chunk))
            }),
        )
        .sendable();

        // The eof used for the data child should appear _before_ our own stats tables.
        let data_eof = eof.split_off();

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

        // If data was written with dictionary layout, we should write the codes.
        // We should cache the values on write, to make sure it preserves all values correctly.
        // The dictionary should be small enough to avoid spilling out of memory.

        // After stream write completes, the zones accumulator should be unlocked after having
        // witnessed every zone in sequence.
        let zone_accumulator = mem::take(zone_accumulator.lock().deref_mut());

        let (zone_map, bloom_filter) = zone_accumulator.finish();

        // NOTE: we only write bloom filters if we also write zone maps.
        // Maybe we want to separate them, but if building zone maps failed then something is pretty
        // weird here anyway.
        let Some(stats_table) = zone_map.finish()? else {
            // If we have no stats (e.g. the DType doesn't support them), then we just return the
            // child layout.
            return Ok(data_layout);
        };

        let bloom_filter_segment_id = match bloom_filter.finish()? {
            None => None,
            Some(zone_filters) => {
                let mut filters = vec![];
                for filter in zone_filters.into_iter() {
                    // We need to record failures instead.
                    filters.push(filter.serialize());
                }

                // Bloom filter should also appear before the stats table
                let bloom_filter_id = eof.split_off().downgrade();

                Some(segment_sink.write(bloom_filter_id, filters).await?)
            }
        };

        // Write the stats table to a new sequential stream.
        let stats_stream = stats_table
            .array()
            .to_array_stream()
            .sequenced(eof.split_off());

        // Write the stats into the provided child layout.
        let zones_layout = self
            .stats
            .write_stream(ctx, segment_sink.clone(), stats_stream, eof, handle)
            .await?;

        Ok(ZonedLayout::new(
            data_layout,
            zones_layout,
            self.options.block_size,
            stats_table.present_stats().clone(),
            bloom_filter_segment_id,
        )
        .into_layout())
    }

    fn buffered_bytes(&self) -> u64 {
        self.child.buffered_bytes() + self.stats.buffered_bytes()
    }
}
