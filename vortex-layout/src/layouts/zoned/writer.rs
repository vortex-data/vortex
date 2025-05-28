use std::future;
use std::sync::Arc;

use arcref::ArcRef;
use futures::StreamExt as _;
use futures::stream::once;
use itertools::Itertools;
use parking_lot::Mutex;
use vortex_array::stats::{PRUNING_STATS, Stat};
use vortex_array::{ArrayContext, ArrayRef};
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::layouts::zoned::ZonedLayout;
use crate::layouts::zoned::zone_map::StatsAccumulator;
use crate::segments::SegmentWriter;
use crate::sequence::{SequenceId, SequencePointer};
use crate::{IntoLayout, LayoutStrategy, SendableLayoutWriter, SequentialArrayStream};

pub struct ZonedLayoutOptions {
    /// The size of a statistics block
    pub block_size: usize,
    /// The statistics to collect for each block.
    pub stats: Arc<[Stat]>,
    /// Maximum length of a variable length statistics
    pub max_variable_length_statistics_size: usize,
}

impl Default for ZonedLayoutOptions {
    fn default() -> Self {
        Self {
            block_size: 8192,
            stats: PRUNING_STATS.into(),
            max_variable_length_statistics_size: 64,
        }
    }
}

pub struct ZonedStrategy {
    child: ArcRef<dyn LayoutStrategy>,
    stats: ArcRef<dyn LayoutStrategy>,
    options: ZonedLayoutOptions,
    end_of_file: Arc<Mutex<SequencePointer>>,
}

impl ZonedStrategy {
    pub fn new(
        child: ArcRef<dyn LayoutStrategy>,
        stats: ArcRef<dyn LayoutStrategy>,
        options: ZonedLayoutOptions,
        end_of_file: SequencePointer,
    ) -> Self {
        Self {
            child,
            stats,
            options,
            end_of_file: Arc::new(Mutex::new(end_of_file)),
        }
    }
}

impl LayoutStrategy for ZonedStrategy {
    fn write_stream(
        &self,
        ctx: &ArrayContext,
        dtype: &DType,
        segment_writer: Arc<dyn SegmentWriter>,
        stream: SequentialArrayStream,
    ) -> SendableLayoutWriter {
        let present_stats: Arc<[Stat]> = self.options.stats.iter().sorted().copied().collect();
        let stats_accumulator = Arc::new(Mutex::new(StatsAccumulator::new(
            dtype,
            &present_stats,
            self.options.max_variable_length_statistics_size,
        )));
        let stream = Box::pin(stream.scan(stats_accumulator.clone(), |acc, item| {
            future::ready(Some(accumulate_stats(acc, item)))
        }));

        let ctx = ctx.clone();
        let dtype = dtype.clone();
        let child = self.child.clone();
        let stats_strategy = self.stats.clone();
        let block_size = self.options.block_size;
        let end_of_file = self.end_of_file.clone();
        Box::pin(async move {
            let data_layout = child
                .write_stream(&ctx, &dtype, segment_writer.clone(), stream)
                .await?;

            let Some(stats_table) = stats_accumulator.lock().as_stats_table() else {
                // If we have no stats (e.g. the DType doesn't support them), then we just return the
                // child layout.
                return Ok(data_layout);
            };
            // We must defer creating the stats table LayoutWriter until now, because the DType of
            // the table depends on which stats were successfully computed.
            let stats_array = stats_table.array().to_array().clone();

            // if end of file is at x.y, get x.y.0 and advance eof to x.(y + 1)
            let stats_sequence = end_of_file.lock().advance().descend().downgrade();
            let zones_layout = stats_strategy
                .write_stream(
                    &ctx,
                    &stats_array.dtype().clone(),
                    segment_writer,
                    Box::pin(once(async { Ok((stats_sequence, stats_array)) })),
                )
                .await?;

            Ok(ZonedLayout::new(
                data_layout,
                zones_layout,
                block_size,
                stats_table.present_stats().clone(),
            )
            .into_layout())
        })
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
