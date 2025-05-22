use std::future;
use std::pin::Pin;
 use std::sync::Arc;
 
 use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::once;
use itertools::Itertools;
use parking_lot::Mutex;
use vortex_array::arcref::ArcRef;
use vortex_array::stats::{PRUNING_STATS, Stat, as_stat_bitset_bytes};
use vortex_array::{ArrayContext, ArrayRef};
use vortex_buffer::ByteBufferMut;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::data::Layout;
use crate::layouts::stats::StatsLayout;
use crate::layouts::stats::stats_table::StatsAccumulator;
use crate::segments::{ConcurrentSegmentWriter, NewSegmentWriter};
use crate::sequence::{SequenceId, SequencePointer};
use crate::writer::{LayoutWriter, LayoutWriterExt};
use crate::{LayoutStrategy, LayoutVTableRef};
use crate::{
    LayoutStrategy, LayoutVTableRef, NewLayoutStrategy, NewLayoutWriter, SequentialArrayStream,
};
 
 pub struct StatsLayoutOptions {
     /// The size of a statistics block
     pub block_size: usize,
     /// The statistics to collect for each block.
     pub stats: Arc<[Stat]>,
 }
 
 impl Default for StatsLayoutOptions {
     fn default() -> Self {
         Self {
             block_size: 8192,
             stats: PRUNING_STATS.into(),
         }
     }
 }
 
pub struct NewStatsStrategy {
    child: ArcRef<dyn NewLayoutStrategy>,
    stats: ArcRef<dyn NewLayoutStrategy>,
    options: StatsLayoutOptions,
    end_of_file: Arc<Mutex<SequencePointer>>,
}

impl NewLayoutStrategy for NewStatsStrategy {
    fn write_stream(
        &self,
        ctx: &ArrayContext,
        dtype: &DType,
        segment_writer: Arc<dyn NewSegmentWriter>,
        stream: SequentialArrayStream,
    ) -> Pin<Box<dyn NewLayoutWriter>> {
        let present_stats: Arc<[Stat]> = self.options.stats.iter().sorted().copied().collect();
        let stats_accumulator = Arc::new(Mutex::new(StatsAccumulator::new(
            dtype.clone(),
            &present_stats,
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
            let child_layout = child
                .write_stream(&ctx, &dtype, segment_writer.clone(), stream)
                .await?;

            let Some(stats_table) = stats_accumulator.lock().as_stats_table() else {
                // If we have no stats (e.g. the DType doesn't support them), then we just return the
                // child layout.
                return Ok(child_layout);
            };
            // We must defer creating the stats table LayoutWriter until now, because the DType of
            // the table depends on which stats were successfully computed.
            let stats_array = stats_table.array().clone();

            // if end of file is at x.y, get x.y.0 and advance eof to x.(y + 1)
            let stats_sequence = end_of_file.lock().advance().descend().downgrade();
            let stats_layout = stats_strategy
                .write_stream(
                    &ctx,
                    &dtype,
                    segment_writer,
                    Box::pin(once(async { Ok((stats_sequence, stats_array)) })),
                );

            let mut metadata = ByteBufferMut::empty();

            // First, write the block size to the metadata.
            let block_size = u32::try_from(block_size)?;
            metadata.extend_from_slice(&block_size.to_le_bytes());

            // Then write the bit-set of statistics.
            metadata.extend_from_slice(&as_stat_bitset_bytes(stats_table.present_stats()));

            Ok(Layout::new_owned(
                "stats".into(),
                LayoutVTableRef::new_ref(&StatsLayout),
                dtype.clone(),
                // We report our child data's row count, not the stats table.
                child_layout.row_count(),
                vec![],
                vec![child_layout, stats_layout],
                Some(metadata.freeze().into_inner()),
            ))
        })
    }
}

