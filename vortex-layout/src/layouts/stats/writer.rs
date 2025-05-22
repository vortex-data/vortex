use std::future;
use std::pin::Pin;
 use std::sync::Arc;
 
 use async_trait::async_trait;
use futures::StreamExt;
 use itertools::Itertools;
 use vortex_array::arcref::ArcRef;
 use vortex_array::stats::{PRUNING_STATS, Stat, as_stat_bitset_bytes};
 use vortex_array::{ArrayContext, ArrayRef};
 use vortex_buffer::ByteBufferMut;
 use vortex_dtype::DType;
 use vortex_error::{VortexResult, vortex_bail};
 
 use crate::data::Layout;
 use crate::layouts::stats::StatsLayout;
 use crate::layouts::stats::stats_table::StatsAccumulator;
use crate::segments::ConcurrentSegmentWriter;
use crate::segments::{ConcurrentSegmentWriter, NewSegmentWriter};
use crate::sequence::SequenceId;
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
    options: StatsLayoutOptions,
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
        let stats_accumulator = StatsAccumulator::new(dtype.clone(), &present_stats);
        let stream = Box::pin(stream.scan(stats_accumulator, |acc, item| {
            future::ready(Some(accumulate_stats(acc, item)))
        }));

        let ctx = ctx.clone();
        let dtype = dtype.clone();
        let child = self.child.clone();
        Box::pin(async move {
            let child_layout = child
                .write_stream(&ctx, &dtype, segment_writer, stream)
                .await?;

            // TODO(os): get last sequence_id and write stats layout
            Ok(child_layout)
        })
    }
}

fn accumulate_stats(
    stats_accumulator: &mut StatsAccumulator,
    item: VortexResult<(SequenceId, ArrayRef)>,
) -> VortexResult<(SequenceId, ArrayRef)> {
    let (sequence_id, chunk) = item?;
    stats_accumulator.push_chunk(&chunk)?;
    Ok((sequence_id, chunk))
}

